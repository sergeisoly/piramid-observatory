import * as THREE from 'three/webgpu';
import { attribute, uniform } from 'three/tsl';
import type { Chapter, ChapterFactory, Control, Hotspot, Readout, Result } from '../engine/types';
import { resolveParams } from '../engine/params';
import { C, CSS, type RGB } from '../engine/palette';
import { LineField } from '../kit/lines';
import { MatrixField } from '../kit/shapes';
import { mulberry32, gaussian, smooth, easeInOut, clamp, lerp } from '../lib/math';
import { makeToy, runSOGI, tally, softplus, causalMap, type Run, type ToySpec, type Tally } from './elicitation.sim';

/**
 * CAUSAL ELICITATION, a miniature of the search in Mack, Panickssery and Turner (arXiv:2606.29604).
 * Four slides on one flat figure in the camera's focal plane, authored in pixels of the 1600 × 900
 * frame (y down):
 *
 *   5.1   the paper's procedure as a diagram: training prompts → search without labels → many
 *         perturbations → a behavior metric on validation prompts picks one → held-out test
 *   5.2   the detector bars: one bar per planted detector below its threshold line; one perturbation
 *         aligned with the easiest detector changes its input
 *   5.3   two grids of perturbations against detectors: gradient iteration alone, and with soft
 *         orthogonalization; two perturbations that start alike are followed through both runs. Over each
 *         grid a count strip (perturbations per detector), the number of detectors found, and a bracket
 *         over every run of missed detectors
 *   5.4   the paper's result, quoted: BigCodeBench pass rates of the password-locked Llama-3.3-70B (Table 3)
 *
 * Every number on 5.2 and 5.3 comes from elicitation.sim.ts, run in build() with the current parameters.
 * 5.1 is a diagram (its score bars are drawn, not computed) and 5.4 only quotes the paper.
 */

type V3 = [number, number, number];

const controls: Control[] = [
  { key: 'seed', label: 'seed  slice and initial perturbations', kind: 'range', min: 1, max: 40, step: 1, value: 22, hint: 'Redraws $V$, $W$, $h_0$, $s$ and the random initial perturbations $\\phi_i$.' },
  { key: 'K', label: '$K$  planted detectors', kind: 'range', min: 3, max: 16, step: 1, value: 12, hint: 'Number of hidden directions $v_j$. Each detector gets one column in the alignment matrices.' },
  { key: 'm', label: '$m$  perturbations', kind: 'range', min: 6, max: 48, step: 1, value: 36, hint: 'Too few perturbations can miss detectors. With soft orthogonalization, some candidates can remain below the alignment cutoff for every detector.' },
  { key: 'R', label: '$R$  perturbation radius', kind: 'range', min: 0.25, max: 2, step: 0.05, value: 1, hint: 'Below $R = 1/3$ no perturbation can lift $v_j \\cdot h$ over $\\tau$. The search then usually recovers fewer detectors than at $R = 1$.' },
  { key: 'T', label: '$T$  soft-orthogonalization temperature', kind: 'range', min: 0.25, max: 4, step: 0.25, value: 1, hint: 'Lower $T$ puts more weight on similarity when pushing candidates apart. At equal similarity, neighbors with larger gradients exert more influence.' },
  { key: 'sharp', label: '$k$  detector sharpness', kind: 'range', min: 1, max: 8, step: 0.5, value: 2, hint: '$\\sigma(x) = \\log(1 + e^{kx})/k$. Sharper responses leave less gradient below $\\tau$. This can make some detectors harder to find.' },
];

// ── timeline (chapter seconds): one beat per slide ──
const BEAT = [0.8, 4.6, 8.4, 14.8];
const DUR = 19.4;
const D = 32, ITERS = 30, INNER = 10, THR = 0.9;

// ── the figure plane: focal plane of a frontal reference camera, authored in pixels ──
const REF_Z = 24, REF_FOV = 38;
const WPP = (REF_Z * Math.tan((REF_FOV * Math.PI) / 360)) / 450; // world units per pixel
const Wp = (x: number, y: number): V3 => [(x - 800) * WPP, (409.5 - y) * WPP, 0]; // 409.5: the engine shifts the lens up by 4.5 %
const U = (y: number) => 900 - y; // screen y (down) → figure y (up)
const unWp = (q: [number, number]): [number, number] => [q[0] / WPP + 800, 409.5 - q[1] / WPP];
const P2 = (x: number, y: number): V3 => [x, U(y), 0];

// layout (pixels)
const CY0 = 262, CY1 = 540; // value range of every bar chart, top and bottom
const MAIN = { x0: 420, x1: 860 }; // slide 5.2
const GP = { x: 236 }, GS = { x: 610 }, GY = 276, CW = 18; // the two grids of slide 5.3
const GRID_MAX_H = 288, ROW_MAX_H = 8; // keep up to 48 candidate rows inside the authored frame
const STRIP0 = GY - 8, STRIP_H = 44; // the count strips over the grids: baseline and largest height
// slide 5.1, the procedure: rows of drawn perturbations and their drawn validation scores (a diagram: made-up lengths)
const PIPE_SCORE = [0.34, 0.55, 0.22, 0.42, 0.28, 0.9, 0.38, 0.5, 0.18, 0.31];
const PIPE_PICK = 5;
const PY0 = 282, PRH = 16; // first row, row pitch
// slide 5.4, the paper's Table 3 (BigCodeBench greedy pass rate, held-out problems; Llama-3.3-70B organism of Taylor et al.)
const T3 = [
  { v: 0.43, s: '0.43', name: 'locked', sub: 'no password' },
  { v: 0.52, s: '0.52', name: 'CPE edit', sub: 'no password' },
  { v: 0.54, s: '0.54', name: 'unlocked', sub: 'with the password' },
];
const RX0 = 480, RX1 = 900, RY0 = 560, RYH = 300, RVMAX = 0.6; // chart frame: left, right, baseline, height, top value
const rY = (v: number) => RY0 - (v / RVMAX) * RYH;
const rBX = (b: number) => RX0 + 80 + b * 125; // center of bar b

/** A number as the page prints it: at most two decimals, no trailing zeros. */
const fmt = (x: number) => (isFinite(x) ? String(+x.toFixed(2)).replace('-', '−') : '—');
const num = (x: number, d = 2) => (isFinite(x) ? x.toFixed(d).replace('-', '−') : '—');
const SMALL = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty'];
/** A count in words, as running text writes it. */
const words = (n: number) => (n < 20 ? SMALL[n] : n < 50 ? TENS[Math.floor(n / 10)] + (n % 10 ? '-' + SMALL[n % 10] : '') : String(n));
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** Filled axis-aligned rectangles in the figure plane (bars). Same color semantics as LineField. */
class QuadField {
  readonly positions: Float32Array;
  readonly colors: Float32Array;
  readonly object: THREE.Mesh;
  readonly uIntensity = uniform(1);
  private geo = new THREE.BufferGeometry();
  private pA: THREE.BufferAttribute;
  private cA: THREE.BufferAttribute;
  constructor(readonly max: number) {
    this.positions = new Float32Array(max * 12);
    this.colors = new Float32Array(max * 12);
    const idx = new Uint32Array(max * 6);
    for (let i = 0; i < max; i++) idx.set([4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3], i * 6);
    this.pA = new THREE.BufferAttribute(this.positions, 3);
    this.cA = new THREE.BufferAttribute(this.colors, 3);
    this.pA.setUsage(THREE.DynamicDrawUsage);
    this.cA.setUsage(THREE.DynamicDrawUsage);
    this.geo.setAttribute('position', this.pA);
    this.geo.setAttribute('color', this.cA);
    this.geo.setIndex(new THREE.BufferAttribute(idx, 1));
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = (attribute('color', 'vec3') as any).mul(this.uIntensity);
    m.transparent = true;
    m.depthWrite = false;
    m.blending = THREE.AdditiveBlending;
    m.side = THREE.DoubleSide;
    this.object = new THREE.Mesh(this.geo, m);
    this.object.frustumCulled = false;
    this.count = 0;
  }
  set count(n: number) {
    const c = Math.max(0, Math.min(this.max, Math.floor(n)));
    this.geo.setDrawRange(0, c * 6);
    this.object.visible = c > 0;
  }
  /** Rectangle between screen points (x0, y0) and (x1, y1). */
  quad(i: number, x0: number, y0: number, x1: number, y1: number, c: RGB, k = 1) {
    const p = this.positions, q = this.colors, j = i * 12, a = U(y0), b = U(y1);
    p.set([x0, a, 0, x1, a, 0, x1, b, 0, x0, b, 0], j);
    for (let v = 0; v < 4; v++) { q[j + v * 3] = c[0] * k; q[j + v * 3 + 1] = c[1] * k; q[j + v * 3 + 2] = c[2] * k; }
  }
  commit() {
    this.pA.needsUpdate = true;
    this.cA.needsUpdate = true;
  }
}

interface Track {
  run: Run;
  n: number; // snapshots
  tl: Tally[]; // per snapshot
  rows: number[]; // display row → candidate, ordered by final assignment
}

const elicitation: ChapterFactory = {
  id: 'elicitation',
  label: 'elicitation · a search over weight perturbations',
  controls,
  async build(ctx, params) {
    const P = resolveParams(controls, params);
    const SEED = Math.round(P.seed), K = Math.round(P.K), M = Math.round(P.m);
    const CH = Math.min(ROW_MAX_H, GRID_MAX_H / M);
    const RAD: number = P.R, TEMP: number = P.T, SHARP: number = P.sharp;

    // ═════════════ the computation ═════════════
    const spec: ToySpec = { d: D, K, tau: 1, sharp: SHARP, gain: 5, sNorm: 3, R: RAD, h0Norm: 2, offLo: -1, offHi: 0 };
    const rnd = mulberry32(SEED);
    const toy = makeToy(rnd, gaussian(rnd), spec);
    const mkRun = (repel: boolean) =>
      runSOGI(toy, gaussian(mulberry32(SEED * 7919 + 13)), { m: M, iters: ITERS, repel, T: TEMP, inner: INNER });
    const runS = mkRun(true); // candidates pushed apart (gradient iteration + soft orthogonalization)
    const runP = mkRun(false); // gradient steps alone, same initialization
    await ctx.log('both searches complete from the same initial candidates');

    // columns and bars: detectors ordered by margin τ − v_j·h₀, easiest (smallest) first
    const depth = Array.from(toy.pre0, (x) => -x);
    const colOrd = Array.from({ length: K }, (_, j) => j).sort((a, b) => depth[a] - depth[b]); // display column → detector
    const colPos = new Array<number>(K);
    colOrd.forEach((j, c) => (colPos[j] = c));

    const mkTrack = (run: Run): Track => {
      const tl = run.snaps.map((_, s) => tally(run, K, s, THR)), end = tl[tl.length - 1], last = run.snaps[run.snaps.length - 1];
      const key = (i: number) => {
        let mx = -2, arg = 0;
        for (let j = 0; j < K; j++) if (last.cos[i * K + j] > mx) { mx = last.cos[i * K + j]; arg = j; }
        return end.owner[i] >= 0 ? colPos[end.owner[i]] * 10 - mx : mx > 0.3 ? 1000 + colPos[arg] * 10 - mx : 5000 - mx;
      };
      return { run, n: run.snaps.length, tl, rows: Array.from({ length: M }, (_, i) => i).sort((a, b) => key(a) - key(b)) };
    };
    const trS = mkTrack(runS), trP = mkTrack(runP);
    const endS = trS.tl[trS.n - 1], endP = trP.tl[trP.n - 1];
    // Detectors matched by an aligned weight change that also raises their input above threshold.
    const foundAboveThreshold = (run: Run, end: Tally) => {
      const u = new Float64Array(D), pre = new Float64Array(K), act = new Float64Array(K);
      let n = 0;
      for (let j = 0; j < K; j++) {
        let on = false;
        for (let i = 0; i < M && !on; i++) if (end.owner[i] === j) { causalMap(toy, run.A, run.B, i * D, u, pre, act); on = pre[j] > 0; }
        if (on) n++;
      }
      return n;
    };
    const aboveS = foundAboveThreshold(runS, endS), aboveP = foundAboveThreshold(runP, endP);

    // slide 5.3 follows two candidates through both runs (they start from the same edits in both). Wanted: a pair
    // that, with candidates pushed apart, leans toward one detector in the first iterations and then splits, and
    // that with gradient steps alone ends on that one detector. 'stay' keeps the detector, 'leave' moves on.
    const missedSet = new Set(colOrd.filter((j) => endP.counts[j] === 0));
    const argmaxAt = (cos: Float32Array, i: number) => { let b = 0; for (let j = 1; j < K; j++) if (cos[i * K + j] > cos[i * K + b]) b = j; return b; };
    const follow = (() => {
      let best: { stay: number; leave: number; j0: number; jL: number; itSplit: number } | null = null, bestKey = -Infinity;
      for (let i = 0; i < M; i++)
        for (let k = 0; k < M; k++) {
          if (i === k) continue;
          const j0 = endP.owner[i];
          if (j0 < 0 || endP.owner[k] !== j0 || endS.owner[i] !== j0 || endS.owner[k] < 0 || endS.owner[k] === j0) continue;
          let lean = -1, itAlike = -1;
          for (let it = 1; it <= 8; it++) {
            const c = runS.snaps[it * runS.sub].cos;
            if (argmaxAt(c, i) !== j0 || argmaxAt(c, k) !== j0) continue;
            const m = Math.min(c[i * K + j0], c[k * K + j0]);
            if (m > lean) { lean = m; itAlike = it; }
          }
          if (lean < 0.25) continue;
          let itSplit = itAlike + 1; // first later iteration at which 'leave' points elsewhere
          while (itSplit < ITERS && argmaxAt(runS.snaps[itSplit * runS.sub].cos, k) === j0) itSplit++;
          const key = lean + (missedSet.has(endS.owner[k]) ? 1 : 0);
          if (key > bestKey) { bestKey = key; best = { stay: i, leave: k, j0, jL: endS.owner[k], itSplit }; }
        }
      return best;
    })();
    // the grids run slowly through the first iterations, where the candidates sort themselves out
    const SLOW = follow ? clamp(follow.itSplit + 3, 6, 14) : 8;

    // slide 5.3: what the gradient-only run misses and where it crowds
    const missedCols = colOrd.map((j, c) => (endP.counts[j] === 0 ? c : -1)).filter((c) => c >= 0);
    const nMiss = missedCols.length;
    const missHardest = nMiss > 0 && missedCols.every((c) => c >= K - nMiss); // the missed ones are exactly the hardest
    const pileEasiest = endP.counts[colOrd[0]];
    const theHardest = nMiss === 1 ? 'the hardest' : `the ${words(nMiss)} hardest`; // what the gradient-only run misses, when it misses exactly the hardest

    // The bar chart follows the final pushed-apart candidate most aligned with the easiest detector.
    const preFull = (() => {
      const last = runS.snaps[runS.snaps.length - 1], e = colOrd[0];
      let best = 0;
      for (let i = 0; i < M; i++) if (last.cos[i * K + e] > last.cos[best * K + e]) best = i;
      const u = new Float64Array(D), pre = new Float64Array(K), act = new Float64Array(K);
      causalMap(toy, runS.A, runS.B, best * D, u, pre, act);
      return pre;
    })();
    // Check how much the edit also shifts the other detectors and whether all those shifts are negative.
    const sideOf = (pre: Float64Array) => {
      let mx = 0, down = true;
      for (let c = 1; c < K; c++) { const dl = pre[colOrd[c]] - toy.pre0[colOrd[c]]; mx = Math.max(mx, Math.abs(dl)); if (dl >= 0) down = false; }
      return mx < 0.02 ? null : down ? 'lower' : 'move';
    };
    const sideFull = sideOf(preFull);
    // the 60 runs checked outside the film (seeds 1 to 60, which include the Explore range) used these settings only
    const AUDITED = K === 12 && M === 36 && RAD === 1 && TEMP === 1 && SHARP === 2;

    // Choose the hardest detector from the unmodified toy, independently of either search's outcome.
    const kStar = colOrd[K - 1];
    const NVAL = 8, NTEST = 32;
    const gv = gaussian(mulberry32(SEED * 31 + 5));
    const prS = new Float64Array((NVAL + NTEST) * D), prPre = new Float64Array(NVAL + NTEST);
    for (let p = 0; p < NVAL + NTEST; p++) {
      let pre = -spec.tau;
      for (let c = 0; c < D; c++) {
        prS[p * D + c] = toy.s[c] + 0.12 * gv();
        pre += toy.V[kStar * D + c] * (toy.h0[c] + 0.05 * gv());
      }
      prPre[p] = pre;
    }
    const mu = (A: Float64Array | null, B: Float64Array | null, o: number, lo: number, hi: number) => {
      let acc = 0;
      for (let p = lo; p < hi; p++) {
        let as = 0, vb = 0;
        if (A && B) for (let c = 0; c < D; c++) { as += A[o + c] * prS[p * D + c]; vb += toy.V[kStar * D + c] * B[o + c]; }
        acc += softplus(prPre[p] + RAD * as * vb, SHARP);
      }
      return acc / (hi - lo);
    };
    const validationWinner = (run: Run) => {
      let bestScore = -Infinity, best = 0;
      for (let i = 0; i < M; i++) {
        const score = mu(run.A, run.B, i * D, 0, NVAL);
        if (score > bestScore) { bestScore = score; best = i; }
      }
      return mu(run.A, run.B, best * D, NVAL, NVAL + NTEST);
    };
    const vStar = toy.V.slice(kStar * D, kStar * D + D);
    const muSoft = validationWinner(runS), muGradient = validationWinner(runP);
    const muNone = mu(null, null, 0, NVAL, NVAL + NTEST);
    const muOracle = mu(toy.sHat, vStar, 0, NVAL, NVAL + NTEST);
    await ctx.log('selection ✓  drawing the figure');

    // ═════════════ the figure ═════════════
    const group = new THREE.Group();
    const fig = new THREE.Group();
    const o = Wp(0, 900);
    fig.position.set(o[0], o[1], o[2]);
    fig.scale.setScalar(WPP);
    group.add(fig);
    const INK = C.white;
    const G = 0.78; // MatrixField ink is 1.8 v²: v = G·x gives 1.1 x²

    // Bar lengths measure signed distance from the detector threshold.
    let vMin = 0, vMax = 0;
    for (let j = 0; j < K; j++) { vMin = Math.min(vMin, toy.pre0[j], preFull[j]); vMax = Math.max(vMax, preFull[j]); }
    const vLo = Math.floor((vMin - 0.35) * 2) / 2, vHi = Math.max(0.8, vMax + 0.5);
    const tickStep = Math.max(1, Math.ceil((vHi - vLo) / 6));
    const vY = (v: number) => CY1 - ((v - vLo) / (vHi - vLo)) * (CY1 - CY0);
    const slot = (g: { x0: number; x1: number }) => (g.x1 - g.x0) / K;
    const barX = (g: { x0: number; x1: number }, c: number) => g.x0 + (c + 0.5) * slot(g);
    const bars = new QuadField(K * 3 + 2 * PIPE_SCORE.length + 8);
    const lines = new LineField(K * 8 + 60);
    const pipe = new LineField(160); // slides 5.1 and 5.4: boxes, arrows, brackets, axes
    // the grids
    const GW = K * CW, GH = M * CH;
    const mkMatrix = (rows: number, cols: number, cx: number, cy: number, pos: RGB) => {
      const m = new MatrixField({ rows, cols, cell: 1, gap: 0.12, height: 0, neg: C.slate, pos });
      m.object.scale.set(CW, 1, CH);
      m.object.rotation.x = Math.PI / 2;
      m.object.position.set(cx, U(cy), 0);
      fig.add(m.object);
      return m;
    };
    const ABL = INK.map((v) => v * 0.55) as RGB; // gradient steps alone: key ink
    const heatP = mkMatrix(M, K, GP.x + GW / 2, GY + GH / 2, ABL);
    const heatS = mkMatrix(M, K, GS.x + GW / 2, GY + GH / 2, C.amber);
    const gridLines = new LineField(32);
    for (const x0 of [GP.x, GS.x]) gridLines.polyline([P2(x0 - 2, GY - 2), P2(x0 + GW + 2, GY - 2), P2(x0 + GW + 2, GY + GH + 2), P2(x0 - 2, GY + GH + 2)], INK, 0.45, true);
    // a bracket over every run of neighboring missed columns, in each grid (the end of the search decides them)
    const runsOf = (cols: number[]) => {
      const out: [number, number][] = [];
      for (const c of cols) { const last = out[out.length - 1]; if (last && c === last[1] + 1) last[1] = c; else out.push([c, c]); }
      return out;
    };
    const missedColsS = colOrd.map((j, c) => (endS.counts[j] === 0 ? c : -1)).filter((c) => c >= 0);
    const missRuns: [number, [number, number][]][] = [[GP.x, runsOf(missedCols)], [GS.x, runsOf(missedColsS)]];
    const missY = STRIP0 - 6;
    const missLines = new LineField(3 * 2 * K + 3);
    for (const [x0, rs] of missRuns)
      for (const [c0, c1] of rs) {
        const xa = x0 + c0 * CW + 3, xb = x0 + (c1 + 1) * CW - 3;
        missLines.push(P2(xa, missY + 5), P2(xa, missY), INK, 0.9);
        missLines.push(P2(xa, missY), P2(xb, missY), INK, 0.9);
        missLines.push(P2(xb, missY), P2(xb, missY + 5), INK, 0.9);
      }
    // a bracket beside the right grid over its last rows: the perturbations that end on no detector
    if (endS.idle > 0) {
      const xI = GS.x + GW + 12, y0 = GY + (M - endS.idle) * CH + 1, y1 = GY + GH - 1;
      missLines.push(P2(xI - 5, y0), P2(xI, y0), INK, 0.9);
      missLines.push(P2(xI, y0), P2(xI, y1), INK, 0.9);
      missLines.push(P2(xI, y1), P2(xI - 5, y1), INK, 0.9);
    }
    gridLines.commit(); missLines.commit();
    // the count strips over the grids: one bar per detector, as tall as the perturbations that point at it
    let pileMax = 1;
    for (const tr of [trP, trS]) for (const tl of tr.tl) pileMax = Math.max(pileMax, tl.maxPile);
    const stripPx = Math.min(4, STRIP_H / pileMax); // pixels per perturbation
    const strip = new QuadField(2 * K);
    // one "missed" over each bracket; brackets close enough for their words to touch share one word, and a word
    // rises above any count number it would overlap
    const MISS_W = 62; // allow the word to grow relative to the plot in a narrow viewport
    const missLabels: [number, number][] = [];
    for (const [g, [x0, rs]] of missRuns.entries()) {
      const end = (g === 0 ? trP : trS).tl[(g === 0 ? trP : trS).n - 1];
      const groups: [number, number][] = []; // pixel spans of the brackets that share a word
      for (const [c0, c1] of rs) {
        const xa = x0 + c0 * CW, xb = x0 + (c1 + 1) * CW, last = groups[groups.length - 1];
        if (last && (xa + xb) / 2 - MISS_W / 2 < (last[0] + last[1]) / 2 + MISS_W / 2) last[1] = xb;
        else groups.push([xa, xb]);
      }
      for (const [xa, xb] of groups) {
        const cx = (xa + xb) / 2;
        let y = missY - 2;
        for (let c = 0; c < K; c++) {
          const n = end.counts[colOrd[c]], xc = x0 + (c + 0.5) * CW;
          if (n > 0 && Math.abs(xc - cx) < MISS_W / 2 + 10) y = Math.min(y, STRIP0 - n * stripPx - 13);
        }
        missLabels.push([cx, y]);
      }
    }
    // slide 5.3: an outline around the rows of the two candidates the figure follows, in both grids
    const pairLines = new LineField(16, { width: 2 });
    fig.add(bars.object, lines.object, pipe.object, gridLines.object, missLines.object, pairLines.object, strip.object);
    const FOLLOW = C.cyan, FOLLOW_CSS = CSS.cyan;

    // ═════════════ timeline ═════════════
    const [B1, B2, B3, B4] = BEAT;
    // 5.1: the procedure, drawn station by station from left to right
    const aPipe = (t: number) => smooth(B1 - 0.4, B1 + 0.1, t) * (1 - smooth(B2, B2 + 0.5, t));
    const stage = (t: number, k: number) => smooth(B1 + 0.1 + 0.55 * k, B1 + 0.55 + 0.55 * k, t) * aPipe(t);
    const scoreGrow = (t: number) => easeInOut(smooth(B1 + 1.75, B1 + 2.5, t));
    // 5.2: the detector bars
    const aMain = (t: number) => smooth(B2 + 0.2, B2 + 0.8, t) * (1 - smooth(B3, B3 + 0.5, t));
    const grow = (t: number) => easeInOut(smooth(B2 + 0.4, B2 + 1.4, t));
    const aEdit = (t: number) => smooth(B2 + 1.5, B2 + 2.0, t); // tick, gap and the edit's label
    const push = (t: number) => easeInOut(smooth(B2 + 1.9, B2 + 3.0, t));
    // 5.3: the grids
    const aGrid = (t: number) => smooth(B3 + 0.3, B3 + 0.9, t) * (1 - smooth(B4, B4 + 0.5, t));
    const G_RUN = [B3 + 1.1, B3 + 5.1]; // the grids run through the search's iterations
    // 5.4: the paper's result
    const aRes = (t: number) => smooth(B4 + 0.3, B4 + 0.9, t);
    const resGrow = (t: number, b: number) => easeInOut(smooth(B4 + 0.3 + 0.22 * b, B4 + 1.4 + 0.22 * b, t));
    /** Fraction of the iterations on display: the first SLOW iterations take 80% of the run. */
    const gridAt = (t: number) => {
      const u = clamp((t - G_RUN[0]) / (G_RUN[1] - G_RUN[0]));
      return (u < 0.8 ? (SLOW * u) / 0.8 : SLOW + ((ITERS - SLOW) * (u - 0.8)) / 0.2) / ITERS;
    };
    const aFollow = (t: number) => smooth(G_RUN[0] - 0.4, G_RUN[0] + 0.1, t) * aGrid(t);
    const finished = (t: number) => smooth(G_RUN[1] + 0.1, G_RUN[1] + 0.6, t);
    const aMiss = (t: number) => finished(t) * aGrid(t);

    const iterationAt = (t: number) => Math.round((gridAt(t) || 0) * ITERS);
    const drawHeat = (tr: Track, heat: MatrixField, iteration: number, alpha: number) => {
      const cos = tr.run.snaps[iteration * tr.run.sub].cos;
      heat.uOpacity.value = alpha;
      for (let r = 0; r < M; r++) {
        const i = tr.rows[r];
        for (let c = 0; c < K; c++) heat.values[r * K + c] = G * Math.max(0, cos[i * K + colOrd[c]]); // ink ∝ cos² for positive alignments
      }
      heat.commit();
    };

    /** Bars of one chart into quads q0…, its line work into `lines` from segment n; returns the next segment. */
    const drawChart = (g: { x0: number; x1: number }, q0: number, n: number, pre: ArrayLike<number> | null, w: number, gr: number, tick: number, alpha: number) => {
      const s = slot(g);
      for (let c = 0; c < K; c++) {
        const j = colOrd[c], v = pre ? lerp(toy.pre0[j], pre[j], w) : toy.pre0[j];
        const endpoint = vY(v * gr), on = v > 0 && gr > 0.99;
        bars.quad(q0 + c, barX(g, c) - s * 0.3, vY(0), barX(g, c) + s * 0.3, endpoint, on ? C.amber : INK, (on ? 1.2 : 0.42) * alpha);
        if (tick > 0.01) lines.seg(n++, P2(barX(g, c) - s * 0.48, vY(toy.pre0[j])), P2(barX(g, c) + s * 0.48, vY(toy.pre0[j])), INK, 1.1 * tick * alpha);
      }
      lines.seg(n++, P2(g.x0 - 6, vY(0)), P2(g.x1 + 10, vY(0)), INK, 1.8 * alpha); // the threshold line
      return n;
    };

    // 5.3 in the order of the figure: the iteration, its pile-up and misses, the push, what the push finds; every
    // clause that depends on the run picks its branch here (audited over the Explore range, see the notes)
    const harderHalf = nMiss > 0 && missedCols.every((c) => c >= K / 2); // every detector the control misses lies in the harder half
    const missPhrase = missHardest ? theHardest : harderHalf ? `${words(nMiss)} in the harder half` : words(nMiss);
    const allOrSome = (n: number) => (n === K ? `all ${words(K)}` : `${words(n)} of the ${words(K)}`);
    const cap3 =
      'Gradient iteration turns candidates toward larger changes in the later layer. ' +
      (endS.found > endP.found
        ? (pileEasiest >= 2 && pileEasiest === endP.maxPile ? `Left alone, it crowds the easiest detector and misses ${missPhrase}. ` : `Left alone, it misses ${missPhrase}. `)
        : endS.found === endP.found
          ? `It finds ${allOrSome(endP.found)} detectors. `
          : `It finds ${allOrSome(endP.found)} detectors, the higher count in this run. `) +
      'Soft orthogonalization pushes similar candidates apart after each iteration. ' +
      `With this step, the search finds ${allOrSome(endS.found)} detectors in this toy draw.`;

    const results: Result[] = [
      {
        label: 'detectors matched and raised above threshold, with soft orthogonalization',
        measured: `${aboveS} of ${K}`,
        reference: `planted: ${K}`,
        headline: true,
        group: 'check',
        kind: 'planted structure',
        claim: 'a weight change found without behavior labels aligns with a planted detector and lifts its input above threshold',
        say: aboveS === K ? 'Every detector has an aligned candidate that lifts its input above threshold.' : `${aboveS} of the ${K} detectors have an aligned candidate that lifts their input above threshold.`,
      },
      {
        label: 'detectors matched and raised above threshold, gradient iteration alone',
        measured: `${aboveP} of ${K}`,
        reference: `planted: ${K}`,
        group: 'check',
        kind: 'planted structure',
        say: `With gradient iteration alone, ${aboveP} of ${K} detectors have an aligned candidate that lifts their input above threshold.`,
      },
    ];

    const chapter: Chapter = {
      id: 'elicitation',
      pillar: 'tools',
      kicker: 'INTERPRETABILITY APPLICATIONS',
      title: 'CAUSAL ELICITATION',
      source: 'Mack, Panickssery, Turner · Mechanistically Eliciting Latent Behaviors · arXiv:2606.29604',
      duration: DUR,
      group,
      results,
      beats: [
        {
          at: B1,
          caption:
            'Mack and colleagues look for weight changes that could reveal a hidden behavior. Their method, Causal Perturbative Elicitation (CPE), searches for fixed-size edits with large, consistent effects in a later layer. ' +
            'The search uses no behavior labels. The researchers then use validation scores to choose an edit to test on new prompts.',
        },
        {
          at: B2,
          caption:
            `To evaluate the search, this experiment plants ${words(K)} detectors whose ideal weight changes are known. ` +
            'A candidate <em>finds</em> a detector when its weight change closely aligns with the ideal change. A large enough change can lift the detector above threshold and amplify its downstream signal.',
        },
        {
          at: B3,
          caption: cap3,
        },
        {
          at: B4,
          caption:
            'Taylor and colleagues fine-tuned a 70B model to underperform at coding without a password. On held-out problems, CPE’s selected edit recovers most of the pass-rate gap. ' +
            'Size-matched random edits barely changed scores in separate 8B tests. Which features the unedited model uses to enforce the lock or solve the coding problems remains open.',
        },
      ],
      shots: [
        { t: 0, pos: [0, 0, REF_Z], look: [0, 0, 0], fov: REF_FOV },
        { t: DUR, pos: [0, 0, REF_Z], look: [0, 0, 0], fov: REF_FOV },
      ],

      update(t, api) {
        const am = aMain(t), ag = aGrid(t), ap = aPipe(t), ar = aRes(t);
        const iteration = iterationAt(t);
        const tallyP = trP.tl[iteration * runP.sub], tallyS = trS.tl[iteration * runS.sub];
        const lab = (key: string, x: number, y: number, html: string, alpha: number, align: 'left' | 'center' | 'right' | 'below' | 'middle' = 'left', color: string = CSS.white, size: 'sm' | 'md' | 'xs' = 'sm', help?: string) => {
          if (alpha > 0.02) api.label(key, Wp(x, y), html, { alpha, color, align, size, help });
        };
        const box = (x0: number, y0: number, x1: number, y1: number, c: RGB, k: number) => pipe.polyline([P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1)], c, k, true);
        const arrow = (x0: number, x1: number, y: number, c: RGB, k: number) => {
          pipe.push(P2(x0, y), P2(x1, y), c, k);
          pipe.push(P2(x1 - 7, y - 5), P2(x1, y), c, k);
          pipe.push(P2(x1 - 7, y + 5), P2(x1, y), c, k);
        };
        const bracket = (x0: number, x1: number, y: number, c: RGB, k: number) => {
          pipe.push(P2(x0, y + 7), P2(x0, y), c, k);
          pipe.push(P2(x0, y), P2(x1, y), c, k);
          pipe.push(P2(x1, y), P2(x1, y + 7), c, k);
        };
        pipe.count = 0;

        // ── bars (5.2), and the drawn rows of 5.1 and the quoted bars of 5.4 share the quads ──
        let n = 0;
        let nq = 0;
        if (am > 0.004) {
          n = drawChart(MAIN, 0, n, preFull, push(t), grow(t), aEdit(t), am);
          nq = K;
          const xAxis = MAIN.x0 - 6;
          lines.seg(n++, P2(xAxis, CY0), P2(xAxis, CY1), INK, 0.55 * am);
          for (let v = Math.ceil(vLo / tickStep) * tickStep; v <= vHi; v += tickStep) {
            lines.seg(n++, P2(xAxis - 4, vY(v)), P2(xAxis, vY(v)), INK, 0.75 * am);
            lab(`b-tick${v}`, xAxis - 6, vY(v), String(v).replace('-', '−'), am, 'right');
          }
        }

        // ── 5.1: the paper's procedure, as a diagram ──
        const NR = PIPE_SCORE.length, rowY = (r: number) => PY0 + r * PRH;
        const yPick = rowY(PIPE_PICK) + 5;
        if (ap > 0.004) {
          const s0 = stage(t, 0), s1 = stage(t, 1), s2 = stage(t, 2), s3 = stage(t, 3), s4 = stage(t, 4);
          // training prompts: three cards with two lines of text each
          for (let q = 0; q < 3; q++) {
            const y = 312 + q * 40;
            box(218, y, 318, y + 28, INK, 0.8 * s0);
            pipe.push(P2(228, y + 10), P2(300, y + 10), INK, 0.35 * s0);
            pipe.push(P2(228, y + 18), P2(282, y + 18), INK, 0.35 * s0);
          }
          arrow(330, 385, 360, INK, 0.8 * s1);
          // the model: layers stacked bottom (early) to top (late); the perturbation sits low, the change is read high
          box(398, 272, 578, 450, INK, 0.8 * s1);
          for (let q = 0; q < 6; q++) {
            const y = 296 + q * 28, hot = q === 4, read = q === 0;
            pipe.push(P2(412, y), P2(564, y), INK, (hot || read ? 1.5 : 0.3) * s1);
          }
          arrow(590, 638, 360, INK, 0.8 * s2);
          // the perturbations the search keeps: one row each
          for (let r = 0; r < NR; r++) bars.quad(nq++, 650, rowY(r), 700, rowY(r) + 10, C.slate, 1.0 * s2);
          // their validation scores, grown; the highest is picked
          for (let r = 0; r < NR; r++) {
            const w = 190 * PIPE_SCORE[r] * scoreGrow(t), pk = r === PIPE_PICK && t > B1 + 2.5;
            if (w > 0.5) bars.quad(nq++, 720, rowY(r), 720 + w, rowY(r) + 10, pk ? C.amber : C.slate, (pk ? 1.2 : 0.55) * s3);
          }
          pipe.push(P2(720, rowY(0) - 6), P2(720, rowY(NR - 1) + 16), INK, 0.6 * s3);
          arrow(952, 1000, yPick, C.amber, 1.0 * s4);
          // the held-out test: the picked perturbation alone
          bars.quad(nq++, 1022, yPick - 5, 1072, yPick + 5, C.slate, 1.0 * s4);
          box(1012, yPick - 36, 1170, yPick + 36, C.amber, 0.8 * s4);
          // which steps see the behavior: brackets over the two stages
          bracket(218, 700, 246, INK, 0.8 * stage(t, 1.5));
          bracket(720, 1170, 246, C.amber, 0.9 * stage(t, 3));

          lab('p-title', 218, 170, 'Finding and testing weight edits', ap, 'left', CSS.white, 'md', '<p>Search prompts supply examples for measuring the effect of a weight change. The validation scores determine which candidate goes to the held-out test. The drawn lengths are illustrative.</p>');
          lab('p-tag', 218, 194, 'Illustration', ap * 0.9, 'left', CSS.slate);
          lab('p-sup0', 459, 232, 'Search without behavior labels', stage(t, 1.5), 'center', CSS.white, 'md');
          lab('p-sup1', 945, 232, 'Select and test using a behavior score', stage(t, 3), 'center', CSS.amber, 'md');
          lab('p-in', 268, 470, 'search prompts', s0, 'below');
          lab('p-read', 488, 298, 'effect measured here', s1, 'center', CSS.white, 'xs');
          lab('p-pert', 488, 410, 'perturbation added here', s1, 'center', CSS.white, 'xs');
          lab('p-model', 488, 470, 'the model', s1, 'below');
          lab('p-rows', 675, 470, 'weight changes', s2, 'below');
          lab('p-val', 818, 470, 'validation prompts', s3, 'below');
          lab('p-pick', 720 + 190 * PIPE_SCORE[PIPE_PICK] + 8, yPick - 1, 'picked', s3 * smooth(B1 + 2.5, B1 + 2.9, t), 'left', CSS.amber);
          lab('p-test', 1091, yPick + 50, 'held-out prompts', s4, 'below');
          lab('p-res', 1082, yPick - 1, 'score', s4, 'left', CSS.amber);
        }

        // ── 5.4: the paper's Table 3, quoted ──
        if (ar > 0.004) {
          pipe.push(P2(RX0, RY0), P2(RX1, RY0), INK, 0.9 * ar);
          pipe.push(P2(RX0, RY0), P2(RX0, rY(RVMAX)), INK, 0.9 * ar);
          for (const v of [0.2, 0.4, 0.6]) {
            pipe.push(P2(RX0 - 5, rY(v)), P2(RX0, rY(v)), INK, 0.9 * ar);
            pipe.push(P2(RX0, rY(v)), P2(RX1, rY(v)), INK, 0.15 * ar);
          }
          const BW = 70, cols: RGB[] = [C.slate, C.amber, C.slate]; // the two comparators in the key ink of controls
          T3.forEach((r, b) => {
            const g = resGrow(t, b), top = rY(r.v * g);
            bars.quad(nq++, rBX(b) - BW / 2, RY0, rBX(b) + BW / 2, top, cols[b], (b === 1 ? 1.2 : b === 0 ? 0.6 : 1.0) * ar);
            lab('r-v' + b, rBX(b), top - 4, r.s, ar * smooth(0.6, 0.95, g), 'center', b === 1 ? CSS.amber : CSS.white, 'md');
            lab('r-n' + b, rBX(b), RY0 + 8, `${r.name}<small>${r.sub}</small>`, ar, 'below', b === 1 ? CSS.amber : CSS.white);
          });
          // the gap the password opens over the locked model, as a bracket beside the chart
          const gk = ar * smooth(B4 + 2.4, B4 + 2.9, t), xg = RX1 + 14;
          for (const [xa, xb] of [[rBX(0) + 38, rBX(1) - 38], [rBX(1) + 38, rBX(2) - 38], [rBX(2) + 38, xg]]) pipe.push(P2(xa, rY(T3[0].v)), P2(xb, rY(T3[0].v)), INK, 0.35 * gk);
          pipe.push(P2(rBX(2) + 38, rY(T3[2].v)), P2(xg, rY(T3[2].v)), INK, 0.35 * gk);
          pipe.push(P2(xg, rY(T3[0].v)), P2(xg, rY(T3[2].v)), INK, 0.9 * gk);
          lab('r-gap', xg + 10, (rY(T3[0].v) + rY(T3[2].v)) / 2, 'Password gap', gk, 'left');
          [0, 0.2, 0.4, 0.6].forEach((v, q) => lab('r-y' + q, RX0 - 9, rY(v), v.toFixed(1), ar * 0.85, 'right'));
          lab('r-ax', RX0 - 60, rY(0.3), 'Pass rate', ar, 'right');
          lab('r-t', RX0, rY(RVMAX) - 52, 'BigCodeBench results', ar, 'left', CSS.white, 'md', '<p>The bars show pass rates on held-out coding problems for the same password-locked Llama-3.3-70B model. The bracket spans the gap between its scores without and with the password.</p><p>Mack and co-authors tested same-size random weight edits separately on 8B models (Table 1).</p>');
          lab('r-tag', RX0, rY(RVMAX) - 28, 'Reported by the authors · Mack et al., Table 3 · Llama-3.3-70B', ar * 0.9, 'left', CSS.slate);
        }
        bars.count = nq;
        bars.commit();
        lines.count = n;
        lines.commit();
        pipe.commit();

        // ── grids ──
        gridLines.uIntensity.value = ag;
        missLines.uIntensity.value = aMiss(t);
        let ns = 0;
        if (ag > 0.004) {
          drawHeat(trP, heatP, iteration, ag);
          drawHeat(trS, heatS, iteration, ag);
          // the count strips follow the grids through the iterations
          for (const [tl, x0, col, k] of [[tallyP, GP.x, ABL, 1.0], [tallyS, GS.x, C.amber, 1.1]] as [Tally, number, RGB, number][]) {
            for (let c = 0; c < K; c++) {
              const h = tl.counts[colOrd[c]] * stripPx;
              if (h > 0.05) strip.quad(ns++, x0 + c * CW + 3, STRIP0 - h, x0 + (c + 1) * CW - 3, STRIP0, col, k * ag);
            }
          }
        }
        if (ag <= 0.004) {
          heatP.uOpacity.value = 0;
          heatS.uOpacity.value = 0;
        }
        strip.count = ns;
        strip.commit();
        pairLines.count = 0;
        const af5 = follow ? aFollow(t) : 0;
        if (af5 > 0.004 && follow)
          for (const [tr, x0] of [[trP, GP.x], [trS, GS.x]] as [Track, number][]) {
            const rA = tr.rows.indexOf(follow.stay), rB = tr.rows.indexOf(follow.leave);
            // two neighboring rows share one outline
            const spans = Math.abs(rA - rB) === 1 ? [[Math.min(rA, rB), Math.max(rA, rB)]] : [[rA, rA], [rB, rB]];
            for (const [r0, r1] of spans) {
              const y0 = GY + r0 * CH - 1, y1 = GY + (r1 + 1) * CH + 1;
              pairLines.polyline([P2(x0 - 5, y0), P2(x0 + GW + 5, y0), P2(x0 + GW + 5, y1), P2(x0 - 5, y1)], FOLLOW, 1.1 * af5, true);
            }
          }
        pairLines.commit();

        // ═════════════ labels ═════════════
        // slide 5.2
        const midMain = (MAIN.x0 + MAIN.x1) / 2;
        lab('bar', midMain, CY1 + 10, 'planted detectors', am * smooth(B2 + 0.6, B2 + 1.2, t), 'below');
        lab('line', MAIN.x1 + 16, vY(0), 'Detector threshold', am * smooth(B2 + 0.9, B2 + 1.5, t));
        lab('b-title', MAIN.x0, CY0 - 72, 'Detector input before and after an edit', am, 'left', CSS.white, 'md', '<p>Each bar shows a detector’s input minus its threshold. Detectors are ordered by the initial gap to threshold, with the easiest at the left. The detectors use softplus and respond weakly even below threshold.</p>');
        lab('b-tag', MAIN.x0, CY0 - 49, 'Computed for this presentation', am * 0.9, 'left', CSS.slate);
        lab('b-axis', MAIN.x0, CY0 - 26, 'Input − threshold', am, 'left', CSS.slate);
        lab('b-key', MAIN.x1 + 16, vY(0) + 48, 'Bars · after edit<br>Ticks · before edit', am * aEdit(t), 'left');
        // slide 5.3
        const yTitle = STRIP0 - STRIP_H - 34;
        const gridHelp = `<p>Each row is one candidate, ordered by its final match. Each column is one detector. Stronger shading means closer positive alignment with its ideal weight change. The bars count candidates above a cosine of ${THR}.</p>` + (follow ? '<p>A and B lean toward the same detector early in the search. Their outlined rows show where each search takes them.</p>' : '');
        lab('gP', GP.x - 2, yTitle - 42, 'Gradient iteration alone', ag, 'left', CSS.white, 'md', gridHelp);
        lab('gS', GS.x - 2, yTitle - 42, 'With soft orthogonalization', ag, 'left', CSS.amber, 'md', gridHelp + (endS.idle > 0 ? `<p>The bracket groups candidates that finish at or below the ${THR} cutoff for every detector.</p>` : ''));
        // how many detectors each search has found so far, then the count over each column and the missed ones
        lab('nP', GP.x - 2, yTitle, `${tallyP.found} of ${K} found`, ag, 'left', CSS.white, 'md');
        lab('nS', GS.x - 2, yTitle, `${tallyS.found} of ${K} found`, ag, 'left', CSS.amber, 'md');
        const aEnd = aMiss(t);
        for (const [k, tr, x0, css] of [['P', trP, GP.x, CSS.white], ['S', trS, GS.x, CSS.amber]] as [string, Track, number, string][]) {
          const end = tr.tl[tr.n - 1];
          for (let c = 0; c < K; c++) {
            const nj = end.counts[colOrd[c]];
            if (nj > 0) lab(`n${k}${c}`, x0 + (c + 0.5) * CW, STRIP0 - nj * stripPx + 6, String(nj), aEnd, 'center', css, 'xs');
          }
        }
        missLabels.forEach(([x, y], q) => lab('m' + q, x, y, 'missed', aEnd, 'center'));
        for (const [k, x0] of [['P', GP.x], ['S', GS.x]] as [string, number][]) {
          lab('e' + k, x0 + GW / 2, GY + GH + 6, 'Easiest → hardest', ag, 'below');
        }
        const lx = GS.x + GW + 30;
        // where the search stands, and the two perturbations the figure follows
        lab('g-tag', lx, GY - 40, 'Computed for this presentation', ag * 0.9, 'left', CSS.slate);
        lab('iter', lx, GY - 6, iteration === 0 ? 'Same starting candidates' : `Iteration ${iteration} of ${ITERS}`, ag * (1 - finished(t)));
        if (follow) {
          const aF = aFollow(t);
          for (const [k, tr, x0] of [['P', trP, GP.x], ['S', trS, GS.x]] as [string, Track, number][])
          {
            const rA = tr.rows.indexOf(follow.stay), rB = tr.rows.indexOf(follow.leave);
            if (Math.abs(rA - rB) <= 2) lab(`f${k}0`, x0 - 9, GY + ((rA + rB) / 2 + 0.5) * CH, rA < rB ? 'A, B' : 'B, A', aF, 'right', FOLLOW_CSS);
            else [rA, rB].forEach((r, n) => lab(`f${k}${n}`, x0 - 9, GY + (r + 0.5) * CH, n === 0 ? 'A' : 'B', aF, 'right', FOLLOW_CSS));
          }
          const cS = colPos[follow.j0], cL = colPos[follow.jL];
          lab('oP', GP.x + GW / 2, GY + GH + 30, `A and B match ${cS === 0 ? 'the easiest detector' : 'one detector'}`, aEnd, 'below', FOLLOW_CSS);
          lab('oS', GS.x + GW / 2, GY + GH + 30, `B matches ${missedSet.has(follow.jL) ? 'a missed detector' : cL >= K / 2 ? 'a harder detector' : 'another detector'}`, aEnd, 'below', FOLLOW_CSS);
        }
        const idleY = GY + GH - endS.idle * CH / 2;
        const keyY = endS.idle > 0 && idleY < GY + 98 ? idleY + 64 : GY + 32;
        lab('lg1', lx, keyY, 'Rows · candidates<br>Columns · detectors<br>Bars · matches per detector', ag);
        // the rows at the bottom of the right grid: perturbations that end on no detector
        if (endS.idle > 0) {
          lab('idle', lx, idleY, `${words(endS.idle)} candidate${endS.idle === 1 ? '' : 's'}<br>match${endS.idle === 1 ? 'es' : ''} no detector`, aEnd);
        }
      },

      readouts(t) {
        const rows: Readout[] = [];
        if (t >= G_RUN[1] && t < B4) {
          rows.push({ label: 'detectors found, gradient iteration alone', value: `${endP.found} of ${K}` });
          rows.push({ label: 'detectors found, with soft orthogonalization', value: `${endS.found} of ${K}` });
          rows.push({ label: 'found and raised above threshold, gradient iteration alone', value: `${aboveP} of ${K}`, detail: true });
          rows.push({ label: 'found and raised above threshold, with soft orthogonalization', value: `${aboveS} of ${K}`, detail: true });
        }
        if (t >= B4 + 1.2) {
          rows.push({ label: 'paper: locked, no password', value: '0.43', note: 'BigCodeBench pass rate on held-out problems, Table 3' });
          rows.push({ label: 'paper: picked perturbation, no password', value: '0.52' });
          rows.push({ label: 'paper: unlocked, with the password', value: '0.54' });
        }
        return rows;
      },

      // Notes explain choices and limits that the caption and figure do not show.
      details(t) {
        if (t < B3 || t >= B4) return undefined;
        return {
          title: 'A simplified search',
          body:
            `<p>Both searches start from the same candidates. The toy seeks large downstream changes on one input. It uses gradient iteration with or without CPE’s soft orthogonalization step. In the paper, the search also learns downstream directions and penalizes similarity in its objective.</p>` +
            `<p>A candidate finds a detector when its weight change has a cosine above ${THR} with that detector’s ideal weight change. Raising the detector above threshold also depends on the candidate’s radius and the detector’s starting gap.</p>` +
            (AUDITED
              ? '<p>Across 60 random draws at these settings, soft orthogonalization found more detectors than gradient iteration alone in 58 runs and all twelve in 59. These checks measure variation in the toy, not performance on language models.</p>'
              : ''),
        };
      },
      inspect(t) {
        const hs: Hotspot[] = [];
        if (aPipe(t) > 0.5)
          hs.push({
            id: 'procedure',
            pos: Wp(206, 246),
            title: 'Reusing the discovered weight changes',
            body:
              '<p>CPE adds rank-1 weight changes at several attention output projections. At each edited layer, a rank-1 component reads one input direction and writes along one output direction.</p>' +
              '<p>Because discovery uses no behavior score, the same candidates can be evaluated for several behaviors. To look for suppression, select the candidate with the lowest validation score for that behavior.</p>',
          });
        if (aMain(t) > 0.5) {
          const otherBars = sideFull === 'lower'
            ? '<p>The same weight change points against the other detectors’ ideal changes, which lowers their inputs.</p>'
            : sideFull
              ? '<p>The same weight change also has components along other detectors’ ideal changes, so their inputs move too.</p>'
              : '';
          hs.push({
            id: 'bars',
            pos: Wp(MAIN.x0 - 24, CY0 - 72),
            title: 'A test with known target directions',
            body:
              '<p>The detectors were constructed to be reachable by rank-1 weight changes. Recovering their known directions checks the search in this toy; it does not establish how language models store behavior.</p>' + otherBars,
          });
        }
        if (aGrid(t) > 0.5) {
          hs.push({
            id: 'gradonly',
            pos: Wp(GP.x - 8, STRIP0 - STRIP_H - 76),
            title: 'Why candidates gather at easy detectors',
            body:
              '<p>At the unmodified input, detectors closer to threshold have steeper responses. This can favor them during search. Their influence on each update also depends on the candidate’s downstream effect.</p>' +
              '<p>Mack, Panickssery and Turner report a similar collapse with small gradient-ascent steps. This control uses gradient iteration, which replaces candidates with normalized gradients at each step.</p>',
          });
          hs.push({
            id: 'selection',
            pos: Wp(GS.x + GW + 16, GY - 6),
            title: 'Selecting for a planted target',
            body:
              `<p>The target is the detector farthest below threshold in the unmodified toy. For each search, the detector’s response on ${words(NVAL)} noisy inputs selects one candidate. Both winners are scored on the same ${NTEST} fresh noisy inputs.</p>` +
              `<p>The mean detector response is ${fmt(muGradient)} for gradient iteration alone and ${fmt(muSoft)} with soft orthogonalization. Leaving the weights alone gives ${fmt(muNone)}. An exactly aligned change of the same size gives ${fmt(muOracle)}.</p>` +
              '<p>All candidates learn the same input direction up to sign. On these noisy inputs, the candidates rank by their alignment with the target detector.</p>' +
              (AUDITED ? '<p>Soft orthogonalization improved the held-out response for this target in 41 of 60 runs. The number of detectors found and the quality of the best candidate for one detector measure different properties of the search.</p>' : ''),
          });
        }
        if (aRes(t) > 0.5) {
          hs.push({
            id: 'result',
            pos: Wp(RX0 - 14, rY(RVMAX) - 60),
            title: 'BigCodeBench test and source values',
            body:
              '<p>The paper reports 10 search problems, 100 validation problems to choose the adapter, and 744 held-out problems. Unit tests score a single greedily generated solution per problem.</p>' +
              '<p>Table 3 gives 0.43 for the locked model and reports that CPE’s 0.52 is significantly higher. The surrounding text gives 0.46 for the locked model. Under either figure, the edit closes most of the locked-to-unlocked gap.</p>' +
              '<p>The comparison establishes an effect of applying this edit. Which features the untouched model uses to implement the lock or solve the problems remains open.</p>' +
              '<p>In a separate post-hoc test, Taylor and colleagues also elicited Tarun’s coding performance with a different system prompt (arXiv:2512.07810, Appendix A.3.2).</p>',
          });
          hs.push({
            id: 'controls',
            pos: Wp(RX1 + 2, rY(0.12) - 8),
            title: 'Comparisons on the 8B models',
            body:
              '<p>Table 1 compares five tests on Llama-3.1-8B and Qwen3-8B. Random adapters matched in size and layer locations score within 0.02 of the unmodified model in every test.</p>' +
              '<p>The authors ranked sparse autoencoder features by their predicted first-order effect on a later layer. Steering with those features scored below CPE in all five tests. GRPO, reinforcement learning trained on the behavior score, had higher point estimates in all four comparisons. Both methods are in the paper’s statistically best group on the two Qwen tests.</p>',
          });
        }
        return hs;
      },

      probe(t, p) {
        if (!p.plane) return null;
        const [x, y] = unWp(p.plane);
        const charts: [{ x0: number; x1: number }, ArrayLike<number> | null, number, string][] = [];
        if (aMain(t) > 0.5) {
          const w = push(t);
          charts.push([MAIN, w > 0 ? preFull : null, w, w < 1 ? 'partial weight change' : 'candidate weight change']);
        }
        for (const [g, pre, w, name] of charts) {
          if (x < g.x0 || x >= g.x1 || y < CY0 - 20 || y > CY1 + 10) continue;
          const c = Math.min(K - 1, Math.floor((x - g.x0) / slot(g))), j = colOrd[c];
          const v = pre ? lerp(toy.pre0[j], pre[j], w) : toy.pre0[j];
          return {
            pos: Wp(barX(g, c), vY(v * grow(t))),
            title: `Detector ${j + 1} input relative to threshold`,
            rows: [
              { label: 'gap below threshold, unmodified model', value: fmt(depth[j]) },
              { label: pre ? `with the ${name}` : 'unmodified model', value: fmt(v) },
              { label: 'above response threshold', value: v > 0 ? 'yes' : 'no' },
            ],
          };
        }
        if (aRes(t) > 0.5)
          for (let b = 0; b < T3.length; b++) {
            if (Math.abs(x - rBX(b)) > 35 || y > RY0 + 4 || y < rY(T3[b].v) - 20) continue;
            return {
              pos: Wp(rBX(b), rY(T3[b].v)),
              title: `${cap(T3[b].name)}, ${T3[b].sub}`,
              rows: [{ label: 'paper: BigCodeBench pass rate', value: T3[b].s }],
              note: 'Quoted from Table 3 of the paper: greedy pass rate on held-out problems.',
            };
          }
        if (aGrid(t) > 0.5) {
          for (const [tr, x0, name] of [[trP, GP.x, 'gradient iteration alone'], [trS, GS.x, 'with soft orthogonalization']] as [Track, number, string][]) {
            if (x < x0 || x >= x0 + GW || y < GY || y >= GY + GH) continue;
            const c = Math.min(K - 1, Math.floor((x - x0) / CW)), r = Math.min(M - 1, Math.floor((y - GY) / CH));
            const i = tr.rows[r], j = colOrd[c], k = iterationAt(t), cs = tr.run.snaps[k * tr.run.sub].cos[i * K + j];
            return {
              pos: Wp(x0 + (c + 0.5) * CW, GY + (r + 0.5) * CH),
              title: `Perturbation ${i + 1} and detector ${j + 1}, ${name}`,
              rows: [
                { label: 'iteration', value: `${k} of ${ITERS}` },
                { label: 'cosine', value: num(cs, 3) },
                { label: 'finds it', value: cs > THR ? 'yes' : 'no' },
              ],
            };
          }
        }
        return null;
      },
    };
    return chapter;
  },
};

export default elicitation;
