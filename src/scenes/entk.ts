import * as THREE from 'three/webgpu';
import { uniform, vec3 } from 'three/tsl';
import type { Chapter, ChapterFactory, Control, Hotspot, ParamValue, Probe, Readout, Result } from '../engine/types';
import { resolveParams } from '../engine/params';
import { C, CSS, THEME, mixRGB, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { MatrixField } from '../kit/shapes';
import { mulberry32, gaussian, smooth, easeInOut, clamp, lerp } from '../lib/math';
import { makeTrainer, entkKernel, eigFull, fourierMode, type Family, type Layer } from './entk.math';

/**
 * EMPIRICAL NTK — J. Lin, "Feature Identification via the Empirical NTK" (arXiv:2510.00468).
 *
 * The miniature: a quadratic MLP f(x) = W2 (W1 x)² is trained on (a + b) mod p with full-batch AdamW.
 * At a set of checkpoints the class-averaged empirical NTK over all p² inputs is formed in closed form
 * and fully diagonalized (Householder + QL). Everything drawn is read off those decompositions: the
 * kernel matrix, its eigenvalues, how much of each family of waves lies in the two groups of
 * eigenvectors (a continuous subspace overlap, tr(P_group P_family), at most 2⌊p/2⌋). A group is a block
 * of 4⌊p/2⌋ consecutive eigenvectors after the leading one. The Fourier family sizes set the boundaries
 * after 32 and 64 nonleading modes at p = 17.
 * The groups are not detected from the spectrum: the top-two-drop rule picks other boundaries before late
 * training (cross-review, 17 Nov 2026), so a detected size would change along the replay.
 *
 * The chapter's main result is the comparison with the kernel at initialization: group 1 already holds
 * the waves in a and in b, group 2 rises from about the random level to almost all the waves in a + b.
 * Overlaps are drawn as shares between 0 and 1 (the subspace overlap divided by the family's size). Near the
 * half crossing of group 2 the checkpoints are refined by bisection, so that the drawn curve follows the turn.
 * The last slide keeps the overlap chart beside the projected pattern and discusses a follow-up ablation.
 *
 * Deviations from the paper: p = 17 and n = 256 instead of p = 29 and
 * n = 512; W1 starts near zero and W2 at σ = 0.5, so that the first-layer kernel (which carries the
 * x·x′ structure) dominates at initialization; 400 epochs of AdamW with weight decay 0.1. The film shows one
 * run, a typical one; the run-dependent sentences were audited outside it (see RUNS).
 *
 * Staging: five flat panels in the plane z = 0, each in its own group. The camera stays put; a panel
 * fades in on the slide that talks about it and out when the text moves on, and the engine re-frames
 * the panels on screen. Panels: the sums, one pattern on the grid, the kernel, its eigenvalues, training.
 */

type V3 = [number, number, number];

const EPOCHS = 400;
const LR = 0.05, SIGMA1 = 0.01, SIGMA2 = 0.5;
const CP_FRAC = [0, 0.1, 0.2, 0.275, 0.35, 0.425, 0.5, 0.6, 0.75, 1];
const SNAP_EVERY = 5; // weights are kept every 5 epochs, so that the turn of group 2 can be refined
const REFINE = 3; // extra eigensolves spent on bisecting the turn
const FAMS: Family[] = ['a', 'b', 'sum', 'diff'];
const FAM_TEX: Record<Family, string> = { a: 'a', b: 'b', sum: '(a+b)', diff: '(a-b)' };
const FAM_WORD: Record<Family, string> = { a: '$a$', b: '$b$', sum: '$a + b$', diff: '$a - b$' };

// ── figure geometry (world units; every panel lies in the plane z = 0) ──
const S = 6, HS = 3; // side and half side of a square panel
const CW = 6.4; // eigenvalue panel width
const EW = 9, EH = 5; // training panel
const AX = -2.45; // center of the sums
const DX_R = 5.75, DX_L = -6.0; // center of the pattern beside the sums, and beside training
const BX = -3.7; // center of the kernel
const CX = 0.6; // left edge of the eigenvalues
const EX = -0.6; // left edge of training
const CBAR = 0.15, CBAR_GAP = 0.25; // the pattern's color bar
const PSI_GAIN = 0.74, PSI_CLIP = 1.25; // heatmap value = clip(ψ√(N/2)) · gain
const K_INK = THEME === 'paper' ? 1.3 : 0.6;
const CAM_D = 20; // camera distance; fov 38°
const PPU = 450 / (CAM_D * Math.tan((19 * Math.PI) / 180)); // frame pixels per world unit at z = 0

// ── timeline (seconds): one beat per slide ──
const T2 = 4.8, T3 = 9.0, T4 = 14.0, T5 = 18.4, T6 = 22.0, T7 = 27.0, DUR = 31;
const REPLAY0 = T6 + 0.8, REPLAY1 = T6 + 3.9; // slide 4.6: training replays 0 → 400 epochs

// The run-dependent sentences against 32 runs from different random starts (seeds 1 to 32, the Explore
// range), checked outside the film with the chapter's own code at the default settings. The two
// largest nonconstant spectral drops at the last checkpoint matched the proposed cuts in 30 of 32.
const RUNS = { n: 32, seeds: [1, 32], spectralMatches: 30 };

const f2 = (x: number) => x.toFixed(2);
const f3 = (x: number) => x.toFixed(3);
const pct = (x: number) => (100 * x).toFixed(0) + '%';
/** A positive real for use inside maths: 4 significant digits, scientific outside [1e-3, 1e4). */
const sci = (x: number) => {
  if (x === 0 || !isFinite(x)) return '0';
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e4) return x.toPrecision(4);
  const e = Math.floor(Math.log10(a));
  return `${(x / Math.pow(10, e)).toFixed(3)}\\times 10^{${e}}`;
};
// Digits above this scale are not meaningful when a denominator is at the numerical floor.
const times = (x: number) => (x > 1e4 ? '$> \\times 10^{4}$' : `$\\times ${f2(x)}$`);

// Explore-mode knobs. Hidden while interactivity is frozen; kept as they were.
const controls: Control[] = [
  {
    key: 'p', label: '$p$  modulus, $N = p^2$ inputs', kind: 'select', value: 17,
    options: [11, 13, 17, 19].map((v) => ({ value: v, label: `${v}  (N = ${v * v})` })),
    hint: 'Each group has $4\\lfloor p/2\\rfloor$ eigenvectors. Changing $p$ moves the rank boundaries.',
  },
  { key: 'n', label: '$n$  hidden width', kind: 'range', value: 256, min: 64, max: 384, step: 32, hint: 'Changes the number of hidden neurons in the quadratic network.' },
  { key: 'frac', label: '$\\alpha$  train fraction', kind: 'range', value: 0.8, min: 0.5, max: 0.95, step: 0.05, hint: 'Changes how many input pairs the network sees during training.' },
  { key: 'wd', label: 'wd  AdamW weight decay', kind: 'range', value: 0.1, min: 0, max: 0.3, step: 0.01, hint: 'With the default seed, the run at wd = 0 does not generalize in 400 epochs. At wd = 0.3 the first drop closes and the cliffs mix.' },
  { key: 'seed', label: 'seed  init and data split', kind: 'range', value: 22, min: 1, max: 32, step: 1, hint: 'Shows the run-to-run spread of the drops, the overlaps and the epoch of generalization.' },
  {
    key: 'layer', label: '$\\theta$  weights entering the kernel', kind: 'select', value: 'both',
    options: [
      { value: 'both', label: 'K1 + K2  (W1 and W2)' },
      { value: 'K1', label: 'K1  (W1 only)' },
      { value: 'K2', label: 'K2  (W2 only)' },
    ],
    hint: 'Selects the layer whose weights enter the eNTK. In the default run $K_1$ carries the $a$ and $b$ modes; the $a\\pm b$ modes lead $K_2$.',
  },
  {
    key: 'family', label: '$F$  Fourier family drawn in panel (d)', kind: 'select', value: 'film', live: true,
    options: [{ value: 'film', label: 'as in this presentation' }, ...FAMS.map((f) => ({ value: f as string, label: `cos 2πk${FAM_TEX[f].replace('-', '−')}/p` }))],
    hint: 'Pins panel (d) to one mode, projected onto the cliff that overlaps its family most.',
  },
  { key: 'k', label: '$k$  frequency of the drawn mode', kind: 'range', value: 2, min: 1, max: 9, step: 1, live: true, hint: 'The chapter clamps it to $\\lfloor p/2\\rfloor$. $\\lVert P f\\rVert^2$ in Measured is the squared overlap of this one mode.' },
];

const entk: ChapterFactory = {
  id: 'entk',
  label: 'entk · train, kernel, eigensolve',
  controls,
  async build(ctx, params) {
    const P = resolveParams(controls, params);
    const p = P.p as number, n = P.n as number, N = p * p;
    const half = (p - 1) >> 1, SH = 4 * half; // waves per family pair = group size
    const FDIM = 2 * half; // waves in one family (cos and sin of ⌊p/2⌋ frequencies)
    const layer = P.layer as Layer;
    const kDefault = 2;
    const Y0 = p - 1; // the answer class drawn in green: a + b = p − 1, one unbroken diagonal

    // ───────────────────────── 1. train ─────────────────────────
    const rnd = mulberry32(P.seed as number), gs = gaussian(rnd);
    const baseEpochs = CP_FRAC.map((f) => Math.round(f * EPOCHS));
    const snapEpochs = Array.from(new Set([...baseEpochs, ...Array.from({ length: Math.floor(EPOCHS / SNAP_EVERY) + 1 }, (_, i) => i * SNAP_EVERY)])).sort((x, y) => x - y);
    const trainer = makeTrainer(
      { p, n, epochs: EPOCHS, lr: LR, wd: P.wd as number, trainFrac: P.frac as number, sigma1: SIGMA1, sigma2: SIGMA2, checkpoints: snapEpochs },
      rnd, gs,
    );
    const chunk = Math.max(10, Math.round((60 * 289 * 17 * 256) / (N * p * n)));
    let doneEp = 0;
    while (!trainer.advance(chunk)) {
      doneEp += chunk;
      await ctx.log(`AdamW epoch ${Math.min(doneEp, EPOCHS)} / ${EPOCHS}`);
    }
    const run = trainer.result;
    const nTest = N - run.nTrain;
    const sane = (x: number) => (isFinite(x) ? x : 0);
    const snapAt = new Map(run.snaps.map((s) => [s.epoch, s]));

    // ───────────────────────── 2. kernel + eigendecomposition per checkpoint ─────────────────────────
    const NB = Math.min(N, 2 * SH + 1 + 31); // eigenvalues drawn
    const modes: Float32Array[][] = FAMS.map((f) => Array.from({ length: half }, (_, i) => Float32Array.from(fourierMode(p, f, i + 1, 0))));
    const modesSin: Float32Array[][] = FAMS.map((f) => Array.from({ length: half }, (_, i) => Float32Array.from(fourierMode(p, f, i + 1, 1))));
    /** One checkpoint: eigenvalues, the 2·SH group eigenvectors, and group × family overlaps. */
    type Cp = { epoch: number; lam: Float64Array; logv: Float32Array; Q: Float32Array; ov: number[][] };
    let Kfinal: Float64Array = new Float64Array(0);
    const kTone = new Float32Array(N * N); // the trained kernel as drawn: √(|entry| / max)
    let kmax = 1, kminOff = 0;
    const solve = (epoch: number): Cp => {
      const snap = snapAt.get(epoch)!;
      const K = entkKernel(p, n, snap.W1, snap.W2, layer);
      for (let i = 0; i < K.length; i++) if (!isFinite(K[i])) K[i] = 0;
      const eig = eigFull(K, N);
      if (epoch === EPOCHS) {
        Kfinal = K;
        kmax = 1e-30; kminOff = Infinity;
        for (const entry of K) { kmax = Math.max(kmax, entry); kminOff = Math.min(kminOff, entry); }
        // Match the color scale even in layer-restricted kernels with large negative entries.
        const k0 = kminOff >= 0 ? kminOff : 0;
        const kr = Math.max(1e-30, kmax - k0, -kminOff);
        for (let i = 0; i < K.length; i++) kTone[i] = Math.sqrt((Math.abs(K[i]) - k0) / kr);
      }
      const lam = eig.values;
      const l1 = Math.max(lam[1], 1e-300);
      const logv = new Float32Array(NB);
      for (let r = 0; r < NB; r++) logv[r] = Math.log10(Math.max(lam[r], l1 * 1e-9) / l1);
      const Q = Float32Array.from(eig.vectors.subarray(N, (2 * SH + 1) * N)); // ranks 1 … 2·SH
      const ov: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0]];
      for (let s = 0; s < 2; s++)
        for (let f = 0; f < 4; f++) {
          let acc = 0;
          for (let k = 0; k < half; k++)
            for (const m of [modes[f][k], modesSin[f][k]])
              for (let r = s * SH; r < (s + 1) * SH; r++) {
                let d = 0;
                const ro = r * N;
                for (let i = 0; i < N; i++) d += Q[ro + i] * m[i];
                acc += d * d;
              }
          ov[s][f] = sane(acc);
        }
      return { epoch, lam, logv, Q, ov };
    };
    const cps: Cp[] = [];
    for (let c = 0; c < baseEpochs.length; c++) {
      cps.push(solve(baseEpochs[c]));
      await ctx.log(`eNTK ${N}×${N} at epoch ${baseEpochs[c]}: eigensolve ${c + 1} / ${baseEpochs.length}`);
    }
    /** Match each family to the group holding more of it after training, including nondefault URL settings. */
    const last = cps[cps.length - 1];
    const famGroup = FAMS.map((_, f) => (last.ov[0][f] >= last.ov[1][f] ? 0 : 1));
    const sA = famGroup[0], sB = famGroup[1], sS = famGroup[2], sD = famGroup[3];
    const aOrdinal = sA === 0 ? 'first' : 'second', sumOrdinal = sS === 0 ? 'first' : 'second';
    const inputGroups = sA === sB ? `the ${aOrdinal} group` : `groups ${sA + 1} and ${sB + 1}`;
    const inputHolds = sA === sB ? 'holds' : 'hold';
    const defaultGrouping = sA === 0 && sB === 0 && sS === 1 && sD === 1;
    // Bisect the interval where half the a + b family enters its matched group (cliff 2 by default).
    const half2 = (cp: Cp) => cp.ov[sS][2] / FDIM;
    for (let it = 0; it < REFINE; it++) {
      const c = cps.findIndex((cp, i) => i > 0 && half2(cp) >= 0.5 && half2(cps[i - 1]) < 0.5);
      if (c < 0) break;
      const e0 = cps[c - 1].epoch, e1 = cps[c].epoch;
      const mid = Math.round((e0 + e1) / 2 / SNAP_EVERY) * SNAP_EVERY;
      if (mid <= e0 || mid >= e1 || !snapAt.has(mid)) break;
      cps.splice(c, 0, solve(mid));
      await ctx.log(`eNTK at epoch ${mid}: refining the turn of cliff ${sS + 1}`);
    }
    run.snaps.length = 0; // the weights are no longer needed
    snapAt.clear();
    const NC = cps.length;
    const fin = cps[NC - 1], ini = cps[0];

    // Match the floor used by the spectrum bars; two noise-level eigenvalues have displayed ratio ×1.
    const lamF = (cp: Cp, r: number) => Math.max(cp.lam[r], Math.max(cp.lam[1], 1e-300) * 1e-9);
    const ratio = (cp: Cp, r: number) => sane(lamF(cp, r) / lamF(cp, r + 1));
    const drop1 = ratio(fin, SH), drop2 = ratio(fin, 2 * SH);
    const largestDrops = Array.from({ length: N - 2 }, (_, i) => i + 1)
      .sort((r, s) => ratio(fin, s) - ratio(fin, r)).slice(0, 2).sort((r, s) => r - s);
    let stepIn = 1;
    for (let r = 1; r < 2 * SH; r++) if (r !== SH) stepIn = Math.max(stepIn, ratio(fin, r));
    /** Find the first crossing that stays within five percentage points of the threshold for the next seven epochs. */
    const firstAbove = (accuracy: Float32Array, threshold: number) => {
      const floor = threshold - 0.05;
      for (let epoch = 0; epoch < accuracy.length; epoch++) {
        if (!(accuracy[epoch] >= threshold)) continue;
        let sustained = true;
        const lookaheadEnd = Math.min(accuracy.length, epoch + 8);
        for (let later = epoch; later < lookaheadEnd; later++) {
          if (accuracy[later] < floor) {
            sustained = false;
            break;
          }
        }
        if (sustained) return epoch;
      }
      return -1;
    };
    const epTest10 = firstAbove(run.testAcc, 0.1), epTest90 = firstAbove(run.testAcc, 0.9);
    const rise = epTest10 >= 0 && epTest90 >= 0 ? epTest90 - epTest10 : -1; // epochs from 10% to 90% of the held-out sums correct
    const chance = (SH * FDIM) / N; // expected overlap of a random group of SH directions with one family
    const ovS1 = cps.map((c) => c.ov[sA][0] / FDIM), ovS2 = cps.map((c) => c.ov[sS][2] / FDIM);
    /** For the build log, interpolate the first epoch at which the matched group's sum-feature overlap reaches 0.5. */
    const epOv50 = (() => {
      for (let c = 1; c < NC; c++)
        if (ovS2[c] >= 0.5 && ovS2[c - 1] < 0.5) return Math.round(lerp(cps[c - 1].epoch, cps[c].epoch, (0.5 - ovS2[c - 1]) / (ovS2[c] - ovS2[c - 1])));
      return ovS2[0] >= 0.5 ? 0 : -1;
    })();
    await ctx.log(`drops ×${f2(drop1)}, ×${f2(drop2)} · cliff ${sS + 1} at half overlap at epoch ${epOv50} · staging`);

    // the outcome, in words: every sentence below that depends on the run picks its branch here
    const g1Init = ini.ov[sA][0] >= 0.9 * FDIM && ini.ov[sB][1] >= 0.9 * FDIM;
    const g2Start = ini.ov[sS][2], g2End = fin.ov[sS][2];
    const g2AtChance = Math.abs(g2Start - chance) <= 0.5 * chance;
    const startComparison = g2AtChance ? 'near the random baseline' : g2Start > chance ? 'above the random baseline' : 'below the random baseline';
    const g2Almost = g2End >= 0.9 * FDIM;
    const g2Change = g2End - g2Start;
    const g2Rise = g2Change >= 0.5 * FDIM;
    const changeDescription = g2Change > 0.02 * FDIM ? 'increased' : g2Change < -0.02 * FDIM ? 'decreased' : 'barely changed';
    const drops = drop1 > 1.15 && drop2 > 1.15;
    const gradual = rise >= 50; // no sudden jump: held-out accuracy takes at least 50 epochs from 10% to 90%

    // Spectrum axis: log10(λ_r / λ_1). Limit the display to the neighborhood of the first two groups.
    const HI = 0.12;
    const LO = Math.max(-2.6, Math.min(-0.6, fin.logv[NB - 1]) - 0.13);
    const leadingOffscale = fin.logv[0] > HI;
    const spectrumClipped = fin.logv[NB - 1] < LO;

    // ───────────────────────── 3. the figure ─────────────────────────
    const group = new THREE.Group();
    const gA = new THREE.Group(), gB = new THREE.Group(), gC = new THREE.Group(), gD = new THREE.Group(), gE = new THREE.Group();
    gA.position.x = AX; gB.position.x = BX; gC.position.x = CX; gD.position.x = DX_R; gE.position.x = EX;
    group.add(gA, gB, gC, gD, gE);
    const frameOf = (lf: LineField, x0: number, y0: number, x1: number, y1: number, k = 0.6) =>
      lf.polyline([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], C.white, k, true);
    const tickMain = [0, half, p - 1];
    const GRAY = mixRGB(C.white, C.slate, 0.45);
    const cA = S / p;
    /** Center of the cell (a, b) on a square grid panel (the sums and the pattern), panel-local. */
    const gpos = (a: number, b: number): V3 => [-HS + (a + 0.5) * cA, -HS + (b + 0.5) * cA, 0];
    const gridTicks = (lf: LineField) => {
      for (let i = 0; i < p; i++) {
        const main = tickMain.includes(i), L = main ? 0.14 : 0.06, q = gpos(i, i);
        lf.push([q[0], -HS - 0.03, 0], [q[0], -HS - 0.03 - L, 0], C.white, main ? 0.85 : 0.5);
        lf.push([-HS - 0.03, q[1], 0], [-HS - 0.03 - L, q[1], 0], C.white, main ? 0.85 : 0.5);
      }
    };

    // ── the sums: one marker per (a, b); a to the right, b upwards ──
    const axA = new LineField(8 + 2 * p);
    frameOf(axA, -HS - 0.03, -HS - 0.03, HS + 0.03, HS + 0.03);
    gridTicks(axA);
    axA.commit();
    const dots = new PointField({ count: N, size: 0.25, bokeh: 0, twinkle: 0 });
    for (let x = 0; x < N; x++) { const g = gpos((x / p) | 0, x % p); dots.setPos(x, g[0], g[1], g[2]); }
    const RSEG = 14;
    const testIdx: number[] = [];
    for (let x = 0; x < N; x++) if (!run.isTrain[x]) testIdx.push(x);
    const rings = new LineField(Math.max(1, nTest * RSEG), { width: 1.5 });
    for (const x of testIdx) rings.circle(gpos((x / p) | 0, x % p), 0.3 * cA, [1, 0, 0], [0, 1, 0], C.white, 1, RSEG);
    const isY0 = (x: number) => (((x / p) | 0) + (x % p)) % p === Y0;
    const order0 = new Float32Array(N); // reveal order: along the anti-diagonals
    for (let x = 0; x < N; x++) order0[x] = (((x / p) | 0) + (x % p)) / (2 * p - 2);
    gA.add(axA.object, rings.object, dots.object);

    // ── one pattern on the grid: a heatmap of p × p cells, b upwards, with its color bar ──
    const CBROWS = 40;
    const psi = new MatrixField({ rows: p, cols: p, cell: S / p, gap: 0.08, height: 0, neg: C.white, pos: C.amber });
    psi.object.rotation.x = Math.PI / 2;
    const cbP = new MatrixField({ rows: CBROWS, cols: 1, cell: S / CBROWS, gap: 0, height: 0, neg: C.white, pos: C.amber });
    for (let i = 0; i < CBROWS; i++) cbP.values[i] = PSI_CLIP * (1 - (2 * i) / (CBROWS - 1)) * PSI_GAIN;
    cbP.commit();
    cbP.object.rotation.x = Math.PI / 2;
    cbP.object.scale.set(CBAR / (S / CBROWS), 1, 1);
    const cbPx = HS + CBAR_GAP + CBAR / 2;
    cbP.object.position.set(cbPx, 0, 0);
    const axD = new LineField(16 + 2 * p);
    frameOf(axD, -HS - 0.03, -HS - 0.03, HS + 0.03, HS + 0.03);
    frameOf(axD, cbPx - CBAR / 2 - 0.02, -HS, cbPx + CBAR / 2 + 0.02, HS, 0.5);
    const cbPy = (u: number) => (u / PSI_CLIP) * HS;
    for (const u of [-1, 0, 1]) axD.push([cbPx + CBAR / 2 + 0.02, cbPy(u), 0], [cbPx + CBAR / 2 + 0.12, cbPy(u), 0], C.white, 0.85);
    gridTicks(axD);
    axD.commit();
    const diagD = new LineField(1, { width: 2.5 }); // the green diagonal a + b = p − 1
    diagD.push(gpos(0, Y0), gpos(Y0, 0), C.green, 1);
    diagD.commit();
    gD.add(psi.object, cbP.object, axD.object, diagD.object);

    // ── the kernel: row 0 at the top ──
    const cB = S / N;
    const bx = (j: number) => -HS + (j + 0.5) * cB, by = (i: number) => HS - (i + 0.5) * cB;
    const mat = new MatrixField({ rows: N, cols: N, cell: cB, gap: 0, height: 0, neg: C.amber, pos: C.amber });
    mat.object.rotation.x = Math.PI / 2;
    const axB = new LineField(8 + 2 * (p + 1) + 6);
    frameOf(axB, -HS - 0.03, -HS - 0.03, HS + 0.03, HS + 0.03);
    for (let a = 0; a <= p; a++) { // block boundaries: rows and columns with the same a
      const u = ((a * p) / N) * S, main = tickMain.includes(a), L = main ? 0.14 : 0.06;
      axB.push([-HS - 0.03, HS - u, 0], [-HS - 0.03 - L, HS - u, 0], C.white, main ? 0.85 : 0.5);
      axB.push([-HS + u, -HS - 0.03, 0], [-HS + u, -HS - 0.03 - L, 0], C.white, main ? 0.85 : 0.5);
    }
    // the kernel's color bar: ink from the largest entry (top) to 0 (bottom), on the same scale as the matrix
    const cbK = new MatrixField({ rows: CBROWS, cols: 1, cell: S / CBROWS, gap: 0, height: 0, neg: C.amber, pos: C.amber });
    for (let i = 0; i < CBROWS; i++) cbK.values[i] = 1 - i / (CBROWS - 1);
    cbK.commit();
    cbK.object.rotation.x = Math.PI / 2;
    cbK.object.scale.set(CBAR / (S / CBROWS), 1, 1);
    const cbKx = HS + CBAR_GAP + CBAR / 2;
    cbK.object.position.set(cbKx, 0, 0);
    frameOf(axB, cbKx - CBAR / 2 - 0.02, -HS, cbKx + CBAR / 2 + 0.02, HS, 0.5);
    for (const y of [-HS, HS]) axB.push([cbKx + CBAR / 2 + 0.02, y, 0], [cbKx + CBAR / 2 + 0.12, y, 0], C.white, 0.85);
    axB.commit();
    const scanB = new LineField(1, { width: 1.5 });
    const leadB = new LineField(12, { width: 1.5 });
    // callout targets: a dark block (same a) and a thin line beside the diagonal (same b), both near the lower right
    const blockA = p - 2, blockC = ((blockA + 0.5) * p) / N * S;
    const tgtBlock: V3 = [-HS + blockC, HS - blockC, 0.02];
    const iLine = (p - 8) * p + half, jLine = iLine + 3 * p; // (a, b) against (a + 3, b), well clear of the dark blocks
    const tgtLine: V3 = [bx(jLine), by(iLine), 0.02];
    const markB = new PointField({ count: 2, size: 0.2, bokeh: 0, twinkle: 0 });
    markB.setPos(0, ...tgtBlock); markB.setPos(1, ...tgtLine);
    markB.setColor(0, C.white, 1); markB.setColor(1, C.white, 0);
    markB.commit();
    gB.add(mat.object, cbK.object, axB.object, scanB.object, leadB.object, markB.object);

    // ── its eigenvalues: one stem and marker per eigenvalue, largest first ──
    const spec = new PointField({ count: NB, size: 0.15, bokeh: 0, twinkle: 0 });
    const stems = new LineField(NB, { width: 2 });
    const dynC = new LineField(64);
    const yTicks = [1, 0.1, 0.01].filter((v) => Math.log10(v) > LO + 0.03);
    const xTicks = [1, SH, 2 * SH, NB - 1];
    const barW = (CW - 0.3) / NB;
    const barX = (r: number) => 0.15 + (r + 0.5) * barW;
    const barY = (lv: number) => -HS + S * clamp((lv - LO) / (HI - LO));
    const groupOf = (r: number) => (r === 0 ? 0 : r <= SH ? 1 : r <= 2 * SH ? 2 : 3);
    const groupCol: RGB[] = [C.white, C.gold, C.cyan, GRAY];
    const quad = (col: RGB) => {
      const uK = uniform(0);
      const m = new THREE.MeshBasicNodeMaterial();
      m.colorNode = (vec3(col[0], col[1], col[2]) as any).mul(uK);
      m.transparent = true;
      m.depthWrite = false;
      m.blending = THREE.AdditiveBlending;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), m);
      mesh.frustumCulled = false;
      return { mesh, uK };
    };
    const shade1 = quad(C.gold), shade2 = quad(C.cyan);
    for (const [sh, r0] of [[shade1, 1], [shade2, SH + 1]] as const) {
      sh.mesh.position.set((barX(r0) + barX(r0 + SH - 1)) / 2, 0, -0.01);
      sh.mesh.scale.set(SH * barW, S, 1);
    }
    gC.add(shade1.mesh, shade2.mesh, stems.object, dynC.object, spec.object);

    // ── training: synchronized accuracy and alignment panels during the replay ──
    const ex = (e: number) => (e / EPOCHS) * EW;
    const splitAt = (t: number) => smooth(T6, T6 + 0.6, t) * (1 - smooth(T7, T7 + 0.6, t));
    const alignmentTop = (t: number) => lerp(EH / 2, -0.45, splitAt(t));
    const panelY = (v: number, bottom: number, top: number) => lerp(bottom, top, 0.05 + 0.88 * v);
    const ey = (v: number, t = 0) => panelY(v, -EH / 2, alignmentTop(t));
    const accuracyBottom = 0.65;
    const accuracyY = (v: number) => panelY(v, accuracyBottom, EH / 2);
    const epochTicks = [0, EPOCHS / 4, EPOCHS / 2, 3 * EPOCHS / 4, EPOCHS];
    const axE = new LineField(32);
    const axAccuracy = new LineField(32);
    const curves = new LineField(2 * NC + EPOCHS, { width: 2 });
    const dynE = new LineField(48); // synchronized replay cursors and the dashed random level
    const chartPts = new PointField({ count: 2 * NC + 1, size: 0.15, bokeh: 0, twinkle: 0 });
    gE.add(axE.object, axAccuracy.object, curves.object, dynE.object, chartPts.object);

    // ───────────────────────── 4. functions of time ─────────────────────────
    const live: { family: Family | null; k: number } = { family: FAMS.includes(P.family) ? (P.family as Family) : null, k: +P.k || kDefault };

    // panel visibility (0 … 1) and the pattern's place, all functions of t
    const inA = (t: number) => smooth(0.6, 1.2, t) * (1 - smooth(T3 - 0.4, T3 + 0.3, t));
    // from T5 on, the pattern stays beside training through the summary slide; it jumps sides while it is invisible
    const inD = (t: number) => Math.max(smooth(T2 + 0.6, T2 + 1.2, t) * (1 - smooth(T3 - 0.4, T3 + 0.3, t)), smooth(T5 + 0.4, T5 + 1.1, t));
    const inB = (t: number) => smooth(T3 + 0.2, T3 + 0.8, t) * (1 - smooth(T5 - 0.4, T5 + 0.3, t));
    const inC = (t: number) => smooth(T4 + 0.4, T4 + 1.0, t) * (1 - smooth(T5 - 0.4, T5 + 0.3, t));
    const inE = (t: number) => smooth(T5 + 0.4, T5 + 1.1, t);
    /** The summary drops the accuracy panel; both feature families retain their comparison with initialization. */
    const sumUp = (t: number) => smooth(T7, T7 + 0.6, t);
    const dxAt = (t: number) => (t >= T4 ? DX_L : DX_R);

    /** Epoch on display: 0 until the replay of slide 4.6, then 0 → 400. */
    const epochAt = (t: number) => (t < T6 ? 0 : EPOCHS * easeInOut(smooth(REPLAY0, REPLAY1, t)));
    const cpAt = (ep: number) => {
      let c = 0;
      while (c < NC - 2 && cps[c + 1].epoch <= ep) c++;
      const w = clamp((ep - cps[c].epoch) / Math.max(1, cps[c + 1].epoch - cps[c].epoch));
      return { c, w };
    };

    const uCur = new Float32Array(N);
    /** Project a unit feature, using the same amplitude scale as the original cosine. */
    const project = (c: number, grp: number, f: Float32Array, out: Float32Array) => {
      const Q = cps[c].Q;
      out.fill(0);
      let s = 0;
      for (let r = grp * SH; r < (grp + 1) * SH; r++) {
        let dt = 0;
        const ro = r * N;
        for (let i = 0; i < N; i++) dt += Q[ro + i] * f[i];
        s += dt * dt;
        for (let i = 0; i < N; i++) out[i] += dt * Q[ro + i];
      }
      for (let i = 0; i < N; i++) out[i] *= Math.sqrt(N / 2);
      return s;
    };
    const exact = new Float32Array(N);
    for (let i = 0; i < N; i++) exact[i] = Math.cos((2 * Math.PI * kDefault * (((i / p) | 0) + (i % p))) / p);

    // The epoch cursor moves continuously. The map and its cell probe share one measured projection.
    const state = { kind: -1, fi: 2, grp: -1, k: kDefault, norm2: 1, ep: 0, mapEpoch: 0 };
    const fieldAt = (t: number) => {
      const ep = epochAt(t);
      state.ep = ep;
      const family = t >= T4 ? live.family : null;
      const kind = t < T4 ? 0 : family ? 3 : t < T6 ? 1 : 2;
      const fi = family ? FAMS.indexOf(family) : kind === 1 ? 0 : 2;
      const k = family ? clamp(Math.round(live.k), 1, half) : kDefault;
      const checkpoint = ep >= EPOCHS ? NC - 1 : cpAt(ep).c;
      const mapEpoch = cps[checkpoint].epoch;
      if (state.kind === kind && state.fi === fi && state.k === k && state.mapEpoch === mapEpoch) return;
      state.kind = kind;
      state.fi = fi;
      state.k = k;
      state.mapEpoch = mapEpoch;
      if (kind === 0) {
        state.grp = -1;
        state.norm2 = 1;
        uCur.set(exact);
      } else {
        state.grp = famGroup[fi];
        state.norm2 = project(checkpoint, state.grp, modes[fi][k - 1], uCur);
      }
    };

    // kernel content cache: the instanced buffer is rewritten only when the number of filled rows changes
    let matRows = -1;
    const setMatrixRows = (rows: number) => {
      if (rows === matRows) return;
      matRows = rows;
      mat.values.set(kTone);
      mat.values.fill(0, rows * N);
      mat.commit();
    };
    const SCAN0 = T3 + 0.35, SCAN1 = T3 + 2.3;

    // the caption sentences that depend on the run (each branch audited over the Explore range, see RUNS)
    const capFirst = (g1Init
      ? `Before training, ${inputGroups} already ${inputHolds} almost all the Fourier features in $a$ and in $b$. `
      : `Before training, ${inputGroups} ${inputHolds} a share of the Fourier features in $a$ and in $b$. `) +
      'Lin traces this to the one-hot input encoding and the approximate shift symmetry of the initial kernel. The grid shows the part of one wave captured by the group. The plot measures overlap across the whole family.';
    const capSecond =
      'In the lazy training regime, the empirical kernel stays nearly fixed. ' +
      `Here training ${changeDescription} the ${sumOrdinal} group’s overlap with the sum features, from ${f2(g2Start / FDIM)} (${startComparison}) to ${f2(g2End / FDIM)}.` +
      (gradual ? ' Accuracy on held-out input pairs rises gradually in this run.' : '');
    const capEnd =
      'The known Fourier features overlap a group of kernel eigenvectors. Do the corresponding weight directions matter for prediction? ' +
      'In a separate modular-addition experiment, PIRAMID reports that removing weight directions mapped from the sum-feature eigenspace reduced accuracy much more than an equal-sized random ablation.';
    // the closing board's line for this chapter: the comparison with the kernel before training
    const sayMain =
      `The ${sumOrdinal} group’s overlap with the a + b features starts ${startComparison}. ` +
      `Training ${changeDescription} its overlap, from ${f2(g2Start / FDIM)} to ${f2(g2End / FDIM)}.`;
    // the runs checked outside the film are quoted only at the settings they were run with, and only where this run agrees with them
    const atDefaults = p === 17 && n === 256 && P.frac === 0.8 && P.wd === 0.1 && layer === 'both' && (P.seed as number) >= RUNS.seeds[0] && (P.seed as number) <= RUNS.seeds[1];
    const sayRuns = atDefaults && g1Init && g2AtChance && g2Almost && g2Rise;
    // Lin's timing observation, for the Training note only; this run's shape of held-out accuracy decides the sentence
    const timing =
      `<p>In Lin’s MLP, the growth rate of the sum-feature overlap had a local peak near the onset of grokking, a sudden rise in held-out accuracy. ` +
      (rise >= 0 && gradual
        ? 'Here held-out accuracy climbs gradually, with no sudden jump. That observation is therefore not tested.</p>'
        : 'This experiment does not measure the timing of that peak.</p>');

    const TICK = { color: CSS.white } as const;
    const POST = 'https://www.lesswrong.com/posts/T2REsZneix3bmAKtL/piramid-progress-and-plans';

    const results: Result[] = [
      {
        headline: true,
        group: 'check', // it reproduces Lin's own result at a smaller size
        kind: 'against initialization',
        label: `share of the Fourier features in $a + b$ inside group ${sS + 1}, before → after training (random directions ${f2(chance / FDIM)})`,
        measured: `${f2(g2Start / FDIM)} → ${f2(g2End / FDIM)}`,
        reference: 'paper ($p = 29$): small at initialization, 26.53 of 28 after training, a share of 0.95',
        claim: `how much of the features that carry the answer the ${sumOrdinal} group holds, before and after training`,
        say: sayMain,
      },
      {
        group: 'check',
        label: `share of the Fourier features in $a$ inside group ${sA + 1} before training`,
        measured: f2(ini.ov[sA][0] / FDIM),
        reference: 'paper ($p = 29$): present at initialization, 27.54 of 28 after training, a share of 0.98',
        say: g1Init
          ? `Before any training, ${inputGroups} of the kernel already ${inputHolds} the Fourier features in a and in b, as the paper explains from the inputs.`
          : `Before any training, ${inputGroups} of the kernel ${inputHolds} only part of the Fourier features in a and in b.`,
      },
      {
        group: 'check',
        label: `eigenvalue drops after ${SH} and ${2 * SH} nonconstant modes`,
        measured: `${times(drop1)}, ${times(drop2)}`,
        reference: 'paper: two cliffs of $4\\lfloor p/2\\rfloor$ eigenvectors each (Fig. 1)',
      },
    ];

    const chapter: Chapter = {
      id: 'entk',
      pillar: 'tools',
      kicker: 'INTERPRETABILITY APPLICATIONS',
      title: 'EMPIRICAL NTK',
      source: 'J. Lin · Feature Identification via the Empirical NTK · arXiv:2510.00468',
      links: [
        { text: 'Feature Identification via the Empirical NTK', url: 'https://arxiv.org/abs/2510.00468' },
        { text: 'Progress and Plans', url: POST },
      ],
      duration: DUR,
      group,
      details(t) {
        if (t < T3 || t >= T4) return undefined;
        return {
          title: 'Averaging over answers',
          body: `<p>This experiment averages the kernels for the ${p} answer scores, then finds the eigenvectors of that average. Lin finds the spectrum for each answer separately and averages those spectra.</p>` +
            (atDefaults ? '<p>In four runs at these settings, the separate answer kernels also showed weak initial and strong trained alignment between the second group and the sum features. The two largest gaps in the averaged spectra could move to different ranks.</p>' : ''),
        };
      },
      results,
      beats: [
        {
          at: 0.6,
          caption: `Jennifer Lin asks whether gradients can reveal the features a network learned. She tests the idea on modular addition, where the learned solution is already understood. Her network adds two numbers and takes the remainder modulo 29. This experiment uses the same task modulo ${p}.`,
        },
        {
          at: T2,
          hold: T3 - 0.45, // the sums and the wave fade out from T3 − 0.4
          caption: `The known solution combines <em>Fourier features</em>, cosine and sine waves over the numbers 0 to ${p - 1}. Squaring the hidden activations mixes waves in $a$ and $b$ to produce waves in $a + b$ and $a - b$. The sum waves repeat every ${p}, so they can encode the remainder.`,
        },
        {
          at: T3,
          caption: `Take a small step along the gradient of one answer score on input $x'$. How much does that score move on input $x$? The gradient dot product predicts the change per unit step size. The <em>empirical neural tangent kernel</em> (eNTK) shown here averages those dot products over answer scores.`,
          eq: "\\bar K(x,x')=\\tfrac1p\\sum_{c}\\nabla_{\\theta}f_c(x)\\cdot\\nabla_{\\theta}f_c(x')",
          eqNote: `$f_c$ is the output for answer $c$, one of the $p = ${p}$ answers`,
        },
        {
          at: T4,
          hold: T5 - 0.45, // the kernel and its eigenvalues fade out from T5 − 0.4
          caption: `Each eigenvector is a pattern across input pairs. Its eigenvalue measures how strongly the kernel responds along that pattern. Lin divides the strongest patterns into groups at sharp drops in eigenvalue. She calls each group a cliff. Here the known Fourier families set the two group sizes at ${SH} each. ` +
            (drops
              ? 'The measured spectrum also drops at both boundaries.'
              : 'At least one boundary falls short of a clear spectral drop.'),
          eq: '',
        },
        { at: T5, caption: capFirst },
        { at: T6, hold: T7 - 0.45, caption: capSecond },
        { at: T7, caption: capEnd },
      ],
      shots: [
        { t: 0, pos: [0, 0.25, CAM_D], look: [0, 0.25, 0], fov: 38 },
        { t: DUR, pos: [0, 0.25, CAM_D], look: [0, 0.25, 0], fov: 38 },
      ],
      set(key: string, value: ParamValue) {
        if (key === 'family') live.family = FAMS.includes(value as Family) ? (value as Family) : null;
        if (key === 'k' && isFinite(+value)) live.k = +value;
      },
      update(t, api) {
        fieldAt(t);
        const ep = state.ep;
        const aA = inA(t), aB = inB(t), aC = inC(t), aD = inD(t), aE = inE(t);
        gA.visible = aA > 0.002; gB.visible = aB > 0.002; gC.visible = aC > 0.002; gD.visible = aD > 0.002; gE.visible = aE > 0.002;
        gD.position.x = dxAt(t);
        const W = (g: THREE.Group, x: number, y: number): V3 => [g.position.x + x, g.position.y + y, 0];
        const tickX = (key: string, pos: V3, text: string, alpha: number, color: string = CSS.white) => api.label(key, pos, text, { color, alpha, align: 'center', dy: 38 });
        const tickY = (key: string, pos: V3, text: string, alpha: number, color: string = CSS.white) => api.label(key, pos, text, { color, alpha, align: 'right', dx: 4 });
        const tickR = (key: string, pos: V3, text: string, alpha: number, color: string = CSS.white) => api.label(key, pos, text, { color, alpha, align: 'left', dx: -4 });
        /** A panel's plain name at its top-left corner, with lines under it. */
        const title = (key: string, pos: V3, name: string, subs: string[], alpha: number, help?: string) => {
          const h = 17.6 + 16.7 * subs.length;
          api.label(key, pos, name + subs.map((s) => `<small>${s}</small>`).join(''), { ...TICK, alpha, align: 'left', size: 'md', dx: -12, dy: -(h / 2 + 8), help });
        };

        // ── the sums ──
        if (gA.visible) {
          axA.uIntensity.value = aA;
          const green = Math.max(smooth(2.8, 3.4, t), t >= T7 ? 1 : 0);
          let q = 0;
          for (let x = 0; x < N; x++) {
            const ap = t >= T7 ? 1 : smooth(0.8 + 1.6 * order0[x], 1.3 + 1.6 * order0[x], t);
            const col = isY0(x) ? mixRGB(C.white, C.green, green) : C.white;
            if (run.isTrain[x]) {
              dots.setColor(x, col, ap * aA * (isY0(x) ? 1 + 0.2 * green : 0.85));
              dots.sizes[x] = isY0(x) ? 1 + 0.25 * green : 0.85;
            } else {
              dots.setColor(x, C.white, 0);
              const cc = rings.colors, o = q * RSEG * 6, k = ap * aA * (isY0(x) ? 1 + 0.2 * green : 1);
              for (let j = 0; j < RSEG * 6; j += 3) { cc[o + j] = col[0] * k; cc[o + j + 1] = col[1] * k; cc[o + j + 2] = col[2] * k; }
              q++;
            }
          }
          dots.commit(false, true, true);
          rings.commit();
          title('tA', W(gA, -HS, HS), 'Modular addition', [], aA,
            `<p>Each grid position is one input pair $(a,b)$. The network answers $(a+b) \\bmod ${p}$, the remainder after division by ${p}.</p>`);
          tickMain.forEach((v, i) => {
            tickX('ta' + i, W(gA, gpos(v, 0)[0], -HS - 0.03), String(v), aA);
            tickY('tb' + i, W(gA, -HS - 0.03, gpos(0, v)[1]), String(v), aA);
          });
          tickX('taT', W(gA, gpos((half + p - 1) / 2, 0)[0], -HS - 0.03), '$a$', aA);
          tickY('tbT', W(gA, -HS - 0.03, gpos(0, (half + p - 1) / 2)[1]), '$b$', aA);
          const kx = -HS - 0.8, k1 = aA * (t >= T7 ? 1 : smooth(1.6, 2.2, t));
          api.label('kA2', W(gA, kx, 0.25), 'ring = held out from training', { ...TICK, alpha: k1, align: 'right' });
          api.label('kA3', W(gA, kx, -0.25), 'green = sums with the same answer', { color: CSS.green, alpha: aA * green, align: 'right' });
        }

        // ── one pattern on the grid ──
        if (gD.visible) {
          axD.uIntensity.value = aD;
          psi.uOpacity.value = aD;
          cbP.uOpacity.value = aD;
          const pv = psi.values;
          for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) pv[(p - 1 - b) * p + a] = clamp(uCur[a * p + b], -PSI_CLIP, PSI_CLIP) * PSI_GAIN;
          psi.commit();
          // Mark the diagonal on the exact wave (4.2) and on the trained pattern beside the overlap chart (4.7).
          const dg = t < T3 ? smooth(T2 + 1.6, T2 + 2.2, t) : t >= T7 ? smooth(T7 + 0.9, T7 + 1.5, t) : 0;
          diagD.uIntensity.value = aD * dg;
          diagD.object.visible = dg > 0.002 && state.kind !== 3;
          // from 4.5 on the panel draws the projection of a known feature onto its matched cliff
          const mapTime = state.mapEpoch === 0 ? 'epoch 0 · before training' : state.mapEpoch === EPOCHS ? `epoch ${EPOCHS} · after training` : `epoch ${state.mapEpoch}`;
          const [nameD, sub]: [string, string[]] = state.kind === 0 ? ['Fourier feature in $a + b$', []]
            : [`Projected Fourier feature in ${FAM_WORD[FAMS[state.fi]]}`, [mapTime]];
          title('tD', W(gD, -HS, HS), nameD, sub, aD,
            '<p>Each cell is one input pair $(a,b)$. Color gives the feature\'s value there, using the scale at right.</p>' +
            (state.kind === 0 ? '' : `<p>The map shows the projection into group ${state.grp + 1} at the labeled epoch. It updates at each measured checkpoint. The color scale stays fixed. The plot shows overlap for the whole Fourier family.</p>`));
          tickMain.forEach((v, i) => {
            tickX('da' + i, W(gD, gpos(v, 0)[0], -HS - 0.03), String(v), aD);
            tickY('db' + i, W(gD, -HS - 0.03, gpos(0, v)[1]), String(v), aD);
          });
          tickX('daT', W(gD, gpos((half + p - 1) / 2, 0)[0], -HS - 0.03), '$a$', aD);
          tickY('dbT', W(gD, -HS - 0.03, gpos(0, (half + p - 1) / 2)[1]), '$b$', aD);
          tickR('dc1', W(gD, cbPx + CBAR / 2 + 0.12, cbPy(1)), '+1', aD);
          tickR('dc0', W(gD, cbPx + CBAR / 2 + 0.12, 0), '0', aD);
          tickR('dcm', W(gD, cbPx + CBAR / 2 + 0.12, cbPy(-1)), '−1', aD);
          if (diagD.object.visible) api.label('dG', W(gD, 0, -HS - 0.03), 'green line · same modular sum', { color: CSS.green, alpha: aD * dg, align: 'below', dy: 34 });
        }

        // ── the kernel ──
        if (gB.visible) {
          axB.uIntensity.value = aB;
          const scan = smooth(SCAN0, SCAN1, t);
          setMatrixRows(t < SCAN0 ? 0 : Math.round(scan * N));
          mat.uOpacity.value = K_INK * aB;
          mat.object.visible = t >= SCAN0;
          cbK.uOpacity.value = K_INK * aB;
          scanB.count = 0;
          if (scan > 0 && scan < 1) { const xi = Math.min(N - 1, Math.floor(scan * N)); scanB.push([-HS - 0.2, by(xi), 0], [HS, by(xi), 0], C.amber, 0.9); }
          scanB.commit();
          title('tB', W(gB, -HS, HS), 'Empirical NTK', [], aB,
            `<p>The kernel uses the trained network’s weights. Rows and columns are input pairs $(a,b)$, grouped by $a$. ${kminOff >= 0 ? 'Darker cells have larger values.' : 'Darker cells have larger absolute values. Hover over a cell for its signed value.'} The color uses a square-root scale.</p>` +
            (layer === 'K2' ? '' : '<p>Blocks on the diagonal highlight inputs with the same $a$. Thin lines highlight inputs with the same $b$.</p>'));
          tickMain.forEach((v, i) => {
            tickY('kb' + i, W(gB, -HS - 0.03, HS - ((v * p) / N) * S), `$a = ${v}$`, aB);
            tickX('ka' + i, W(gB, -HS + ((v * p) / N) * S, -HS - 0.03), `$a = ${v}$`, aB);
          });
          api.label('kaT', W(gB, 0, -HS - 0.03), 'input pairs $(a,b)$', { ...TICK, alpha: aB, align: 'center', dy: 60 });
          api.label('kcH', W(gB, cbKx, HS), kminOff >= 0 ? 'max' : 'max $|K|$', { ...TICK, alpha: aB, align: 'center', dy: -13 });
          api.label('kc0', W(gB, cbKx, -HS), kminOff >= 0 ? 'min' : '0', { ...TICK, alpha: aB, align: 'center', dy: 14 });
          // the two callouts, at the right; they leave before the eigenvalues come in beside the kernel
          const co = layer === 'K2' ? 0 : aB * smooth(T3 + 2.8, T3 + 3.4, t) * (1 - smooth(T4 - 0.3, T4 + 0.3, t));
          leadB.count = 0;
          const lx = HS + 1.05, yBlock = tgtBlock[1] + 0.35, yLine = tgtLine[1] + 0.6;
          if (co > 0.002) {
            leadB.push([lx - 0.05, yBlock, 0], [tgtBlock[0] + 0.12, tgtBlock[1] + 0.08, 0.02], C.white, 0.9 * co);
            // the outline of that one block: the p × p pairs of sums with the same a
            const b0 = -HS + ((blockA * p) / N) * S, b1 = -HS + (((blockA + 1) * p) / N) * S;
            leadB.polyline([[b0, HS - (b0 + HS), 0.02], [b1, HS - (b0 + HS), 0.02], [b1, HS - (b1 + HS), 0.02], [b0, HS - (b1 + HS), 0.02]], C.white, 0.9 * co, true);
            // one block of pairs (a, b) and (a + 3, b'): its diagonal is the thin line of pairs that share b
            const aR = ((iLine / p) | 0), aCo = ((jLine / p) | 0);
            const lx0 = -HS + ((aCo * p) / N) * S, lx1 = -HS + (((aCo + 1) * p) / N) * S, ly0 = HS - ((aR * p) / N) * S, ly1 = HS - (((aR + 1) * p) / N) * S;
            leadB.polyline([[lx0, ly0, 0.02], [lx1, ly0, 0.02], [lx1, ly1, 0.02], [lx0, ly1, 0.02]], C.white, 0.9 * co, true);
            leadB.push([lx - 0.05, yLine, 0], [lx1 + 0.02, (ly0 + ly1) / 2, 0.02], C.white, 0.9 * co);
          }
          leadB.commit();
          markB.uOpacity.value = co;
          markB.object.visible = co > 0.002;
          api.label('kc1', W(gB, lx, yBlock), `shared $a = ${blockA}$`, { ...TICK, alpha: co, align: 'left', dx: -4 });
          api.label('kc2', W(gB, lx, yLine), 'shared $b$', { ...TICK, alpha: co, align: 'left', dx: -4 });
        }

        // ── its eigenvalues (the trained kernel) ──
        if (gC.visible) {
          const rise = smooth(T4 + 0.55, T4 + 1.85, t), annIn = smooth(T4 + 1.65, T4 + 2.4, t) * aC, dropIn = smooth(T4 + 2.4, T4 + 2.95, t) * aC;
          const lv = fin.logv;
          for (let r = 0; r < NB; r++) {
            const grp = groupOf(r);
            const up = rise <= 0 ? 0 : easeInOut(clamp(rise * 1.5 - 0.5 * (r / NB)));
            const on = smooth(0, 0.2, up) * aC;
            const X = barX(r), Y = lerp(-HS, barY(lv[r]), up);
            spec.setPos(r, X, Y, 0.03);
            spec.setColor(r, groupCol[grp], on * (grp === 3 ? 0.75 : 1.0));
            spec.sizes[r] = grp === 3 ? 0.8 : 1.0;
            stems.seg(r, [X, -HS, 0], [X, Y, 0], groupCol[grp], 0.42 * on);
          }
          spec.commit();
          stems.count = NB;
          stems.commit();
          const shadeK = (THEME === 'paper' ? 0.13 : 0.06) * annIn;
          for (const sh of [shade1, shade2]) { sh.uK.value = shadeK; sh.mesh.visible = shadeK > 0.001; }
          dynC.count = 0;
          dynC.polyline([[0, -HS, 0], [CW, -HS, 0], [CW, HS, 0], [0, HS, 0]], C.white, 0.6 * aC, true);
          for (const v of yTicks) {
            const y = barY(Math.log10(v));
            dynC.push([0, y, 0], [-0.12, y, 0], C.white, 0.85 * aC);
            dynC.push([0, y, 0], [CW, y, 0], C.white, 0.2 * aC);
          }
          for (const r of xTicks) dynC.push([barX(r), -HS, 0], [barX(r), -HS - 0.12, 0], C.white, 0.85 * aC);
          // the broken stem of the largest eigenvalue
          const ob = leadingOffscale ? smooth(T4 + 1.6, T4 + 2.2, t) * aC : 0;
          dynC.push([barX(0) - 0.09, HS - 0.16, 0], [barX(0) + 0.09, HS - 0.08, 0], C.white, 0.9 * ob);
          dynC.push([barX(0) - 0.09, HS - 0.26, 0], [barX(0) + 0.09, HS - 0.18, 0], C.white, 0.9 * ob);
          // The dimension lines mark the change across each group boundary.
          for (const [key, r, d] of [['dr1', SH, drop1], ['dr2', 2 * SH, drop2]] as const) {
            const al = dropIn * smooth(1.04, 1.14, d);
            const x = barX(r) + 0.42, y1 = barY(lv[r]), y2 = barY(lv[r + 1]);
            if (al < 0.01 || lv[r + 1] < LO) continue;
            dynC.push([barX(r) + 0.06, y1, 0], [x + 0.08, y1, 0], C.white, 0.9 * al);
            dynC.push([barX(r + 1) + 0.06, y2, 0], [x + 0.08, y2, 0], C.white, 0.9 * al);
            dynC.push([x, y1, 0], [x, y2, 0], C.white, 1.0 * al);
            api.label(key, W(gC, x, Math.max((y1 + y2) / 2, y2 + 0.3)), times(d), { ...TICK, alpha: al, align: 'left', dx: -4 });
          }
          dynC.commit();
          api.label('cOff', W(gC, barX(0) + 0.12, HS - 0.2), 'Rank 0 is off scale', { color: CSS.slate, alpha: ob, align: 'left', size: 'xs' });
          title('tC', W(gC, 0, HS), 'Trained kernel eigenvalues', ['Relative to rank 1 · log scale'], aC,
            `<p>The trained spectrum is ordered from largest to smallest. The plot shows ranks 0–${NB - 1}, with rank 1 set to 1. Each shaded group has ${SH} ranks, enough for two Fourier families of ${FDIM} features apiece.</p>` +
            (leadingOffscale ? '<p>The largest eigenvalue is above the plot’s scale.</p>' : '') +
            (spectrumClipped ? '<p>Values below the lower limit appear along the bottom edge. Hover to read their values.</p>' : ''));
          yTicks.forEach((v, i) => tickY('cy' + i, W(gC, 0, barY(Math.log10(v))), String(v), aC));
          if (spectrumClipped) api.label('cy-floor', W(gC, CW - 0.12, -HS + 0.27), `≤ ${(10 ** LO).toPrecision(2)} at floor`, { ...TICK, alpha: aC, align: 'right', plate: true, size: 'xs' });
          xTicks.forEach((r, i) => tickX('cx' + i, W(gC, barX(r), -HS), r === NB - 1 ? `rank ${r}` : String(r), aC));
          // each group with what the ground truth expects in it (the film measures the contents on the next two slides)
          const matchedFamilies = (grp: number) => {
            const families = FAMS.filter((_, f) => famGroup[f] === grp).map((f) => FAM_WORD[f]);
            return families.length
              ? `Closest to group ${grp + 1}<small>${families.join(', ')}</small>`
              : `Group ${grp + 1}<small>No best match</small>`;
          };
          tickX('cs1', W(gC, barX((1 + SH) / 2), -HS - 0.95), defaultGrouping ? 'Expected in group 1<small>$a$ and $b$</small>' : matchedFamilies(0), annIn, CSS.gold);
          tickX('cs2', W(gC, barX(SH + (1 + SH) / 2), -HS - 0.95), defaultGrouping ? 'Expected in group 2<small>$a + b$ and $a - b$</small>' : matchedFamilies(1), annIn, CSS.cyan);
        }

        // ── training ──
        if (gE.visible) {
          const alignY = (v: number) => ey(v, t), top = alignmentTop(t);
          const g1 = smooth(T5 + 1.2, T5 + 2.0, t);
          const g1Draw = t >= T6 ? 1 : smooth(T5 + 1.2, T5 + 2.0, t);
          const g2 = smooth(T6, T6 + 0.6, t);
          const hk = 1 - sumUp(t), accuracyAlpha = g2 * aE * hk;
          const eG1 = EPOCHS * g1Draw, eG2 = t >= T7 ? EPOCHS : ep;
          // Both panels use the same epoch positions. Outside the replay, alignment fills the envelope.
          const axes = (field: LineField, bottom: number, ceiling: number, yAt: (v: number) => number) => {
            field.count = 0;
            frameOf(field, 0, bottom, EW, ceiling);
            for (const e of epochTicks) {
              field.push([ex(e), bottom, 0], [ex(e), bottom - 0.1, 0], C.white, 0.85);
              if (e > 0 && e < EPOCHS) field.push([ex(e), bottom, 0], [ex(e), ceiling, 0], C.white, 0.13);
            }
            for (const v of [0, 0.5, 1]) {
              field.push([0, yAt(v), 0], [-0.12, yAt(v), 0], C.white, 0.85);
              if (v !== 0.5) field.push([0, yAt(v), 0], [EW, yAt(v), 0], C.white, 0.2);
            }
            field.commit();
          };
          axes(axE, -EH / 2, top, alignY);
          axes(axAccuracy, accuracyBottom, EH / 2, accuracyY);
          axE.uIntensity.value = aE;
          axAccuracy.uIntensity.value = accuracyAlpha;
          curves.count = 0;
          const seg = (e0: number, v0: number, e1: number, v1: number, col: RGB, k: number, upTo: number, yAt = alignY) => {
            if (e0 >= upTo || k < 0.002) return;
            const w = Math.min(1, (upTo - e0) / Math.max(1e-6, e1 - e0));
            curves.push([ex(e0), yAt(v0), 0], [ex(lerp(e0, e1, w)), yAt(lerp(v0, v1, w)), 0], col, k);
          };
          for (let c = 0; c + 1 < NC; c++) {
            seg(cps[c].epoch, ovS1[c], cps[c + 1].epoch, ovS1[c + 1], C.gold, g1 * aE, eG1);
            seg(cps[c].epoch, ovS2[c], cps[c + 1].epoch, ovS2[c + 1], C.cyan, g2 * aE, eG2 + 1e-6);
          }
          for (let e = 0; e < EPOCHS; e++)
            seg(e, run.testAcc[e], e + 1, run.testAcc[e + 1], C.white, 0.8 * accuracyAlpha, eG2, accuracyY);
          curves.commit();
          for (let c = 0; c < NC; c++) {
            chartPts.setPos(c, ex(cps[c].epoch), alignY(ovS1[c]), 0);
            chartPts.setPos(NC + c, ex(cps[c].epoch), alignY(ovS2[c]), 0);
            chartPts.setColor(c, C.gold, (cps[c].epoch <= eG1 + 1e-6 ? 0.9 : 0) * g1 * aE);
            chartPts.setColor(NC + c, C.cyan, (cps[c].epoch <= eG2 + 1e-6 ? 1 : 0) * g2 * aE);
            chartPts.sizes[c] = chartPts.sizes[NC + c] = 0.8;
          }
          const cur = t >= T6 && t < T7 ? smooth(T6 + 0.4, T6 + 0.8, t) * (1 - smooth(T7 - 0.4, T7, t)) * aE : 0;
          chartPts.setPos(2 * NC, ex(ep), alignY(lerp(ovS2[cpAt(ep).c], ovS2[cpAt(ep).c + 1], cpAt(ep).w)), 0.01);
          chartPts.setColor(2 * NC, C.cyan, 1.2 * cur);
          chartPts.sizes[2 * NC] = 1.25;
          chartPts.commit();
          dynE.count = 0;
          if (cur > 0.002) {
            dynE.push([ex(ep), -EH / 2, 0], [ex(ep), top, 0], C.white, 0.8 * cur);
            dynE.push([ex(ep), accuracyBottom, 0], [ex(ep), EH / 2, 0], C.white, 0.8 * cur);
          }
          const yCh = alignY(chance / FDIM), chK = g1 * aE;
          if (chK > 0.002) for (let x = 0; x < EW; x += 0.24)
            dynE.push([x, yCh, 0], [Math.min(EW, x + 0.12), yCh, 0], GRAY, 0.9 * chK);
          const sk = sumUp(t) * aE, yB = (yCh + alignY(0)) / 2;
          if (sk > 0.002) dynE.push([0.42, yB + 0.08, 0], [0.06, alignY(ovS2[0]) - 0.06, 0], C.cyan, 0.9 * sk);
          dynE.commit();
          title('tE', W(gE, 0, top), 'Fourier feature overlap', [], aE,
            '<p>Each curve shows how much of a Fourier family lies in a fixed-rank group, on a scale from 0 to 1. Partial matches contribute fractions. Dots mark kernel measurements, and lines connect them. The dashed line gives expected overlap with random directions.</p>');
          title('tAccuracy', W(gE, 0, EH / 2), 'Held-out accuracy', [], accuracyAlpha,
            '<p>The curve is computed from this run. It shares its epoch scale and replay cursor with the feature-alignment panel.</p>');
          epochTicks.forEach((e, i) => api.label(`ex${i}`, W(gE, ex(e), -EH / 2), String(e), { ...TICK, alpha: aE, align: 'center', dy: 22 }));
          api.label('exUnit', W(gE, EW / 2, -EH / 2), 'training epoch', { ...TICK, alpha: aE, align: 'center', dy: 43 });
          for (const [i, v] of [0, 0.5, 1].entries()) {
            tickY(`ey${i}`, W(gE, 0, alignY(v)), String(v), aE);
            tickY(`accuracyY${i}`, W(gE, 0, accuracyY(v)), pct(v), accuracyAlpha);
          }
          api.label('lg1', W(gE, ex(0) + 0.2, alignY(0.9)), `$a$ features · group ${sA + 1}`, { color: CSS.gold, alpha: g1 * aE, align: 'left', dx: -4 });
          const eL = 0.52 * EPOCHS;
          api.label('lg2a', W(gE, ex(eL), alignY(0.86)), `$a + b$ features · group ${sS + 1}`, { color: CSS.cyan, alpha: g2 * aE * smooth(0.8 * eL, eL, eG2), align: 'left', dx: -4 });
          api.label('lgR', W(gE, EW - 0.15, yCh), 'random directions', { color: CSS.slate, alpha: chK, align: 'right', dy: -14 });
          api.label('lgB', W(gE, 0.45, yB), 'before training', { color: CSS.cyan, alpha: sk, align: 'left', dx: -4 });
          api.label('lgE', W(gE, EW + 0.2, alignY(ovS2[NC - 1])), 'after training', { color: CSS.cyan, alpha: sk, align: 'left', dx: -4 });
        }
      },
      readouts(t) {
        // Notes may sample the held picture before playback reaches it.
        fieldAt(t);
        const rows: Readout[] = [];
        const ep = state.ep, { c, w } = cpAt(ep);
        const ovAt = (s: number, f: number) => lerp(cps[c].ov[s][f], cps[c + 1].ov[s][f], w);
        if (t >= T4 && t < T5) {
          rows.push(
            { label: `drop after ${SH} nonconstant modes`, value: times(drop1), detail: true },
            { label: `drop after ${2 * SH} nonconstant modes`, value: times(drop2), detail: true },
            { label: 'largest step inside a group', value: times(stepIn), detail: true },
            { label: 'largest eigenvalue against the next', value: times(ratio(fin, 0)), detail: true },
          );
        } else if (t >= T5 && t < T6) {
          rows.push(
            { label: `share of the Fourier features in $a$ inside group ${sA + 1}, before training`, value: f2(ini.ov[sA][0] / FDIM) },
            { label: 'the same for a random set of this many directions', value: f2(chance / FDIM) },
            { label: `share of the Fourier features in $b$ inside group ${sB + 1}, before training`, value: f2(ini.ov[sB][1] / FDIM), detail: true },
          );
        } else if (t >= T6 && t < T7) {
          rows.push(
            { label: `share of the Fourier features in $a + b$ inside group ${sS + 1}, before → after training`, value: `${f2(g2Start / FDIM)} → ${f2(g2End / FDIM)}` },
            { label: 'the same for a random set of this many directions', value: f2(chance / FDIM) },
            { label: 'sum-feature share on the line at this epoch', value: f2(ovAt(sS, 2) / FDIM), detail: true },
            { label: `share of the Fourier features in $a - b$ inside group ${sD + 1}, after training`, value: f2(fin.ov[sD][3] / FDIM), detail: true },
            { label: 'held-out sums correct after training', value: pct(run.testAcc[EPOCHS]), detail: true },
            { label: 'training sums correct after training', value: pct(run.trainAcc[EPOCHS]), detail: true },
          );
        } else if (t >= T7) {
          rows.push(
            { label: `share of the Fourier features in $a + b$ inside group ${sS + 1}, before → after training`, value: `${f2(g2Start / FDIM)} → ${f2(g2End / FDIM)}` },
            { label: 'the same for a random set of this many directions', value: f2(chance / FDIM) },
            { label: "Lin's Gemma-3-270M experiment", value: '13 of 14 grammar labels', note: 'the best eNTK eigenvector has higher held-out AUROC than the best PCA direction on this dataset draw', detail: true },
          );
        }
        if (state.kind === 3 && t >= T4) rows.push({ label: 'part of the drawn feature inside its group', value: f2(state.norm2), detail: true });
        return rows;
      },

      // Notes add context at the time held by the reader, independently of the rendered frame.
      inspect(t) {
        fieldAt(t);
        const hs: Hotspot[] = [];
        const at = (g: THREE.Group, x0: number, yTop: number, lines: number): V3 => [g.position.x + x0 - 30 / PPU, yTop + (24 + 8.4 * lines) / PPU, 0];
        if (inA(t) > 0.5 && t < T2) {
          hs.push({
            id: 'sums',
            pos: at(gA, -HS, HS, 0),
            title: 'Training and unseen sums',
            body:
              `<p>This smaller reconstruction uses ${n} hidden neurons and addition modulo ${p}. Lin's reported run uses 512 neurons and modulus 29. Both networks square their hidden activations.</p>` +
              `<p>The training set is a random selection of ${run.nTrain} of the ${N} possible input pairs. The remaining ${nTest} sums test whether the learned rule works on inputs the network was never trained on.</p>`,
          });
        }
        if (inD(t) > 0.5 && (state.kind !== 2 || t >= T7)) {
          const how = `<p>A single eigenvector can mix several Fourier features. Projecting a known feature onto the whole group tests how much of the feature lies in its span, even when no individual eigenvector matches it.</p>`;
          const body = state.kind === 0
            ? `<p>The wave is $\\cos(2\\pi\\cdot ${kDefault}(a+b)/${p})$. Adding ${p} to the sum changes its phase by whole periods, so it has the same value on all sums with the same remainder.</p>`
            : state.kind === 1
              ? (layer === 'K2'
                  ? `<p>Only the second-layer kernel is selected. Group ${sA + 1} is matched to the features in $a$ using the trained kernel. The pattern shows one feature's part in that group before training.</p>`
                  : `<p>Before training, the expected first-layer kernel entry depends on which operand positions match. It is zero when neither matches and largest for identical input pairs. Shifting all first operands or all second operands by the same amount preserves this pattern. The Fourier waves in $a$ and $b$ reflect that symmetry.</p>` +
                    `<p>The first-layer weights deliberately start much smaller than the second-layer weights. For the combined kernel this makes the first-layer contribution dominate at initialization, exposing structure inherited from the inputs.</p>`) +
                how
              : state.kind === 2
                ? `<p>PIRAMID’s “Progress and Plans” describes mapping the sum-feature eigenspace into weight space using the Jacobian, the matrix of output gradients. The report leaves the kernel variant, architecture, modulus and number of removed directions unspecified.</p>` +
                  `<p>A direction over input-output pairs can be lifted to parameter space with the transpose of the full output Jacobian. The averaged kernel here gives directions over inputs alone.</p>` +
                  `<p>Lin also tested Gemma-3-270M on 1,024 TinyStories windows. At each layer, she compared 25 directions from gradients with respect to the MLP output weights with 25 PCA directions from activations. For each grammar label, Lin picked the best signed direction on one half of the data and evaluated it on the other.</p>`
                : `<p>The selected family is matched to whichever group overlaps it more after training. The pattern shows this feature's part in that group at the displayed epoch.</p>` + how;
          hs.push({ id: 'pattern', pos: at(gD, -HS, HS, state.kind === 0 ? 0 : 1), title: state.kind === 0 ? 'Why the wave follows the answer' : state.kind === 2 ? 'From feature matches to interventions' : 'A Fourier feature inside a rank group', body });
        }
        if (inB(t) > 0.5) {
          hs.push({
            id: 'kernel',
            pos: at(gB, -HS, HS, 0),
            title: 'Structure inherited from the inputs',
            body:
              `<p>Computing the kernel uses the inputs and network weights, without looking at the correct answers.</p>` +
              (layer === 'K2'
                ? `<p>Only the second-layer kernel is selected. Each entry sums products of squared hidden activations. The values are nonnegative.</p>`
                : `<p>The first-layer contribution contains $x \\cdot x'$, which vanishes unless the two inputs share $a$ or share $b$. This accounts for the blocks and lines.</p>` +
                  (layer === 'both' ? '<p>The second-layer contribution sums products of squared hidden activations, which are nonnegative.</p>' : '')),
          });
        }
        if (inC(t) > 0.5) {
          hs.push({
            id: 'eigenvalues',
            pos: at(gC, 0, HS, 1),
            title: 'Why these group sizes are fixed',
            body:
              `<p>Each Fourier family has a cosine and a sine at each of ${half} frequencies, giving ${FDIM} features. A group containing two families therefore needs ${SH} directions. The nearly constant leading eigenvector is excluded from these groups.</p>` +
              `<p>This experiment follows the same ranks back to initialization. The eigenvalue ratio after rank ${2 * SH} changes from ${f3(ratio(ini, 2 * SH))} before training to ${f2(drop2)} after training. The trained spectrum’s two largest adjacent ratios occur after ranks ${largestDrops.join(' and ')}, excluding rank 0 and flooring values near numerical zero.</p>` +
              (atDefaults ? `<p>In ${RUNS.spectralMatches} of ${RUNS.n} runs at these settings, the two largest drops in the trained kernel fell at the Fourier-sized boundaries. None of the 32 initial spectra had both largest drops there.</p>` : '') +
              (defaultGrouping ? '' : `<p>Each family is matched to whichever group overlaps it more after training. That match can differ with the selected kernel.</p>`),
          });
        }
        if (inE(t) > 0.5 && t < T7) {
          hs.push({
            id: 'training',
            pos: at(gE, 0, alignmentTop(t), 0),
            title: t < T6 ? 'What the overlap share measures' : 'What the training curve establishes',
            body: t < T6
              ? `<p>The overlap share adds the squared projections of a family's ${FDIM} Fourier features onto the group's ${SH} eigenvectors, then divides by ${FDIM}. It ranges from 0 to 1. Partial matches contribute fractions, so the share is not a count of features found. A random group of ${SH} directions would hold a share of ${f2(chance / FDIM)}.</p>`
              : `<p>Lin's run modulo 29 reaches an overlap of 26.53 out of 28 for the sum features, a share of 0.95. The comparison with initialization shows how much alignment training adds.</p>` +
                (sayRuns ? `<p>At these default settings, the same qualitative pattern appeared in ${RUNS.n} random starts and training splits. The first group held the features in $a$ and in $b$ before training, and the second went from about the random level to almost all the features in $a + b$.</p>` : '') +
                timing,
          });
        }
        return hs;
      },

      probe(t, pi) {
        fieldAt(t);
        const pl = pi.plane;
        if (!pl) return null;
        const [X, Y] = pl;
        const tol = 10 * pi.unit;
        const epi = Math.min(EPOCHS, Math.max(0, Math.round(state.ep)));
        // the sums
        if (inA(t) > 0.5) {
          const x0 = X - gA.position.x;
          if (x0 >= -HS && x0 < HS && Y >= -HS && Y < HS) {
            const a = Math.floor((x0 + HS) / cA), b = Math.floor((Y + HS) / cA), x = a * p + b;
            return {
              pos: W0(gA, gpos(a, b)),
              title: `The sum $${a} + ${b}$`,
              rows: [
                { label: `answer $(a + b) \\bmod ${p}$`, value: String((a + b) % p) },
                { label: 'used for', value: run.isTrain[x] ? 'training' : 'held out' },
                { label: `answered correctly after ${EPOCHS} epochs`, value: run.correct[EPOCHS * N + x] === 1 ? 'yes' : 'no' },
              ],
            };
          }
        }
        // the kernel
        if (inB(t) > 0.5 && t >= SCAN1) {
          const x0 = X - gB.position.x;
          if (x0 >= -HS && x0 < HS && Y > -HS && Y <= HS) {
            const j = Math.min(N - 1, Math.floor((x0 + HS) / cB)), i = Math.min(N - 1, Math.floor((HS - Y) / cB));
            const ai = (i / p) | 0, bi = i % p, aj = (j / p) | 0, bj = j % p;
            const k = Kfinal[i * N + j], share = (ai === aj ? 1 : 0) + (bi === bj ? 1 : 0);
            return {
              pos: W0(gB, [bx(j), by(i), 0.02]),
              title: `Kernel entry for $${ai} + ${bi}$ and $${aj} + ${bj}$`,
              rows: [
                { label: 'value', value: f3(k) },
                { label: 'against the largest entry', value: f2(k / kmax) },
                { label: 'matching operand positions', value: String(share) },
              ],
            };
          }
        }
        // the eigenvalues
        if (inC(t) > 0.5 && t > T4 + 1.85) {
          const x0 = X - gC.position.x;
          if (x0 >= 0.15 && x0 < 0.15 + NB * barW) {
            const r = Math.min(NB - 1, Math.floor((x0 - 0.15) / barW));
            const top = barY(fin.logv[r]);
            if (Y >= -HS - tol && Y <= Math.max(top, -HS + 0.15) + tol) {
              const grp = groupOf(r);
              return {
                pos: W0(gC, [barX(r), top, 0.03]),
                title: `Eigenvalue of rank ${r}`,
                rows: [
                  { label: 'value', value: `$${sci(fin.lam[r])}$` },
                  { label: 'against the next one', value: times(ratio(fin, r)) },
                  { label: 'rank group', value: grp === 0 ? leadingOffscale ? 'leading mode, off scale' : 'leading mode' : grp === 3 ? 'below both groups' : `group ${grp}` },
                ],
              };
            }
          }
        }
        // the pattern
        if (inD(t) > 0.5) {
          const x0 = X - gD.position.x;
          if (x0 >= -HS && x0 < HS && Y >= -HS && Y < HS) {
            const a = Math.floor((x0 + HS) / cA), b = Math.floor((Y + HS) / cA);
            const v = uCur[a * p + b];
            const rows: Readout[] = [
              { label: `answer $(a + b) \\bmod ${p}$`, value: String((a + b) % p) },
              { label: state.kind === 0 ? 'feature value' : 'projected feature value', value: f2(v) + (Math.abs(v) > PSI_CLIP ? ' (clipped)' : '') },
            ];
            if (state.kind !== 0) rows.push(
              { label: 'projection measured at epoch', value: String(state.mapEpoch) },
              { label: 'share of this feature inside its group', value: f2(state.norm2) },
            );
            const fam = FAMS[state.fi], arg = fam === 'a' ? a : fam === 'b' ? b : fam === 'sum' ? a + b : a - b;
            rows.push({ label: 'the exact feature here', value: f2(Math.cos((2 * Math.PI * state.k * arg) / p)) });
            return { pos: W0(gD, gpos(a, b)), title: `The sum $${a} + ${b}$`, rows } satisfies Probe;
          }
        }
        // training
        if (inE(t) > 0.5) {
          const x0 = X - gE.position.x;
          if (x0 > -tol && x0 < EW + tol && Y > -EH / 2 - tol && Y < EH / 2 + tol) {
            const split = splitAt(t) > 0.5, onAccuracy = split && Y >= accuracyBottom;
            if (split && Y > alignmentTop(t) && Y < accuracyBottom) return null;
            const e = Math.round(clamp(x0 / EW) * EPOCHS), { c, w } = cpAt(e);
            const sumShare = lerp(ovS2[c], ovS2[c + 1], w), inputShare = lerp(ovS1[c], ovS1[c + 1], w);
            const beforeReplay = t < T6;
            const nearestShare = beforeReplay || Math.abs(Y - ey(inputShare, t)) < Math.abs(Y - ey(sumShare, t)) ? inputShare : sumShare;
            return {
              pos: W0(gE, [ex(e), onAccuracy ? accuracyY(run.testAcc[e]) : ey(nearestShare, t), 0.01]),
              title: `Epoch ${e}`,
              rows: beforeReplay ? [
                { label: `share of the Fourier features in $a$ inside group ${sA + 1}`, value: f2(inputShare) },
                { label: 'share for random directions', value: f2(chance / FDIM) },
              ] : [
                { label: 'held-out sums correct', value: `${pct(run.testAcc[e])} (${Math.round(run.testAcc[e] * nTest)} of ${nTest})` },
                { label: `share of the Fourier features in $a + b$ inside group ${sS + 1}`, value: f2(lerp(ovS2[c], ovS2[c + 1], w)) },
                { label: `share of the Fourier features in $a$ inside group ${sA + 1}`, value: f2(lerp(ovS1[c], ovS1[c + 1], w)) },
              ],
              note: !beforeReplay && epi < e && t < T7 ? 'The replay has not reached this epoch yet.' : 'Shares are joined linearly between the points at which the kernel was computed.',
            };
          }
        }
        return null;
      },
    };
    /** Panel-local point → world. */
    function W0(g: THREE.Group, q: V3): V3 { return [g.position.x + q[0], g.position.y + q[1], q[2]]; }
    return chapter;
  },
};

export default entk;
