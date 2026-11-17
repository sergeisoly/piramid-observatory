import * as THREE from 'three/webgpu';
import { attribute, uniform } from 'three/tsl';
import type { Chapter, ChapterFactory, FigureSource, FrameApi, Hotspot, LabelOpts, NoteDetails, ParamValue, Readout, Result } from '../engine/types';
import { loadFigureImage } from '../engine/figure-image';
import { resolveParams } from '../engine/params';
import { C, CSS, mixRGB, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { MatrixField } from '../kit/shapes';
import { mulberry32, gaussian, smooth, windowed, clamp, lerp } from '../lib/math';
import * as PM from './parity.math';

/**
 * PARITY BOTTLENECK: the Deep Parity Bottleneck of Mack et al. (arXiv:2607.20652) in miniature.
 * Six slides on one plane:
 *   6.1  the bottleneck in front of an MLP: x, the sparse code over the dictionary, the rebuilt vector
 *   6.2  an index turned into a strip of signs by the parity hash (left panel)
 *   6.3  the same slide gains its right panel: the inner products of all pairs of directions the hash can name,
 *        against the proven bound (with the same check at the paper's width), and why they matter
 *   6.4  an illustration of the encoder's search on a vector with three planted features and eight decoy
 *        coordinates: keeping the 4 largest coordinates against keeping 12 against an exact search, as three tallies
 *   6.5  the paper's evidence: SAEBench scores, then (6.6) circuit edits on three tasks
 * Everything in 6.1 to 6.4 is computed in build(); 6.5 and 6.6 report the paper’s measurements.
 */

const T_EDIT = 32.35, DUR = 34.65;
const WID = 14.08, XL = -WID / 2, XR = WID / 2; // the d coordinates span this width in slides 6.1 and 6.4
const FX0 = -10.0, FX1 = 7.4, FYT = 6.8; // panel frame: left, right, rule under the title
const MARG = XL - 0.15; // right edge of the row labels in the left margin
const HIGH = 4; // extra index bits of the hashed level: dictionary m = 16·d
const K1 = 8; // hashed directions the encoder keeps
const SIGMA = 0.03;
const COEF = [2.0, -1.5, 1.8]; // planted hashed features
const PCOEF = [1.3, -1.15, 1.2, 0.95]; // the large coordinates planted with them
const GEN_SEED = 20652;
// slide 6.4: extra large coordinate components competing for parent slots, and the wider search
const DECOYS = 8, DECOY_AMP = 1.3, K0_WIDE = 12;
const POS: RGB = C.amber, NEG: RGB = C.white; // sign of a value or of a ±1 coordinate
const SEL: RGB = C.cyan; // picked by the seed, or kept by the encoder
const BK = 0.72; // tile amplitude of a ± pattern

// slide time windows (seconds)
// 6.2 and 6.3 share one figure: PB and PC are one window, the hash first (6.2) and the inner products from PC0 on (6.3)
const PA = [0.3, 8.4], PB = [8.4, 21.4], PC = PB, PC0 = 12.9, PD = [21.4, 29.8], PE = [29.8, DUR];
const FADE = 0.5;
const T_ALL = [6, 8, 10]; // log2 of the values the control for d offers
const T_PAPER = 10; // log2 of the paper's width for its 200M model (its 1.3B model has d = 2,048)

const int = (n: number) => Math.round(n).toLocaleString('en-US');
/** Bytes as MiB (or KiB below one MiB), with two significant decimals at most: 3,932,160 → "3.75 MiB". */
const fmtMiB = (b: number) => (b >= 2 ** 20 ? `${+(b / 2 ** 20).toFixed(2)} MiB` : `${+(b / 2 ** 10).toFixed(2)} KiB`);
const sg = (v: number, digits = 2) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(digits);
const expT = (v: number, digits = 1) => { // 5.498e+11 → 5.498 × 10^{11}, inside $…$
  const [m, e] = v.toExponential(digits).split('e');
  return `${m}\\times 10^{${Number(e)}}`;
};
const bitsStr = (i: number, n: number) => i.toString(2).padStart(n, '0').replace(/(.{4})(?=.)/g, '$1 ');
const span = (css: string, s: string) => `<span style="color:${css}">${s}</span>`;
const bump = (x: number, m: number, w: number) => Math.exp(-(((x - m) / w) ** 2));
const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const nw = (n: number) => (n >= 0 && n < WORDS.length ? WORDS[n] : int(n));
type V3 = [number, number, number];

/** Additive flat rectangles with per-vertex color (bars of a chart). Same conventions as kit/LineField. */
class QuadField {
  readonly positions: Float32Array;
  readonly colors: Float32Array;
  readonly object: THREE.Mesh;
  private pA: THREE.BufferAttribute;
  private cA: THREE.BufferAttribute;
  private geo = new THREE.BufferGeometry();
  private n = 0;
  constructor(readonly max: number) {
    this.positions = new Float32Array(max * 18);
    this.colors = new Float32Array(max * 18);
    this.pA = new THREE.BufferAttribute(this.positions, 3);
    this.cA = new THREE.BufferAttribute(this.colors, 3);
    this.pA.setUsage(THREE.DynamicDrawUsage);
    this.cA.setUsage(THREE.DynamicDrawUsage);
    this.geo.setAttribute('position', this.pA);
    this.geo.setAttribute('color', this.cA);
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = (attribute('color', 'vec3') as any).mul(uniform(1));
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = false;
    m.blending = THREE.AdditiveBlending;
    m.side = THREE.DoubleSide;
    this.object = new THREE.Mesh(this.geo, m);
    this.object.frustumCulled = false;
    this.object.visible = false;
    this.geo.setDrawRange(0, 0);
  }
  clear() {
    this.n = 0;
  }
  rect(x0: number, y0: number, x1: number, y1: number, c: ArrayLike<number>, k = 1) {
    if (this.n >= this.max || k <= 0.002) return;
    const p = this.positions, q = this.colors, j = this.n * 18;
    p[j] = x0; p[j + 1] = y0; p[j + 3] = x1; p[j + 4] = y0; p[j + 6] = x1; p[j + 7] = y1;
    p[j + 9] = x0; p[j + 10] = y0; p[j + 12] = x1; p[j + 13] = y1; p[j + 15] = x0; p[j + 16] = y1;
    for (let v = 0; v < 6; v++) {
      p[j + v * 3 + 2] = 0;
      q[j + v * 3] = c[0] * k; q[j + v * 3 + 1] = c[1] * k; q[j + v * 3 + 2] = c[2] * k;
    }
    this.n++;
  }
  commit() {
    this.geo.setDrawRange(0, this.n * 6);
    this.object.visible = this.n > 0; // WebGPU warns on zero-vertex draws
    this.pA.needsUpdate = true;
    this.cA.needsUpdate = true;
  }
}

// paper: Table 4, SAEBench at layer 12 of the 1.3 B model (Appendix G: PT-Large-2L-Aux; the SAEs are trained on the
// dense 1.3B baseline): parity bottleneck (PB), TopK SAE, Matryoshka SAE.
// `s` keeps the paper's digits; `help` gives the film's explanation of each test (the paper gives none).
const SAEBENCH: { name: string; v: [number, number, number]; s: [string, string, string]; help: string; up: boolean; snr: boolean }[] = [
  { name: 'sparse probing', v: [0.802, 0.786, 0.769], s: ['0.802', '0.786', '0.769'], help: 'Sparse probing measures the accuracy of a probe that reads one feature.', up: true, snr: false },
  { name: 'AutoInterp', v: [0.796, 0.859, 0.891], s: ['0.796', '0.859', '0.891'], help: 'In AutoInterp, a language model predicts where a feature fires from its explanation.', up: true, snr: false },
  { name: 'SCR', v: [0.06, 0.175, 0.398], s: ['0.060', '0.175', '0.398'], help: 'Spurious correlation removal (SCR) tests how much a biased probe improves after zeroing 20 features.', up: true, snr: true },
  { name: 'TPP', v: [0.001, 0.08, 0.275], s: ['0.001', '0.080', '0.275'], help: 'Targeted probe perturbation (TPP) tests whether zeroing 20 features associated with one class hurts that class’s probe while preserving performance on other classes.', up: true, snr: true },
  { name: 'absorption', v: [0.01, 0.807, 0.18], s: ['0.01', '0.807', '0.180'], help: 'The absorption test checks how much first-letter signal other features carry when the main letter features account for too little of it.', up: false, snr: false },
];
const SAE_MODEL = ['parity bottleneck', 'TopK sparse autoencoder', 'Matryoshka sparse autoencoder'];
// paper: Section 3.2, Figure 2 and Appendix C. Greedy feature edits (the protocol of Makelov et al., 2024) on three
// circuits: k = 1, 2 or 4 edits, each the single change that brings the activation closest to the target prompt's;
// success is the share of pairs passing a task-specific logit comparison. The paper gives the rates in a figure.
// The examples follow the paper's prompt templates (C.1); the names, the occupation and the year are the film's.
const CIRCUITS = [
  { name: 'indirect object (IOI)', example: 'When Mary and John went to the store,<br>John gave a drink to → Mary' },
  { name: 'gendered pronouns', example: 'The nurse said that → she' },
  { name: 'greater-than', example: 'The war lasted from 1732 to 17<br>→ a digit above 3' },
];
// paper: the 1.3B models. PT-Large-2L reaches validation loss 2.808 on 20B tokens, the matched dense baseline 2.828 on 3.0B;
// Table 3 gives Large-2L 1.22× time per token. With the 6.7× token overhead in Table 2, total time is about 8×.
const LOSS_PB = '2.808', LOSS_DENSE = '2.828', COST = 'about eight';
const CIRCUIT_SUCCESS = 'For indirect object identification, success means the target recipient outranks the unchanged subject. For gendered pronouns and greater-than, the target token must outrank the source token.';
const CIRCUIT_FIGURE = {
  asset: 'figures/parity/circuit-edit-success-figure2.png',
  alt: `Figure 2 compares the parity bottleneck with four sparse autoencoders at one, two and four edits. The parity bottleneck has the highest reported success rate in each group. ${CIRCUIT_SUCCESS}`,
};
const CIRCUIT_SOURCE: FigureSource = {
  ...CIRCUIT_FIGURE,
  title: 'Circuit-edit success rates · Figure 2',
  actionLabel: 'Mack et al. · Figure 2',
  images: [{ ...CIRCUIT_FIGURE, caption: CIRCUIT_SUCCESS }],
  credit: 'Andrew Mack, Kraig Yuheng Tou, Mark Henry, Zhengxun Wu and Lauren Greenspan. Scaling Interpretable Transformers with Parity Bottleneck Layers, arXiv:2607.20652v1, Figure 2. CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Rendered from the authors’ standalone vector figure; axes, legend and plotted marks unchanged.',
  sourceURL: 'https://arxiv.org/abs/2607.20652v1',
};

const parity: ChapterFactory = {
  id: 'parity',
  label: 'parity · hash, inner products, encoder search',
  controls: [
    { key: 'i', label: '$i$  feature index', kind: 'range', min: 0, max: 65535, step: 1, value: 26925, live: true, hint: 'Hashes any index modulo $d^2$. Each row of the seed matrix selects index bits whose parity gives one sign.' },
    { key: 'd', label: '$d$  residual dimension', kind: 'select', value: 256, options: [{ value: 64, label: '64' }, { value: 256, label: '256' }, { value: 1024, label: '1,024' }], hint: 'Sets the seed matrix to $d \\times 2\\log_2 d$, the dictionary to $m = 16d$ directions and the bound to $(1+2\\sqrt d)/d$. The overlap plot checks all $d^2$ hashed directions for this $d$.' },
    { key: 'K0', label: '$K_0$  level-0 beam', kind: 'range', min: 1, max: 12, step: 1, value: 4, hint: 'Number of coordinates of $x$ that the encoder keeps as parents. With fewer parents it can miss planted children.' },
    { key: 'delta', label: '$\\Delta$  children per parent', kind: 'range', min: 4, max: 64, step: 4, value: 32, hint: 'Number of shared XOR generators $g_j$. The encoder scores $K_0 \\cdot \\Delta$ candidates.' },
    { key: 'seed', label: 'seed  planted vector', kind: 'range', min: 1, max: 200, step: 1, value: 1, hint: 'Redraws the three planted features, their parents and the noise.' },
  ],

  async build(ctx, params) {
    const P = resolveParams(parity.controls, params);
    const d: number = T_ALL.some((tt) => 1 << tt === Number(P.d)) ? Number(P.d) : 256;
    const t8 = Math.round(Math.log2(d)), T2 = 2 * t8;
    const K0: number = clamp(Math.round(Number(P.K0)) || 4, 1, 12);
    const DELTA: number = clamp(Math.round(Number(P.delta)) || 32, 1, 64);
    const SEED: number = Math.round(Number(P.seed)) || 1;
    const IMASK = d * d - 1;
    const LB = t8 + HIGH, mDict = 1 << LB, GROWS = 1 << HIGH;

    // ───────────────────────── the mathematics ─────────────────────────
    const M = PM.seedMatrix(t8);
    await ctx.log('seed ✓  overlaps of all pairs');
    const coh = T_ALL.map((tt) => PM.coherence(tt)); // slide 6.3 compares the active width with the paper's width
    const cohD = coh[T_ALL.indexOf(t8)];
    await ctx.log(`overlaps ✓ ${cohD.max.toFixed(4)} ≤ ${cohD.bound.toFixed(4)}  encoder search`);
    // the same exhaustive check at the paper's width (a finite check of this hash family, apart from the theorem)
    const cohP = coh[T_ALL.indexOf(T_PAPER)];
    // memory: the seed as packed bits; the directions stored as 4-byte numbers, both all the hash can name and
    // only the dictionary the miniature instantiates (level 1). Arithmetic only: no speed is measured.
    const seedBytes = (d * T2) / 8, storedBytes = cohD.features * d * 4;
    const dictBytes = (mDict - d) * d * 4;

    const gens = PM.makeGenerators(mulberry32(GEN_SEED), DELTA, t8, HIGH);
    const sq = 1 / Math.sqrt(d);
    const mkInput = (seed: number, decoys = 0) => {
      const rnd = mulberry32(seed);
      const pl = PM.plant(M, gens, t8, COEF, PCOEF, SIGMA, rnd, gaussian(rnd));
      // decoys: extra large coordinates of about the planted parents' size
      const used = new Set(pl.parents.map((q) => q.index)), decoyAt: number[] = [];
      for (let nd = 0; nd < decoys;) {
        const q = Math.floor(rnd() * d);
        if (used.has(q)) continue;
        used.add(q);
        decoyAt.push(q);
        pl.x[q] += (rnd() < 0.5 ? -1 : 1) * DECOY_AMP * (0.85 + 0.3 * rnd());
        nd++;
      }
      const enc = PM.encode(M, gens, pl.x, K0, K1);
      // decoder: x̂ = Σ a_i φ_i over the active set, a = coordinate (level 0) or score (level 1)
      const xh = new Float64Array(d), s = new Int8Array(d);
      for (const p of enc.parents) xh[p] += pl.x[p];
      for (const c of enc.kept) { PM.hashSigns(M, c.child, s); for (let k = 0; k < d; k++) xh[k] += c.z * s[k] * sq; }
      let xx = 0, hh = 0, xhd = 0, mx = 1e-9;
      for (let k = 0; k < d; k++) { xx += pl.x[k] ** 2; hh += xh[k] ** 2; xhd += pl.x[k] * xh[k]; mx = Math.max(mx, Math.abs(pl.x[k])); }
      const norm = Math.sqrt(xx / Math.max(hh, 1e-12)); // the paper rescales x̂ to the norm of x
      const tile = (v: number) => Math.sign(v) * (0.3 + 0.52 * Math.pow(Math.min(1, Math.abs(v) / (0.8 * mx)), 0.7));
      const disp = new Float32Array(d), dispH = new Float32Array(d);
      for (let k = 0; k < d; k++) { disp[k] = tile(pl.x[k]); dispH[k] = tile(xh[k] * norm); }
      const act = [
        ...enc.parents.map((p) => ({ row: 0, col: p, a: pl.x[p] })),
        ...enc.kept.map((c) => ({ row: c.child >> t8, col: c.child & (d - 1), a: c.z })),
      ];
      return { pl, enc, mx, disp, dispH, act, xh, norm, decoyAt, cos: xhd / Math.sqrt(Math.max(xx * hh, 1e-12)) };
    };
    const main = mkInput(SEED * 7919 + 1, DECOYS);
    const inputs = [mkInput(SEED * 7919 + 1001), mkInput(SEED * 7919 + 2001), mkInput(SEED * 7919 + 3001), main];
    const NIN = inputs.length;
    const enc = main.enc, NC = enc.cands.length, ND = enc.distinct.length, NK = enc.kept.length;

    // exhaustive reference: every hashed score from one WHT
    const all = PM.allScores(M, main.pl.x, LB);
    const exOrder = Array.from({ length: mDict - d }, (_, n) => n + d).sort((a, b) => Math.abs(all[b]) - Math.abs(all[a]));
    const exTop = new Set(exOrder.slice(0, K1));
    const planted = main.pl.children.map((c) => {
      const s = enc.distinct.find((q) => q.child === c.index);
      return { ...c, slot: s ? s.rank : -1, kept: !!s && s.kept, z: s ? s.z : all[c.index] };
    });
    const NP = planted.length;
    const recovered = planted.filter((p) => p.kept).length;
    const plantedInExhaustive = planted.filter((p) => exTop.has(p.index)).length;
    // the same vector, the encoder keeping 12 coordinates instead of 4
    const wide = PM.encode(M, gens, main.pl.x, K0_WIDE, K1);
    const recWide = planted.filter((p) => wide.kept.some((s) => s.child === p.index)).length;
    const compareWide = K0 !== K0_WIDE;
    // A DAG child is considered if any parent proposes it, including a parent other than
    // the one used to plant it. Selection among scored children is a separate step.
    const parentKept = planted.filter((p) => p.slot >= 0).length;
    const isPlanted = new Uint8Array(ND);
    for (const p of planted) if (p.slot >= 0) isPlanted[p.slot] = 1;
    const max3 = cohD.max.toFixed(3), bound3 = cohD.bound.toFixed(3);
    const maxP = cohP.max.toFixed(4), boundP = cohP.bound.toFixed(4);
    const results: Result[] = [
      {
        // the headline: an exhaustive check of the paper's hash at the paper's width, against the paper's theorem
        label: `largest inner product between two hashed directions, at ${int(cohP.d)} coordinates, the width of the paper's smaller model`,
        measured: `${maxP} vs ${boundP}`,
        reference: `proven upper bound ${boundP}, checked on all ${int(cohP.features)} directions`,
        headline: true, group: 'check', kind: 'exhaustive check',
        say: cohP.max <= cohP.bound
          ? `At the width of the paper's smaller model, every pair of the ${int(cohP.features)} hashed directions is as close to orthogonal as the paper's proven bound requires.`
          : `At the width of the paper's smaller model, some pairs of the ${int(cohP.features)} hashed directions are further from orthogonal than the paper's proven bound allows.`,
      },
      {
        label: `the same in this miniature, ${d} coordinates`, measured: `${max3} vs ${bound3}`,
        reference: `proven upper bound ${bound3}`, group: 'check',
      },
      {
        // an illustration of the encoder's mechanism: keeping 12 is built to succeed (199 of 200 draws), no test of the paper
        label: `planted features found next to ${DECOYS} decoy coordinates: keeping ${K0} coordinates${compareWide ? ` vs ${K0_WIDE}` : ''} vs an exact search`,
        measured: `${recovered}${compareWide ? ` vs ${recWide}` : ''} vs ${plantedInExhaustive} of ${NP}`,
        reference: `${parentKept} of ${NP} planted features have at least one retained parent and enter the score ranking`,
        group: 'picture', kind: 'illustration',
      },
      { label: 'scores of the planted features, found vs planted', measured: planted.map((p) => (p.kept ? sg(p.z) : '×')).join(', '), reference: `planted ${planted.map((p) => sg(p.coef, 1)).join(', ')}`, group: 'picture' },
    ];
    await ctx.log(`search ✓ ${NC} scored, ${recovered}/${NP} planted  building slides`);

    // ───────────────────────── live indices (kept for the hidden controls) ─────────────────────────
    const live = { i: Math.round(Number(P.i)) & IMASK };
    const sgI = new Int8Array(d); // the signs of the index that slide 6.2 hashes
    const relive = () => { PM.hashSigns(M, live.i, sgI); };
    relive();

    // ───────────────────────── shared drawing surfaces ─────────────────────────
    const group = new THREE.Group();
    const pitch = WID / d, X0 = XL + pitch / 2;
    const dens = clamp(Math.sqrt(pitch / 0.055), 0.5, 1.5); // keeps dense hairline sets readable for d = 1024
    const cx = (k: number) => X0 + k * pitch;
    const wall = (rows: number, cols: number, cell: number, rowPitch: number, x: number, y: number, neg: RGB, pos: RGB, gap = 0.08) => {
      const f = new MatrixField({ rows, cols, cell, gap, height: 0, neg, pos });
      f.object.rotation.x = Math.PI / 2; // tiles face the camera, row 0 on top
      f.object.scale.set(1, 1, rowPitch / cell);
      f.object.position.set(x, y, 0);
      group.add(f.object);
      return f;
    };
    const strip = (h: number, y: number, neg: RGB = NEG, pos: RGB = POS) => wall(1, d, pitch, h, 0, y, neg, pos);
    const show = (f: MatrixField, a: number) => { f.object.visible = a > 0.002; f.uOpacity.value = a; };

    const hair = new LineField(2 * d + NC + 900); // axes, ticks, gridlines, dense thin data
    const fat = new LineField(NC + 700, { width: 2 }); // emphasized data and arrows
    const quads = new QuadField(Math.max(NC, 64) + 160);
    const marks = new PointField({ count: NC + K0 + 32, size: 0.1, bokeh: 0, falloff: 6 });
    group.add(quads.object, hair.object, fat.object, marks.object);
    let nMarks = 0;
    const mark = (x: number, y: number, c: RGB, k: number, size: number) => {
      if (nMarks >= marks.count) return;
      marks.setPos(nMarks, x, y, 0);
      marks.setColor(nMarks, c, k);
      marks.sizes[nMarks++] = size;
    };
    const arrow = (a: V3, b: V3, c: RGB, k: number, head = 0.13) => {
      const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      fat.push(a, b, c, k);
      fat.push(b, [b[0] - ux * head - uy * head * 0.55, b[1] - uy * head + ux * head * 0.55, 0], c, k);
      fat.push(b, [b[0] - ux * head + uy * head * 0.55, b[1] - uy * head - ux * head * 0.55, 0], c, k);
    };
    const dash = (lf: LineField, a: V3, b: V3, c: RGB, k: number, n: number) => {
      for (let q = 0; q < n; q++) {
        const u0 = q / n, u1 = (q + 0.55) / n;
        lf.push([lerp(a[0], b[0], u0), lerp(a[1], b[1], u0), 0], [lerp(a[0], b[0], u1), lerp(a[1], b[1], u1), 0], c, k);
      }
    };
    const box = (lf: LineField, x0: number, y0: number, x1: number, y1: number, c: RGB, k: number) =>
      lf.polyline([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], c, k, true);
    const L = (api: FrameApi, key: string, pos: V3, html: string, alpha: number, color: string = CSS.white, align: LabelOpts['align'] = 'left', size: LabelOpts['size'] = 'sm', dx = 0, dy = 0, help?: string, figure?: FigureSource) => {
      if (alpha > 0.02) api.label(key, pos, html, { alpha: Math.min(1, alpha), color, align, size, dx, dy, help, figure });
    };
    const title = (api: FrameApi, key: string, text: string, alpha: number, x = FX0, y = FYT, help?: string, figure?: FigureSource) => L(api, key, [x, y + 0.3, 0], text, alpha, CSS.white, 'left', 'md', -9, 0, help, figure);
    /** The rule under a slide's title, over the width of that slide's figure. */
    const rule = (x0: number, x1: number, alpha: number, y = FYT) => { if (alpha > 0.002) hair.push([x0, y, 0], [x1, y, 0], C.white, 0.55 * alpha); };
    /** The status of a panel's evidence, set small at the right end of its title rule. */
    const tag = (api: FrameApi, key: string, x: number, y: number, text: string, alpha: number) => L(api, key, [x, y + 0.3, 0], text, alpha, CSS.slate, 'right', 'sm', 0);
    const FYT_B = 6.45, FYT_C = 6.45; // the rules of the two panels of 6.2 sit at one height, closer to their figures

    // ───────────────────────── 6.1: the bottleneck in front of an MLP ─────────────────────────
    const YAX = 6.1, YAG = 3.95, RPA = 0.17, YAH = 1.6, YAB = 0.36;
    const aX = strip(0.36, YAX), aH = strip(0.36, YAH);
    const aAct = wall(GROWS, d, pitch, RPA, 0, YAG, NEG, POS, 0.0);
    const yA = (row: number) => YAG + (GROWS / 2 - row - 0.5) * RPA;
    const CYC0 = PA[0] + 0.7, CYC = 1.6;
    const cycle = (t: number) => {
      const n = clamp(Math.floor((t - CYC0) / CYC), 0, NIN - 1);
      return { n, c: t - (CYC0 + n * CYC) };
    };
    const updA = (t: number, api: FrameApi) => {
      const A = windowed(t, PA[0], PA[1], FADE);
      show(aX, A); show(aH, A); show(aAct, A);
      if (A <= 0.002) return;
      const { n, c } = cycle(t), cur = inputs[n], prev = n > 0 ? inputs[n - 1] : null;
      const e1 = smooth(0, 0.4, c), out = 1 - smooth(0.25, 0.55, c), inn = smooth(0.5, 0.9, c), e3 = smooth(1.0, 1.4, c);
      for (let k = 0; k < d; k++) {
        const xa = prev ? prev.disp[k] : 0, ha = prev ? prev.dispH[k] : 0;
        aX.values[k] = xa + (cur.disp[k] - xa) * e1;
        aH.values[k] = ha + (cur.dispH[k] - ha) * e3;
      }
      aX.commit(); aH.commit();
      for (let r = 0; r < GROWS; r++) quads.rect(XL, yA(r) - RPA * 0.42, XR, yA(r) + RPA * 0.42, C.white, 0.1 * A); // the dictionary slots, one band per d entries
      aAct.values.fill(0);
      const put = (inp: typeof main, a: number) => {
        if (a <= 0.01) return;
        for (const f of inp.act) {
          aAct.values[f.row * d + f.col] = Math.sign(f.a) * (0.6 + 0.4 * Math.min(1, Math.abs(f.a) / 2)) * a;
          const x = cx(f.col), y = yA(f.row), h = 0.16;
          fat.polyline([[x - h, y - h, 0], [x + h, y - h, 0], [x + h, y + h, 0], [x - h, y + h, 0]], SEL, 1.0 * a * A, true);
        }
      };
      if (prev) put(prev, out);
      put(cur, inn);
      aAct.commit();

      // The decoder restores the input norm on a separate path from the sparse code.
      const xa = XL + 0.6, top = YAG + (GROWS / 2) * RPA, bot = YAG - (GROWS / 2) * RPA;
      const normX = XR + 0.32, attentionX = XR + 0.8;
      const aNorm = A * smooth(PA[0] + 1.5, PA[0] + 2.2, t);
      arrow([xa, YAX - 0.26, 0], [xa, top + 0.1, 0], C.white, (0.75 + 0.5 * bump(c, 0.45, 0.2)) * A);
      arrow([xa, bot - 0.1, 0], [xa, YAH + 0.26, 0], C.white, (0.75 + 0.5 * bump(c, 1.0, 0.2)) * A);
      arrow([-3.5, YAH - 0.26, 0], [-3.5, YAB + 0.4, 0], C.white, (0.75 + 0.5 * bump(c, 1.5, 0.2)) * A);
      hair.polyline([[-6, YAB - 0.34, 0], [-1, YAB - 0.34, 0], [-1, YAB + 0.34, 0], [-6, YAB + 0.34, 0]], C.white, (0.8 + 0.4 * bump(c, 1.6, 0.2)) * A, true);
      hair.polyline([[2.0, YAB - 0.34, 0], [6.4, YAB - 0.34, 0], [6.4, YAB + 0.34, 0], [2.0, YAB + 0.34, 0]], C.white, 0.8 * A, true);
      fat.push([XR + 0.05, YAX, 0], [attentionX, YAX, 0], C.white, 0.8 * A);
      fat.push([attentionX, YAX, 0], [attentionX, YAB, 0], C.white, 0.8 * A);
      arrow([attentionX, YAB, 0], [6.45, YAB, 0], C.white, 0.8 * A);
      fat.push([normX, YAX, 0], [normX, YAH, 0], C.ice, 0.95 * aNorm);
      arrow([normX, YAH, 0], [XR + 0.04, YAH, 0], C.ice, 0.95 * aNorm);
      hair.push([XL, yA(0) - RPA / 2, 0], [XR, yA(0) - RPA / 2, 0], C.white, 0.7 * A);

      const mid1 = (YAX - 0.26 + top + 0.1) / 2, mid2 = (bot - 0.1 + YAH + 0.26) / 2;
      const T0 = PA[0];
      title(api, 't-a', 'A parity bottleneck before an MLP', A, FX0, FYT,
        `<p>The encoder selects a few features from the dictionary. Cyan boxes mark the active entries of the sparse code. The decoder adds their weighted directions and rescales the sum to the input norm.</p>` +
        `<p>Level 0 contains ${int(d)} coordinate directions. Level 1 contains ${int(mDict - d)} hashed directions from the ${int(cohD.features)} the hash can name. The strips above and below the code show the original and reconstructed vectors.</p>`);
      rule(FX0, attentionX, A);
      tag(api, 'a-status', attentionX, FYT, 'Illustration', A);
      L(api, 'a-x', [MARG, YAX, 0], 'Residual stream $x$', A, CSS.white, 'right', 'md');
      L(api, 'a-enc', [xa + 0.15, mid1, 0], 'encoder', A * smooth(T0 + 0.7, T0 + 1.4, t));
      L(api, 'a-code', [-1.6, mid1, 0], 'Sparse code', A * smooth(T0 + 1.1, T0 + 1.8, t), CSS.cyan, 'left');
      L(api, 'a-norm', [XR, mid1, 0], '$\\lVert x \\rVert$', aNorm, CSS.ice, 'right');
      const dA = A * smooth(T0 + 0.9, T0 + 1.6, t);
      L(api, 'a-l0', [MARG, yA(0), 0], 'level 0 · coordinates', dA, CSS.white, 'right');
      L(api, 'a-l1', [MARG, (yA(1) + yA(GROWS - 1)) / 2, 0], 'level 1 · hashed directions', dA, CSS.white, 'right');
      L(api, 'a-dec', [xa + 0.15, mid2, 0], 'decoder', A * smooth(T0 + 1.5, T0 + 2.2, t));
      L(api, 'a-xh', [MARG, YAH, 0], 'reconstructed input', A * smooth(T0 + 1.7, T0 + 2.4, t), CSS.white, 'right', 'md');
      L(api, 'a-mlp', [-3.5, YAB, 0], 'MLP', A, CSS.white, 'center', 'md', 0, 17);
      L(api, 'a-att', [4.2, YAB, 0], 'attention', A * smooth(T0 + 2.1, T0 + 2.8, t), CSS.white, 'center', 'md', 0, 17);
      L(api, 'a-leg', [XR, YAX + 0.2, 0], `${span(CSS.amber, '■ above zero')}&nbsp;&nbsp;${span(CSS.slate, '■ below zero')}`, A, CSS.white, 'right', 'sm', 0, -12); // above the right end of the x strip
    };

    // ───────────────────────── 6.2 left: an index becomes a strip of signs ─────────────────────────
    const NB = T2, LW = 6.4, LX0 = -9.7, LX1 = LX0 + LW, LXC = (LX0 + LX1) / 2, BWC = LW / NB;
    const YBIT = 5.05, BH = 0.42, YBX0 = 3.45, YBX1 = 4.15, BXX0 = LX0, BXX1 = LX1;
    const YBIG = 2.55, YTHIN = 1.8, TH = 0.3;
    const WIN = NB, WIN0 = Math.min(64, d - WIN); // the row of large cells enlarges signs WIN0 … WIN0 + WIN − 1
    let kStar = WIN0 + (WIN >> 1); // the sign the figure follows: one whose seed row picks a handful of bits
    for (let k = WIN0 + 2; k < WIN0 + WIN - 2; k++) { const pc = PM.popcount(M[k]); if (pc >= 4 && pc <= 8) { kStar = k; break; } }
    const cellX = (c: number) => LX0 + (c + 0.5) * BWC;
    const pitchL = LW / d, thinX = (k: number) => LX0 + (k + 0.5) * pitchL;
    const bThin = wall(1, d, pitchL, TH, LXC, YTHIN, NEG, POS);
    const TBH = PB[0] + 0.3;
    const hashT = (t: number) => ({
      idx: smooth(TBH, TBH + 0.5, t), box: smooth(TBH + 0.6, TBH + 1.0, t), pick: smooth(TBH + 1.1, TBH + 1.6, t),
      one: smooth(TBH + 1.7, TBH + 2.2, t), sweep: clamp((t - (TBH + 2.4)) / 0.9),
    });
    const pickedOnes = () => PM.popcount(M[kStar] & live.i); // picked bits of the index that are 1
    const updBL = (t: number, api: FrameApi, A: number) => {
      const h = hashT(t), idx = live.i;
      // the index as bits, most significant first
      for (let c = 0; c < NB; c++) {
        const p = NB - 1 - c, bit = (idx >> p) & 1, picked = (M[kStar] >> p) & 1;
        const x0 = LX0 + c * BWC + 0.03, x1 = LX0 + (c + 1) * BWC - 0.03, y0 = YBIT - BH / 2, y1 = YBIT + BH / 2;
        if (bit) quads.rect(x0, y0, x1, y1, picked ? mixRGB(C.white, SEL, h.pick) : C.white, lerp(0.14, 0.45, picked ? h.pick : 0) * A * h.idx);
        box(hair, x0, y0, x1, y1, C.white, 0.6 * A * h.idx);
        if (picked && h.pick > 0.01) {
          box(fat, x0, y0, x1, y1, SEL, h.pick * A);
          fat.push([cellX(c), y0 - 0.02, 0], [cellX(c), lerp(y0, YBX1, h.pick), 0], SEL, 0.85 * h.pick * A);
        }
        L(api, 'b-bit' + c, [cellX(c), YBIT, 0], String(bit), A * h.idx, CSS.white, 'middle', 'md');
      }
      L(api, 'b-idx', [LX0, YBIT + 0.65, 0], `Index ${int(idx)}`, A * h.idx, CSS.white, 'left', 'md', -9);
      L(api, 'b-pick', [LX1, YBIT + 0.42, 0], `bits for sign ${kStar}`, A * h.pick, CSS.cyan, 'right', 'sm', 9);
      // the seed
      quads.rect(BXX0, YBX0, BXX1, YBX1, C.white, 0.05 * A * h.box);
      box(hair, BXX0, YBX0, BXX1, YBX1, C.white, 0.9 * A * h.box);
      L(api, 'b-seed', [(BXX0 + BXX1) / 2, (YBX0 + YBX1) / 2, 0], 'seed matrix', A * h.box, CSS.white, 'middle', 'md');
      // the sign the figure follows: parity of the picked bits that are 1
      const cS = kStar - WIN0, xs = cellX(cS), ones = pickedOnes(), odd = ones & 1;
      if (h.one > 0.01) arrow([xs, YBX0, 0], [xs, lerp(YBX0, YBIG + BH / 2 + 0.03, h.one), 0], SEL, h.one * A, 0.11);
      L(api, 'b-par', [xs + 0.06, (YBX0 + YBIG + BH / 2) / 2, 0], `${ones} ${ones === 1 ? 'one' : 'ones'} → ${odd ? '−' : '+'}`, A * h.one, CSS.cyan, 'left', 'sm');
      // the enlarged row of signs, then every sign of the direction
      const shownBig = (c: number) => (WIN0 + c === kStar ? h.one : clamp(h.sweep * 2.5 * WIN - c));
      for (let c = 0; c < WIN; c++) {
        const k = WIN0 + c, s = sgI[k], a = shownBig(c) * A;
        const x0 = LX0 + c * BWC + 0.03, x1 = LX0 + (c + 1) * BWC - 0.03, y0 = YBIG - BH / 2, y1 = YBIG + BH / 2;
        quads.rect(x0, y0, x1, y1, s > 0 ? POS : C.slate, (s > 0 ? 0.75 : 0.55) * a);
        box(hair, x0, y0, x1, y1, C.white, 0.5 * A * Math.max(h.one, h.sweep > 0 ? 1 : 0));
        if (k === kStar) box(fat, x0 - 0.02, y0 - 0.02, x1 + 0.02, y1 + 0.02, SEL, h.one * A);
        L(api, 'b-s' + c, [cellX(c), YBIG, 0], s > 0 ? '+' : '−', a, CSS.white, 'middle', 'lg');
      }
      const thinA = A * smooth(TBH + 2.3, TBH + 2.6, t);
      show(bThin, thinA);
      if (thinA > 0.002) {
        for (let k = 0; k < d; k++) bThin.values[k] = k < h.sweep * d || (k >= WIN0 && k < WIN0 + WIN && shownBig(k - WIN0) > 0.5) ? sgI[k] * BK : 0;
        bThin.commit();
        // the funnel from the enlarged row to its place in the strip
        const yb = YBIG - BH / 2 - 0.04, yt = YTHIN + TH / 2 + 0.04, w0 = thinX(WIN0) - pitchL / 2, w1 = thinX(WIN0 + WIN - 1) + pitchL / 2;
        hair.push([LX0, yb, 0], [w0, yt, 0], C.white, 0.45 * thinA);
        hair.push([LX1, yb, 0], [w1, yt, 0], C.white, 0.45 * thinA);
        box(hair, w0, YTHIN - TH / 2 - 0.04, w1, YTHIN + TH / 2 + 0.04, C.white, 0.8 * thinA);
      }
      L(api, 'b-thin', [LXC, YTHIN - TH / 2 - 0.1, 0], `the direction: ${d} signs`, thinA, CSS.white, 'below', 'sm');
    };

    // ───────────────────────── 6.3: overlaps between all pairs of directions ─────────────────────────
    const RX0 = -1.6, RX1 = 6.4, RXC = (RX0 + RX1) / 2, HW = (RX1 - RX0) / 2;
    const RXT = RX0 - 0.6; // left edge of the slide's title and rule, clear of the axis labels
    const CY0 = 1.75, CH = 3.85;
    const CXH = 0.3 / 2 ** ((t8 - 6) / 2), SX = HW / CXH; // the overlap axis spans ±0.3, ±0.15 or ±0.075
    const CNA = cohD.atoms.length, CBW = 0.48 * (2 / d) * SX; // half-width of a bar (the values sit 4/d apart)
    const maxMass = cohD.atoms.reduce((m, a) => Math.max(m, a.mass), 0);
    const yStep = maxMass > 0.1 ? 0.05 : maxMass > 0.04 ? 0.02 : 0.01, yTop = Math.ceil(maxMass / yStep - 1e-9) * yStep, CSC = CH / yTop;
    const maxSide = cohD.atoms[CNA - 1].ip >= -cohD.min ? 1 : -1; // the side of the largest |overlap|
    const extAtom = maxSide > 0 ? cohD.atoms[CNA - 1] : cohD.atoms[0];
    const TBC = PC0 + 0.4, barStart = (a: number) => TBC + 0.3 + 0.9 * (a / CNA), TMAX = TBC + 1.6;
    const ovX = (ip: number) => RXC + ip * SX;
    const maxTip = (): V3 => [ovX(maxSide * cohD.max) + maxSide * 0.03, CY0 + extAtom.mass * CSC + 0.08, 0];
    const maxTail = (): V3 => { const tip = maxTip(); return [tip[0] + maxSide * 0.95, tip[1] + 1.0, 0]; };
    const updBR = (t: number, api: FrameApi, A: number) => {
      const ax = smooth(TBC, TBC + 0.4, t) * A, mk = smooth(TMAX, TMAX + 0.5, t) * A;
      if (ax <= 0.002) return;
      hair.push([RX0, CY0, 0], [RX1, CY0, 0], C.white, 0.85 * ax);
      hair.push([RX0, CY0, 0], [RX0, CY0 + CH + 0.06, 0], C.white, 0.85 * ax);
      for (let n = 1; n * yStep <= yTop + 1e-9; n++) {
        const y = CY0 + n * yStep * CSC;
        hair.push([RX0, y, 0], [RX0 - 0.1, y, 0], C.white, 0.9 * ax);
        hair.push([RX0, y, 0], [RX1, y, 0], C.white, 0.18 * ax);
        L(api, 'c-y' + n, [RX0 - 0.1, y, 0], `${Math.round(n * yStep * 100)}%`, ax * 0.85, CSS.white, 'right');
      }
      L(api, 'c-yt', [RX0, CY0 + CH + 0.32, 0], 'share of pairs', ax, CSS.white, 'left', 'sm', -9);
      const xv = (2 / 3) * CXH, xd = xv < 0.095 ? 2 : 1;
      [-xv, 0, xv].forEach((v, n) => {
        hair.push([ovX(v), CY0, 0], [ovX(v), CY0 - 0.08, 0], C.white, 0.9 * ax);
        L(api, 'c-x' + n, [ovX(v), CY0 - 0.1, 0], v === 0 ? '0' : sg(v, xd), ax * 0.85, CSS.white, 'below');
      });
      L(api, 'c-xt', [RXC, CY0 - 0.68, 0], 'Inner product', ax, CSS.white, 'below', 'md');
      // one bar per value the overlap can take
      cohD.atoms.forEach((at, a) => {
        const g = smooth(barStart(a), barStart(a) + 0.5, t);
        if (g > 0) quads.rect(ovX(at.ip) - CBW, CY0, ovX(at.ip) + CBW, CY0 + at.mass * CSC * g, C.slate, 0.9 * A); // neutral: amber means a plus sign on the left
      });
      // the bound and the largest overlap
      if (mk > 0.002) {
        for (const s of [-1, 1]) fat.push([ovX(s * cohD.bound), CY0 - 0.1, 0], [ovX(s * cohD.bound), CY0 + CH + 0.12, 0], C.slate, mk);
        L(api, 'c-bound', [ovX(maxSide * cohD.bound), CY0 + CH + 0.12, 0], `Proven bound<small>${bound3}</small>`, mk, CSS.slate, maxSide > 0 ? 'left' : 'right', 'md', 0, -14); // above the top gridline
        arrow(maxTail(), maxTip(), C.green, mk, 0.16);
        L(api, 'c-max', maxTail(), `Largest magnitude<small>${max3}</small>`, mk, CSS.green, maxSide > 0 ? 'left' : 'right', 'md');
      }
      // the same check at the paper's width, under the axis: per width, the largest inner product as a bar and the
      // proven bound as a tick, on one shared scale
      const mkP = smooth(TMAX + 0.7, TMAX + 1.2, t) * A;
      if (mkP > 0.002) {
        const rows = cohP === cohD ? [cohD] : [cohD, cohP];
        const GX0 = RX0 + 1.6, GX1 = RX1 - 1.3, GV = Math.max(...rows.map((c) => c.bound)) * 1.05;
        const gx = (v: number) => GX0 + (v / GV) * (GX1 - GX0);
        L(api, 'c-g-t', [GX0, CY0 - 1.64, 0], `${span(CSS.green, '▬')} largest magnitude&nbsp;&nbsp;${span(CSS.slate, '|')} proven bound`, mkP, CSS.white, 'left', 'sm');
        rows.forEach((c, n) => {
          const y = CY0 - 2.20 - 0.62 * n, paper = c === cohP;
          hair.push([GX0, y, 0], [gx(c.bound), y, 0], C.white, 0.25 * mkP);
          quads.rect(GX0, y - 0.07, gx(c.max), y + 0.07, C.green, 0.9 * mkP);
          fat.push([gx(c.bound), y - 0.15, 0], [gx(c.bound), y + 0.15, 0], C.slate, mkP);
          L(api, 'c-g' + n, [GX0 - 0.15, y, 0], paper ? `${int(c.d)} coordinates (paper width)` : `${int(c.d)} coordinates`, mkP, paper ? CSS.green : CSS.white, 'right', 'sm');
          L(api, 'c-gv' + n, [gx(c.bound) + 0.15, y, 0], `${paper ? maxP : max3} under ${paper ? boundP : bound3}`, mkP, paper ? CSS.green : CSS.white, 'left', 'sm');
        });
      }
    };
    const updB = (t: number, api: FrameApi) => {
      const A = windowed(t, PB[0], PB[1], FADE);
      if (A <= 0.002) { show(bThin, 0); return; }
      title(api, 't-bl', 'From an index to a direction', A, FX0, FYT_B,
        `<p>The index uses ${NB} bits. Shaded cells are 1 bits, and cyan outlines show which bits the seed matrix picks for sign ${kStar}. An even number of picked 1s gives a plus sign, and an odd number gives a minus sign.</p>` +
        `<p>The seed matrix needs ${int(seedBytes)} bytes as packed bits. The large signs enlarge part of the strip below. The full direction has one sign per coordinate, scaled by $1/\\sqrt d$ to give a unit vector.</p>`);
      rule(FX0, LX1 + 0.3, A, FYT_B);
      updBL(t, api, A);
    };
    const updC = (t: number, api: FrameApi) => {
      const A = windowed(t, PC[0], PC[1], FADE);
      if (A <= 0.002) return;
      const aT = A * smooth(TBC - 0.2, TBC + 0.3, t); // the second panel comes in once the hash is drawn
      title(api, 't-br', 'Hashed direction overlaps', aT, RXT, FYT_C,
        `<p>Each bar gives the share of direction pairs with that inner product. Zero means the directions are orthogonal. A feature along one direction contributes its coefficient times this inner product to the other direction's score.</p>` +
        `<p>The green mark identifies the largest absolute inner product, and gray ticks mark the proven bound. The comparison below uses the same scale at each width.</p>`);
      rule(RXT, RX1 + 1.0, aT, FYT_C);
      tag(api, 't-br-tag', RX1 + 1.0, FYT_C + 0.72, 'Computed for this presentation', aT);
      updBR(t, api, A);
    };

    // ───────────────────────── 6.4: the encoder's search ─────────────────────────
    const DY0 = 5.85, DYN = 4.95, DYL = 3.9, DYB = 2.3, XS = 0.7 / main.mx;
    // Keep the plot clear of the hierarchy above it and the recovery tallies below it.
    const scoreHalfHeight = 1.44;
    const maxScore = enc.cands.reduce((max, c) => Math.max(max, Math.abs(c.z)), 0);
    const scoreLimit = Math.max(2.4, maxScore + 0.14);
    const scoreTick = Math.floor(scoreLimit);
    const scoreBottom = DYB - scoreHalfHeight;
    const DS = scoreHalfHeight / scoreLimit;
    const order = [...enc.parents].sort((a, b) => a - b), GW = WID / order.length;
    const leafX = new Float32Array(NC);
    enc.cands.forEach((c, n) => { leafX[n] = XL + order.indexOf(c.parent) * GW + GW * (0.05 + 0.9 * ((c.j + 0.5) / DELTA)); });
    const leafW = (GW * 0.9) / DELTA;
    const isParent = new Uint8Array(d);
    enc.parents.forEach((p) => { isParent[p] = 1; });
    // the large coordinates: 1 = a planted feature's parent, 2 = a decoy (drawn thick; decoys in neutral gray)
    const large = new Uint8Array(d);
    main.pl.parents.forEach((q) => { large[q.index] = 1; });
    main.decoyAt.forEach((q) => { large[q] = 2; });
    const cut = NK > 0 ? Math.abs(enc.kept[NK - 1].z) : 0; // the K1-th largest |z|
    // Labels of the planted features: one line each, centered between two gridlines of the score axis.
    // A feature that two coordinates propose has two bars, and a label can sit on either side of a bar. A place
    // counts if the label stays inside the axis and crosses no tall bar. The search below takes, for every
    // planted feature, the first such place that also keeps the labels clear of one another.
    const LAB_W = 2.7, LAB_H = 0.2; // LAB_W: estimated width of a label in world units
    type PLabel = { right: boolean; x: number; y: number; x0: number; x1: number; n: number }; // n: the bar it names
    const pOpts: PLabel[][] = planted.map((p) => {
      const band = (p.z < 0 ? -1 : 1) * Math.min(Math.floor(Math.abs(p.z)) + 0.5, scoreLimit - 0.3);
      const opts: PLabel[] = [];
      enc.cands.forEach((c, n) => {
        if (c.child !== p.index) return;
        for (const right of [false, true]) { // right: the label ends at the bar
          const x = leafX[n] + (right ? -0.12 : 0.12);
          opts.push({ right, x, y: DYB + band * DS, x0: right ? x - LAB_W : x, x1: right ? x : x + LAB_W, n });
        }
      });
      const tall = (c: PM.Candidate) => c.z * band > 0 && Math.abs(c.z) > Math.abs(band) - 0.4; // a bar that reaches the label's band
      const clear = opts.filter((o) => o.x0 >= XL && o.x1 <= XR + 0.3 && !enc.cands.some((c, n) => leafX[n] > o.x0 && leafX[n] < o.x1 && tall(c)));
      return clear.length ? clear : opts.slice(0, 1); // empty: no kept coordinate proposed this feature
    });
    const pLab: (PLabel | null)[] = planted.map((_, n) => pOpts[n][0] ?? null);
    const placeLabel = (n: number, done: PLabel[]): boolean => {
      if (n === planted.length) return true;
      if (!pOpts[n].length) return placeLabel(n + 1, done);
      for (const o of pOpts[n]) {
        if (done.some((q) => q.y === o.y && o.x0 < q.x1 && q.x0 < o.x1)) continue;
        if (placeLabel(n + 1, [...done, o])) { pLab[n] = o; return true; }
      }
      return false;
    };
    placeLabel(0, []); // if no arrangement is clear, every label keeps its first place
    // A feature that two coordinates propose has a second bar. That bar gets a short label of its own
    // ("the +2.0 feature again"), so that a tall bar without a name is never read as another feature.
    const DUP_W = 2.3;
    const dupLab: { p: number; l: PLabel }[] = [];
    planted.forEach((p, k) => {
      const main = pLab[k];
      if (!main) return;
      enc.cands.forEach((c, n) => {
        if (c.child !== p.index || n === main.n) return;
        const done = [...pLab.filter((l): l is PLabel => l !== null), ...dupLab.map((q) => q.l)];
        const opts = [false, true].map((right) => {
          const x = leafX[n] + (right ? -0.12 : 0.12);
          return { right, x, y: main.y, x0: right ? x - DUP_W : x, x1: right ? x : x + DUP_W, n };
        });
        const ok = (o: PLabel) => o.x0 >= XL && o.x1 <= XR + 0.3 && !done.some((q) => q.y === o.y && o.x0 < q.x1 && q.x0 < o.x1)
          && !enc.cands.some((cc, m) => m !== n && leafX[m] > o.x0 && leafX[m] < o.x1 && cc.z * (o.y - DYB) > 0 && Math.abs(cc.z) * DS > Math.abs(o.y - DYB) - 0.4 * DS);
        dupLab.push({ p: k, l: opts.find(ok) ?? opts[0] });
      });
    });
    const pLabUp = [...pLab.filter((l): l is PLabel => l !== null), ...dupLab.map((q) => q.l)].sort((a, b) => a.y - b.y);
    // times of the steps: coordinates, the kept ones, their hashed directions, scores, the cut, planted features
    const TB = { x: PD[0] + 0.3, sel: PD[0] + 1.1, fan: PD[0] + 1.9, score: PD[0] + 3.6, prune: PD[0] + 5.0, green: PD[0] + 5.9 };
    const fanStart = (n: number) => TB.fan + 0.9 * (order.indexOf(enc.cands[n].parent) / order.length) + 0.7 * (enc.cands[n].j / DELTA);
    const scoreStart = (n: number) => TB.score + 0.9 * ((leafX[n] - XL) / WID);
    const lostAny = planted.some((q) => q.slot < 0);
    /** One tally: a box per planted feature, filled in ink when found, outlined in red when missed. */
    const tallyBoxes = (found: number) => Array.from({ length: NP }, (_, k) =>
      `<span style="display:inline-block;width:0.8em;height:0.8em;margin-right:0.18em;vertical-align:-0.08em;box-sizing:border-box;border:2px solid ${k < found ? CSS.white : CSS.red};background:${k < found ? CSS.white : 'transparent'}"></span>`).join('');
    const TALLY = [
      { found: recovered, how: `${K0} parent${K0 === 1 ? '' : 's'}` },
      ...(compareWide ? [{ found: recWide, how: `${K0_WIDE} parents` }] : []),
      { found: plantedInExhaustive, how: 'Exhaustive search' },
    ];
    const untried = planted.filter((_, n) => !pLab[n]);
    const updD = (t: number, api: FrameApi) => {
      const A = windowed(t, PD[0], PD[1], FADE);
      if (A <= 0.002) return;
      const aXs = smooth(TB.x, TB.x + 1.0, t), sel = smooth(TB.sel, TB.sel + 0.8, t);
      const prune = smooth(TB.prune, TB.prune + 0.9, t), green = smooth(TB.green, TB.green + 0.7, t);
      const thin = NC > 300 ? 0.6 : 1;

      // the coordinates of x; the K0 largest |x_k| are kept
      hair.push([XL, DY0, 0], [XR, DY0, 0], C.white, 0.8 * A);
      for (let k = 0; k < d; k++) {
        if (isParent[k]) continue;
        const v = main.pl.x[k];
        if (large[k]) fat.push([cx(k), DY0, 0], [cx(k), DY0 + v * XS * aXs, 0], large[k] === 2 ? C.slate : v > 0 ? POS : NEG, lerp(1.0, 0.8, sel) * A);
        else hair.push([cx(k), DY0, 0], [cx(k), DY0 + v * XS * aXs, 0], v > 0 ? POS : NEG, lerp(1.0, 0.6, sel) * dens * A);
      }
      const dn = smooth(TB.sel + 0.5, TB.sel + 1.0, t);
      enc.parents.forEach((p) => {
        const v = main.pl.x[p], col = mixRGB(v > 0 ? POS : NEG, SEL, sel);
        fat.push([cx(p), DY0, 0], [cx(p), DY0 + v * XS * aXs, 0], col, 1.0 * A);
        mark(cx(p), DY0 + v * XS * aXs, SEL, 1.2 * sel * A, 2.2);
        if (dn > 0) {
          fat.push([cx(p), Math.min(DY0, DY0 + v * XS) - 0.02, 0], [cx(p), lerp(DY0, DYN, dn), 0], SEL, 0.9 * A);
          mark(cx(p), lerp(DY0, DYN, dn), SEL, 1.2 * A, 2.2);
        }
      });

      // the coordinates linked to the features the encoder missed: large, but not among the ones it kept
      const lostK = smooth(TB.sel + 0.8, TB.sel + 1.4, t) * A;
      const lost = planted.filter((q) => q.slot < 0);
      lost.forEach((q) => {
        const v = main.pl.x[q.parent];
        mark(cx(q.parent), DY0 + v * XS * aXs, C.red, 1.2 * lostK, 2.2);
      });

      // every kept coordinate proposes p ⊕ g_j; one leaf and one bar per (coordinate, offset)
      hair.push([XL, DYB, 0], [XR, DYB, 0], C.white, 0.8 * A);
      hair.push([XL - 0.1, scoreBottom, 0], [XL - 0.1, DYB + scoreHalfHeight, 0], C.white, 0.85 * A);
      for (let v = -scoreTick; v <= scoreTick; v++) {
        if (v === 0) continue;
        hair.push([XL - 0.1, DYB + v * DS, 0], [XL - 0.2, DYB + v * DS, 0], C.white, 0.9 * A);
        hair.push([XL, DYB + v * DS, 0], [XR, DYB + v * DS, 0], C.white, 0.18 * A);
      }
      for (let g = 1; g < order.length; g++) { // group dividers, interrupted where a planted label crosses them
        const xg = XL + g * GW;
        let y = scoreBottom;
        for (const l of pLabUp) {
          if (xg < l.x0 - 0.1 || xg > l.x1 + 0.1 || l.y - LAB_H < y) continue;
          hair.push([xg, y, 0], [xg, l.y - LAB_H, 0], C.white, 0.25 * A);
          hair.push([xg, l.y - LAB_H, 0], [xg, l.y + LAB_H, 0], C.white, 0.25 * (1 - green) * A);
          y = l.y + LAB_H;
        }
        hair.push([xg, y, 0], [xg, DYL + 0.12, 0], C.white, 0.25 * A);
      }
      if (prune > 0.01 && cut > 0) for (const sgn of [-1, 1]) dash(hair, [XL, DYB + sgn * cut * DS, 0], [XR, DYB + sgn * cut * DS, 0], SEL, 0.9 * prune * A, 48);
      const bw = Math.max(0.011, leafW * 0.31);
      for (let n = 0; n < NC; n++) {
        const c = enc.cands[n], sc = enc.distinct[c.slot], g = smooth(fanStart(n), fanStart(n) + 0.6, t);
        if (g <= 0) continue;
        const pl = isPlanted[c.slot] === 1, good = pl && sc.kept, bad = pl && !sc.kept;
        const a: V3 = [cx(c.parent), DYN, 0], b: V3 = [leafX[n], DYL, 0];
        const tip: V3 = [lerp(a[0], b[0], g), lerp(a[1], b[1], g), 0];
        if (sc.kept && prune > 0.01) fat.push(a, tip, good ? mixRGB(SEL, C.white, green) : SEL, 0.9 * prune * A); // a planted feature's link turns to ink
        hair.push(a, tip, C.white, lerp(0.45, sc.kept ? 0.0 : 0.13, prune) * thin * A);
        const s = smooth(scoreStart(n), scoreStart(n) + 0.5, t);
        let col: RGB = s > 0 ? mixRGB(C.slate, c.z > 0 ? POS : NEG, s) : C.slate, k = 1.0;
        if (!sc.kept) { col = mixRGB(col, C.slate, prune); k = lerp(1.0, 0.6, prune); }
        if (bad) col = mixRGB(col, C.red, green);
        if (g > 0.98) mark(leafX[n], DYL, good ? mixRGB(col, C.white, green) : col, (sc.kept ? 1.1 : lerp(1.0, 0.6, prune)) * A, clamp(leafW * 11, 0.5, 1.6));
        if (s > 0) quads.rect(leafX[n] - bw, DYB, leafX[n] + bw, DYB + c.z * DS * s, col, k * A);
        // a found planted feature keeps its sign's color and gets an ink outline, set off from the bar by a gap
        if (good && green > 0.01) { const m = 0.06, y1 = DYB + c.z * DS; box(fat, leafX[n] - bw - m, Math.min(DYB, y1) - m, leafX[n] + bw + m, Math.max(DYB, y1) + m, C.white, green * A); }
      }

      title(api, 't-d', 'Searching for planted features', A, FX0, FYT,
        `<p>The ${d}-coordinate input contains ${nw(main.pl.parents.length)} planted parents and ${nw(DECOYS)} decoys. The encoder keeps the ${nw(K0)} largest coordinates by magnitude as parents and scores ${DELTA} children per parent. It keeps up to ${nw(K1)} distinct directions with the largest absolute scores. The cyan lines mark the cutoff.</p>` +
        `<p>The tallies show the planted features found at each parent count and by scoring all ${int(mDict - d)} directions. Filled boxes count found features, and red outlines count missed features.</p>` +
        (untried.length ? `<p>No parent survived for the planted ${untried.length === 1 ? 'feature' : 'features'} with ${untried.length === 1 ? 'coefficient' : 'coefficients'} ${untried.map((p) => sg(p.coef, 1)).join(' and ')}.</p>` : ''));
      rule(FX0, FX1, A);
      tag(api, 'd-status', FX1, FYT, 'Illustration', A);
      L(api, 'd-l0', [MARG, DY0 + 0.25, 0], 'Input $x$<small>Gray · decoys</small>', A * smooth(TB.x, TB.x + 0.7, t), CSS.white, 'right', 'md');
      if (lostAny) L(api, 'd-lost', [MARG, DY0 - 0.5, 0], 'Red · dropped parent', smooth(TB.sel + 0.8, TB.sel + 1.4, t) * A, CSS.red, 'right', 'sm');
      L(api, 'd-l1', [MARG, (DYN + DYL) / 2, 0], `${DELTA} children / parent`, A * smooth(TB.fan, TB.fan + 0.7, t), CSS.white, 'right', 'md');
      L(api, 'd-z', [MARG - 0.45, DYB + 0.05, 0], 'Raw score', A * smooth(TB.score - 0.3, TB.score + 0.4, t), CSS.white, 'right', 'md');
      [-scoreTick, scoreTick].forEach((v, n) => L(api, 'd-zt' + n, [XL - 0.2, DYB + v * DS, 0], sg(v, 0), A * 0.85, CSS.white, 'right'));
      // the comparison as three tallies of the planted features: found (filled) and missed (red outline), a row of
      // its own under a rule, with its label in the left margin like the rows above
      const aCmp = A * smooth(TB.green, TB.green + 0.6, t), yCmp = scoreBottom - 0.3;
      if (aCmp > 0.002) hair.push([XL, yCmp - 0.25, 0], [XR, yCmp - 0.25, 0], C.white, 0.35 * aCmp);
      L(api, 'd-cmp-h', [MARG, yCmp - 0.25, 0], 'planted features found', aCmp, CSS.white, 'right', 'md', 0, 26);
      TALLY.forEach((q, n) => L(api, 'd-cmp' + n, [XL + (n + 0.5) * (WID / TALLY.length), yCmp, 0],
        `${tallyBoxes(q.found)}&nbsp;${q.found} of ${NP}<small>${q.how}</small>`,
        A * smooth(TB.green + 0.2 * n, TB.green + 0.2 * n + 0.6, t), CSS.white, 'below', 'md', 0, 22));
      L(api, 'd-leg', [XL, scoreBottom, 0], `${span(CSS.amber, '■')}${span(CSS.white, '■')} kept + / −&nbsp;&nbsp;${span(CSS.cyan, '┄ cutoff')}&nbsp;&nbsp;${span(CSS.slate, '■ dropped')}&nbsp;&nbsp;□ planted and kept${planted.some((p, n) => !p.kept && pLab[n]) ? '&nbsp;&nbsp;' + span(CSS.red, '■ planted and dropped') : ''}`, A * prune, CSS.white, 'left', 'sm', -9, 22);
      planted.forEach((p, n) => {
        const l = pLab[n];
        if (l) L(api, 'd-p' + n, [l.x, l.y, 0], p.kept ? `found ${sg(p.z)}, planted ${sg(p.coef, 1)}` : `planted ${sg(p.coef, 1)}, dropped at ${sg(p.z)}`, A * green, p.kept ? CSS.white : CSS.red, l.right ? 'right' : 'left', 'md');
      });
      dupLab.forEach(({ p, l }, n) => L(api, 'd-pd' + n, [l.x, l.y, 0], `the ${sg(planted[p].coef, 1)} feature again`, A * green, planted[p].kept ? CSS.white : CSS.red, l.right ? 'right' : 'left', 'md'));
    };

    // ───────────────────────── 6.5: published evidence ─────────────────────────
    const TABLE_LEFT = -6.8, TABLE_RIGHT = 7.3, TABLE_HEADER_Y = 5.4, TABLE_ROW_Y = 4.48, TABLE_ROW_STEP = 0.82;
    const TABLE_COLS = [0.0, 3.0, 6.0];
    const CPX0 = -8.0, CPX1 = 11.0, CIRCUIT_TOP = 6.05;
    const circuitFigure = await loadFigureImage(CIRCUIT_SOURCE, { width: CPX1 - CPX0 });
    circuitFigure.object.position.set((CPX0 + CPX1) / 2, CIRCUIT_TOP - circuitFigure.height / 2, 0);
    circuitFigure.object.visible = false;
    group.add(circuitFigure.object);
    const rowY = (g: number) => TABLE_ROW_Y - g * TABLE_ROW_STEP;
    const rowStart = (g: number) => PE[0] + 0.4 + 0.15 * g;
    const updE = (t: number, api: FrameApi) => {
      const A = windowed(t, PE[0], T_EDIT, FADE);
      if (A <= 0.002) return;

      title(api, 't-e', 'SAEBench scores', A, TABLE_LEFT, FYT,
        `<p>Mack and co-authors compare layer 12 of their 1.3B models. Compare methods within each row. The arrows point toward better scores. The authors caution that SCR and TPP have low signal-to-noise below two billion parameters.</p>` +
        SAEBENCH.map((m) => `<p>${m.help}</p>`).join(''));
      rule(TABLE_LEFT, TABLE_RIGHT, A);
      L(api, 'e-table-source', [TABLE_LEFT, 6.45, 0], 'Reported by the authors · Mack et al., Table 4', A, CSS.slate);
      L(api, 'e-metric-head', [TABLE_LEFT, TABLE_HEADER_Y, 0], 'metric', A);
      // Blue links the method name to the original figure's parity series. Score text
      // stays neutral in every column so the accent does not imply a win on each metric.
      ['Parity bottleneck', 'TopK SAE', 'Matryoshka SAE'].forEach((name, b) =>
        L(api, 'e-method-' + b, [TABLE_COLS[b], TABLE_HEADER_Y, 0], name, A, b === 0 ? CSS.ice : CSS.white, 'middle'));
      hair.push([TABLE_LEFT, TABLE_HEADER_Y - 0.2, 0], [TABLE_RIGHT, TABLE_HEADER_Y - 0.2, 0], C.white, 0.55 * A);
      SAEBENCH.forEach((m, g) => {
        const aRow = A * smooth(rowStart(g), rowStart(g) + 0.35, t), y = rowY(g);
        L(api, 'e-g' + g, [TABLE_LEFT, y, 0], `${m.name} ${m.up ? '↑' : '↓'}${m.snr ? ' *' : ''}`, aRow, CSS.white, 'left', 'md');
        m.s.forEach((value, b) =>
          L(api, `e-v${g}-${b}`, [TABLE_COLS[b], y, 0], value, aRow, CSS.white, 'middle', 'md'));
        hair.push([TABLE_LEFT, y - 0.25, 0], [TABLE_RIGHT, y - 0.25, 0], C.white, 0.18 * aRow);
      });
      L(api, 'e-caution', [TABLE_LEFT, -0.02, 0], '* Mack et al. flag low signal-to-noise below 2B parameters', A, CSS.slate, 'left');
    };

    const updEdits = (t: number, api: FrameApi) => {
      const A = smooth(T_EDIT, T_EDIT + FADE, t);
      circuitFigure.setOpacity(A);
      if (A <= 0.002) return;
      title(api, 't-e2', 'Circuit-edit success rates', A, CPX0, FYT,
        `<p>Each group compares the five methods at one edit budget. ${CIRCUIT_SUCCESS}</p>` +
        CIRCUITS.map((c) => `<p><strong>${c.name}</strong><br>${c.example}</p>`).join('') +
        `<p>Open the source figure to read all tasks and methods at full size.</p>`, CIRCUIT_SOURCE);
      rule(CPX0, CPX1, A);
      tag(api, 'e-c-label', CPX1, FYT, 'Reported by the authors', A);
    };

    // ───────────────────────── notes helpers ─────────────────────────
    const TITLE_HOT: V3 = [FX0 - 0.45, FYT + 0.3, 0]; // left of a slide's title
    const maxIsTwoRootD = coh.every((c) => Math.abs(c.max - 2 / Math.sqrt(c.d)) < 1e-9);
    const mib = storedBytes / 2 ** 20;

    // ───────────────────────── text ─────────────────────────
    // 6.1: the paper's own words for two of its three meanings of "feature" (Sec. 1): what an autoencoder returns
    // (representational) and what the computation depends on (computational). They are categories, not stages.
    const cap1 =
      'A sparse autoencoder can recover a feature from an activation even if the model ignores it. Mack and co-authors put a bottleneck before every MLP. ' +
      'The decoder combines the selected directions using their signed scores, then restores the input norm. Attention still reads the full residual stream.';
    // 6.2: the hash. The seed matrix's size is in the figure help and the memory arithmetic is in the notes.
    const cap2 =
      'A <em>parity hash</em> generates each hashed direction from its index, avoiding reads of rarely used vectors from GPU memory. ' +
      'For each coordinate, a <em>seed matrix</em>, a small fixed table of bits, picks bits of the index. An even number of 1s gives a plus sign, and an odd number gives a minus sign.';
    // 6.3: why near-orthogonality matters, then the exhaustive check of the inner products at the width of the paper's
    // smaller model (1,024; its larger model has 2,048). The number is on the figure.
    const cap2b =
      "A direction’s inner product with the activation gives its raw score. Overlap between directions lets one feature affect another’s score. " +
      'Mack and co-authors prove an upper bound on the absolute inner product between distinct hashed directions. ' +
      (cohP.max <= cohP.bound
        ? `At their smaller model’s width of ${int(cohP.d)} coordinates, the largest computed value is ${maxP.replace(/0+$/, '')}, below the bound of ${boundP.replace(/0+$/, '')}.`
        : `At their smaller model’s width of ${int(cohP.d)} coordinates, the largest computed value is ${maxP.replace(/0+$/, '')}, above the bound of ${boundP.replace(/0+$/, '')}.`);
    // 6.4: an illustration of the encoder's mechanism (the figure carries the tag); the outcome is said as it came out
    const cap3 =
      'Here the encoder scores the children of the largest-magnitude coordinates. It can miss a strong candidate if none of its parents survive. ' +
      `With eight decoys competing for parent slots, this example recovers ${recovered} of ${NP} planted features with ${K0} parent${K0 === 1 ? '' : 's'}` +
      (compareWide ? `, and ${recWide} with ${K0_WIDE} parents. ` : '. ') +
      `An exhaustive search recovers ${plantedInExhaustive}.`;
    const cap4 =
      'The bottleneck scores higher on single-feature probing and lower on AutoInterp, a test of feature explanations. ' +
      'It shows less absorption, where other features carry first-letter information missed by the main letter features. ' +
      `At similar validation loss, the estimated training time for PT-Large-2L was ${COST} times that of its dense baseline.`;
    const cap5 =
      'Can feature edits shift a prediction? Mack and co-authors edit features from one prompt toward those of another. ' +
      'At each edit budget, the bottleneck had the highest reported success rate on all three tasks. This compares different models and their editing procedures.';
    // ───────────────────────── chapter ─────────────────────────
    const shot = (t: number, x: number, y: number, dist: number) => ({ t, pos: [x, y, dist] as V3, look: [x, y, 0] as V3, fov: 40 });
    const chapter: Chapter = {
      id: 'parity',
      pillar: 'tools',
      kicker: 'INTERPRETABILITY APPLICATIONS',
      title: 'PARITY BOTTLENECK',
      source: 'Mack, Tou, Henry, Wu, Greenspan · Parity Bottleneck Layers · arXiv:2607.20652',
      duration: DUR,
      group,
      results,
      beats: [
        { at: 0.5, hold: PA[1] - FADE - 0.05, caption: cap1 },
        { at: PB[0], caption: cap2 },
        { at: PC0, hold: PC[1] - FADE - 0.05, caption: cap2b },
        { at: PD[0], hold: PD[1] - FADE - 0.05, caption: cap3 },
        { at: PE[0], hold: T_EDIT - FADE - 0.05, caption: cap4 },
        { at: T_EDIT, caption: cap5 },
      ],
      // The benchmark table and original circuit-edit plot each get a full frame.
      shots: [
        shot(0, 1.7, 3.25, 23.0), shot(PA[1] - FADE, 0.9, 3.3, 22.2),
        shot(PB[0] + 0.4, -1.4, 3.8, 20.5), shot(PB[1] - FADE, -1.4, 3.8, 20.2),
        shot(PD[0] + 0.4, 0.2, 3.2, 22.6), shot(PE[0] + 0.3, 2.0, 3.4, 23.0), shot(DUR, 2.0, 3.4, 23.2),
      ],
      update(t, api) {
        hair.clear(); fat.clear(); quads.clear();
        nMarks = 0;
        updA(t, api); updB(t, api); updC(t, api); updD(t, api); updE(t, api); updEdits(t, api);
        hair.commit(); fat.commit(); quads.commit();
        marks.drawCount = nMarks;
        marks.object.visible = nMarks > 0;
        marks.commit();
      },
      readouts(t): Readout[] {
        if (t < PB[0]) {
          const { n } = cycle(t), inp = inputs[n];
          return [
            { label: 'active entries', value: `${K0 + inp.enc.kept.length} of ${int(mDict)}`, detail: true },
            { label: 'input shown', value: `${n + 1} of ${NIN}`, detail: true },
            { label: 'rebuilt vs original, cosine', value: inp.cos.toFixed(3), detail: true },
          ];
        }
        if (t < PD[0]) {
          return [
            { label: `largest inner product, ${d} coordinates`, value: `${max3} under ${bound3}` },
            { label: `the same at ${int(cohP.d)}, the paper's smaller width`, value: `${maxP} under ${boundP}` },
            { label: 'memory of the seed matrix, packed bits', value: `${int(seedBytes)} bytes` },
            { label: `this dictionary's ${int(mDict - d)} directions as 4-byte numbers`, value: fmtMiB(dictBytes) },
            { label: `all ${int(cohD.features)} directions the hash can name, the same way`, value: `${int(mib)} MiB`, detail: true },
            { label: `pairs covered at ${int(cohP.d)} coordinates`, value: `$${expT(cohP.pairs, 3)}$`, detail: true },
          ];
        }
        if (t < PE[0]) {
          return [
            { label: `planted features found, keeping ${K0} coordinates`, value: `${recovered} of ${NP}` },
            ...(compareWide ? [{ label: `the same, keeping ${K0_WIDE} coordinates`, value: `${recWide} of ${NP}` }] : []),
            { label: 'the same, exact search of every direction', value: `${plantedInExhaustive} of ${NP}` },
            { label: `planted features with a retained parent, keeping ${K0}`, value: `${parentKept} of ${NP}` },
            { label: 'scores, found vs planted', value: planted.map((p) => `${p.kept ? sg(p.z) : '×'} vs ${sg(p.coef, 1)}`).join(', '), detail: true },
            { label: 'distinct directions scored', value: `${ND} of ${int(mDict - d)}`, detail: true },
          ];
        }
        if (t >= T_EDIT) return [];
        return [
          { label: 'Bottleneck model', value: 'PT-Large-2L-Aux' },
          { label: 'Sparse autoencoders', value: 'Trained on layer 12 of the 1.3B dense baseline' },
        ];
      },
      set(key: string, value: ParamValue) {
        const v = Math.round(Number(value));
        if (!isFinite(v)) return;
        if (key === 'i') live.i = clamp(v, 0, 65535) & IMASK;
        else return;
        relive();
      },

      details(t): NoteDetails | undefined {
        if (t >= PD[0] && t < PE[0]) {
          return {
            title: 'What the decoys test',
            body:
              `<p>This is a constructed activation, with no model training. Eight decoy coefficients are added to coordinates outside the planted parents so that they compete for the same slots. The wider search has one slot for each planted parent and decoy. At width 256 and 32 children per parent, four parents recovered all three features on 1 of 200 draws, twelve parents on 199, and exhaustive search on 188. Exhaustive search also admits unrelated directions that can outrank a planted one. </p>` +
              `<p>The illustration uses raw scores. The trained encoder standardizes each feature’s score using its running mean and standard deviation, then ranks by standardized magnitude. The decoder uses the signed standardized scores. The recovery rates apply only to these synthetic inputs.</p>`,
          };
        }
        if (t >= PE[0] && t < T_EDIT) {
          return {
            title: 'The training cost of Large-2L',
            body:
              `<p>The 1.3B PT-Large-2L reaches validation loss ${LOSS_PB} after 20 billion tokens. The matched dense model, GPT-Large-cold, reaches ${LOSS_DENSE} after 3 billion, a 6.7-fold difference in token count (Table 2). On four H200 GPUs, Large-2L takes 1.22 times as long per training token (Table 3). Multiplying these factors gives the training-time estimate.</p>` +
              `<p>The benchmark comparison uses PT-Large-2L-Aux, a variant trained with auxiliary losses.</p>`,
          };
        }
        return undefined;
      },

      // Notes add interpretation or evidence that the caption and figure do not supply.
      inspect(t) {
        const hs: Hotspot[] = [];
        if (windowed(t, PA[0], PA[1], FADE) > 0.05) {
          hs.push({
            id: 'bottleneck', pos: TITLE_HOT,
            title: 'When a feature is computational',
            body:
              `<p>The paper calls directions recovered by an autoencoder <em>representational</em> features. A <em>computational</em> feature is one that downstream computation causally depends on, for a particular task and input.</p>` +
              `<p>The bottleneck limits which directions can reach the MLP. An active feature can still make little difference to the answer. Testing its role requires an intervention on a particular input and task.</p>`,
          });
        }
        if (windowed(t, PB[0], PB[1], FADE) > 0.05) {
          hs.push({
            id: 'hash', pos: [FX0 - 0.45, FYT_B + 0.3, 0],
            title: 'How much memory the hash saves',
            body:
              `<p>At the paper's 1.3B model width of 2,048, storing 32,768 level-1 directions as 4-byte numbers would take 256 MiB per layer. The seed matrix needs about 5.5 KiB as packed bits and can fit in GPU registers. These are storage calculations, not measured speedups.</p>`,
          });
        }
        if (windowed(t, PC[0], PC[1], FADE) > 0.05 && t > TBC) {
          hs.push({
            id: 'overlap', pos: [RXT - 0.45, FYT_C + 0.3, 0],
            title: 'Checking every pair',
            body:
              `<p>The inner product depends only on the XOR of the two indices, the bit pattern marking where they differ. A single Walsh–Hadamard transform therefore covers every pair of the $d^2$ hashed directions. The numerical check covers widths ${T_ALL.map((tt) => int(1 << tt)).join(', ')}, including the paper's smaller model width of ${int(cohP.d)}. Theorem A.1 supplies the general bound $(1 + 2\\sqrt d)/d$. The finite check tests its implementation at these widths.</p>` +
              `<p>${maxIsTwoRootD ? 'At each checked width the largest absolute inner product is exactly $2/\\sqrt d$. ' : ''}The observed inner products take values $4/d$ apart.</p>`,
          });
        }
        if (windowed(t, PD[0], PD[1], FADE) > 0.05) {
          hs.push({
            id: 'search', pos: TITLE_HOT,
            title: 'A child can have several parents',
            body:
              `<p>The hierarchy is a directed acyclic graph (DAG). A child can be proposed by several parents and is scored if any one of them is kept. When all its parents are dropped, it is never scored.</p>` +
              `<p>The model trains with the same hierarchy it uses for inference. With a fixed beam and branch size, the number of child scores grows with the number of levels. That can be logarithmic in dictionary size.</p>` +
              (recovered > 0 ? `<p>A recovered score can differ from its planted coefficient because it also measures contributions from the other planted directions, large coordinate components and noise.</p>` : ''),
          });
        }
        if (smooth(T_EDIT, T_EDIT + FADE, t) > 0.05) {
          hs.push({
            id: 'scores', pos: [CPX0 - 0.55, FYT + 0.3, 0],
            title: 'What the circuit comparison measures',
            body:
              `<p>Following Makelov and co-authors (2024), each greedy step picks the single-feature change that brings the reconstructed activation closest to the target prompt’s. A bottleneck edit can zero, replace or rescale a feature. An autoencoder edit sets a feature to its target-prompt value. The authors choose a layer for each task and model, and the autoencoders run on a separate dense model.</p>` +
              `<p>The researchers patch the final token at the selected block’s output and compare two logits. ${CIRCUIT_SUCCESS} They evaluate pairs where the source prediction was initially correct. For greater-than they report 17–23 eligible pairs per run.</p>` +
              `<p>The interpretability model is PT-Large-2L-Aux. Its smallest advantage was on indirect object identification, a task known to rely heavily on attention. Appendix C describes the prompt pairs, layer selection and interventions.</p>`,
          });
        }
        return hs;
      },

      probe(t, p) {
        if (!p.plane) return null;
        const [X, Y] = p.plane, tol = 8 * p.unit;
        const colOf = (x: number) => (x < XL || x > XR ? -1 : Math.min(d - 1, Math.floor((x - XL) / pitch)));

        // 6.1: x, the rebuilt vector and the slots of the dictionary
        if (windowed(t, PA[0], PA[1], FADE) > 0.3) {
          const k = colOf(X);
          if (k < 0) return null;
          const { n, c } = cycle(t);
          const cur = inputs[n], prev = n > 0 ? inputs[n - 1] : null;
          const e1 = smooth(0, 0.4, c), inn = smooth(0.5, 0.9, c), e3 = smooth(1.0, 1.4, c);
          if (Math.abs(Y - YAX) < 0.18 + tol / 2 || Math.abs(Y - YAH) < 0.18 + tol / 2) {
            const top = Math.abs(Y - YAX) < Math.abs(Y - YAH);
            const q = (top ? e1 : e3) > 0.5 || !prev ? cur : prev, nIn = q === cur ? n : n - 1;
            const xk = q.pl.x[k], hk = q.xh[k] * q.norm, isP = q.enc.parents.includes(k);
            return {
              pos: [cx(k), top ? YAX : YAH, 0],
              title: top ? `Coordinate ${k} of $x$` : `Coordinate ${k} of the rebuilt vector`,
              rows: [
                { label: 'input', value: `${nIn + 1} of ${NIN}` },
                { label: 'in $x$', value: sg(xk, 4) },
                { label: 'rebuilt, rescaled', value: sg(hk, 4) },
                { label: 'difference', value: sg(xk - hk, 4) },
                { label: 'kept by the encoder', value: isP ? 'yes' : 'no' },
              ],
            };
          }
          const top = YAG + (GROWS / 2) * RPA;
          if (Y <= top && Y >= YAG - (GROWS / 2) * RPA) {
            const r = Math.min(GROWS - 1, Math.floor((top - Y) / RPA)), i = r * d + k;
            const q = inn > 0.5 || !prev ? cur : prev, nIn = q === cur ? n : n - 1;
            const f = q.act.find((a) => a.row === r && a.col === k);
            return {
              pos: [cx(k), yA(r), 0],
              title: `Dictionary entry ${int(i)}`,
              rows: [
                { label: 'kind', value: r === 0 ? `coordinate ${k} of $x$` : 'hashed direction' },
                { label: 'index in bits', value: bitsStr(i, LB) },
                { label: `active for input ${nIn + 1}`, value: f ? `yes, weight ${sg(f.a, 3)}` : 'no' },
              ],
            };
          }
          return null;
        }

        // 6.2–6.3: a bit of the index or a sign (left panel), or one bar of the inner products (right panel).
        if (windowed(t, PB[0], PB[1], FADE) > 0.3) {
          const onB = X < (LX1 + RX0) / 2;
          const h = hashT(t);
          if (onB && Math.abs(Y - YBIT) < BH / 2 + tol / 2 && X >= LX0 && X <= LX1 && h.idx > 0.5) {
            const c = Math.min(NB - 1, Math.floor((X - LX0) / BWC)), pb = NB - 1 - c;
            return {
              pos: [cellX(c), YBIT, 0],
              title: `Bit ${pb} of index ${int(live.i)}`,
              rows: [
                { label: 'value', value: String((live.i >> pb) & 1) },
                { label: `picked for sign ${kStar}`, value: (M[kStar] >> pb) & 1 ? 'yes' : 'no' },
              ],
            };
          }
          if (onB && Math.abs(Y - YBIG) < BH / 2 + tol / 2 && X >= LX0 && X <= LX1 && h.one > 0.5) {
            const c = Math.min(WIN - 1, Math.floor((X - LX0) / BWC)), k = WIN0 + c;
            const ones = PM.popcount(M[k] & live.i);
            return {
              pos: [cellX(c), YBIG, 0],
              title: `Sign ${k} of the direction`,
              rows: [
                { label: 'bits the seed picks', value: String(PM.popcount(M[k])) },
                { label: 'of them 1 in the index', value: String(ones) },
                { label: 'sign', value: ones & 1 ? 'minus' : 'plus' },
              ],
            };
          }
          if (!onB && t > TBC && Y >= CY0 - 0.2 && Y <= CY0 + CH + 0.15) {
            let best = -1, bd = Math.max(CBW, tol);
            cohD.atoms.forEach((at, a) => {
              if (t < barStart(a) + 0.1) return;
              const dx = Math.abs(X - ovX(at.ip));
              if (dx <= bd) { bd = dx; best = a; }
            });
            if (best >= 0) {
              const at = cohD.atoms[best];
              const pairs = at.count * (cohD.features / 2);
              return {
                pos: [ovX(at.ip), CY0 + (at.mass * CSC) / 2, 0],
                title: `Inner product ${sg(at.ip, 4)}`,
                rows: [
                  { label: 'pairs of directions', value: pairs < 1e10 ? int(pairs) : `$${expT(pairs, 3)}$` },
                  { label: 'share of pairs', value: at.mass < 1e-3 ? `$${expT(at.mass, 2)}$` : `${(at.mass * 100).toFixed(2)}%` },
                  { label: 'against the bound', value: `${(Math.abs(at.ip) / cohD.bound).toFixed(3)} of it` },
                ],
              };
            }
          }
          return null;
        }

        // 6.4: a coordinate of x or one hashed direction tried by the encoder.
        if (windowed(t, PD[0], PD[1], FADE) > 0.3) {
          const aXs = smooth(TB.x, TB.x + 1.0, t), sel = smooth(TB.sel, TB.sel + 0.8, t);
          if (aXs > 0.05 && Math.abs(Y - DY0) < 0.8) {
            const k = colOf(X);
            if (k < 0) return null;
            let kk = k, bd = Infinity; // snap to a kept coordinate if one is within the tolerance
            for (const q of enc.parents) { const dx = Math.abs(cx(q) - X); if (dx < Math.max(tol, pitch) && dx < bd) { bd = dx; kk = q; } }
            const v = main.pl.x[kk];
            let rank = 1;
            for (let m = 0; m < d; m++) if (Math.abs(main.pl.x[m]) > Math.abs(v)) rank++;
            const pp = main.pl.parents.find((q) => q.index === kk);
            return {
              pos: [cx(kk), DY0 + v * XS * aXs, 0],
              title: `Coordinate ${kk} of $x$`,
              rows: [
                { label: 'value', value: sg(v, 4) },
                { label: 'rank by size', value: `${rank} of ${d}` },
                { label: 'kept by the encoder', value: sel > 0.5 ? (isParent[kk] ? 'yes' : 'no') : '…' },
                { label: 'planted', value: pp ? `yes, ${sg(pp.coef)}` : 'no' },
              ],
            };
          }
          if (t >= TB.fan && Y < DYL + 0.2 && Y > scoreBottom - 0.1) {
            let best = -1, bd = Math.max(tol, leafW / 2);
            for (let n = 0; n < NC; n++) {
              if (t < fanStart(n) + 0.3) continue;
              const dx = Math.abs(X - leafX[n]);
              if (dx < bd) { bd = dx; best = n; }
            }
            if (best < 0) return null;
            const c = enc.cands[best], sc = enc.distinct[c.slot];
            const scored = t >= scoreStart(best) + 0.25, pruned = t > TB.prune + 0.5, greenOn = t > TB.green + 0.4;
            const pl = planted.find((q) => q.index === c.child);
            const rows: Readout[] = [
              { label: 'parent coordinate', value: int(c.parent) },
              { label: 'index in bits', value: bitsStr(c.child, LB) },
            ];
            if (sc.from.length > 1) rows.push({ label: 'proposed by', value: `${sc.from.length} parents, ranked once` });
            if (scored) {
              rows.push({ label: 'raw score', value: sg(c.z, 4) });
              rows.push({ label: 'exhaustive raw score', value: sg(all[c.child], 4) });
            }
            if (pruned) rows.push({ label: 'rank by size of score', value: `${sc.rank + 1} of ${ND}: ${sc.kept ? 'kept' : 'dropped'}` });
            if (greenOn) rows.push({ label: 'planted', value: pl ? `yes, ${sg(pl.coef, 1)}` : 'no' });
            const onBar = scored && Y < DYL - 0.3;
            return {
              pos: onBar ? [leafX[best], DYB + (c.z * DS) / 2, 0] : [leafX[best], DYL, 0],
              title: `Hashed direction ${int(c.child)}`,
              rows,
            };
          }
          return null;
        }

        // 6.5: exact values in the published SAEBench table.
        if (windowed(t, PE[0], T_EDIT, FADE) > 0.3) {
          for (let g = 0; g < SAEBENCH.length; g++) {
            if (Math.abs(Y - rowY(g)) > 0.3 || t < rowStart(g) + 0.2) continue;
            const m = SAEBENCH[g];
            for (let b = 0; b < TABLE_COLS.length; b++) {
              if (Math.abs(X - TABLE_COLS[b]) > 0.55) continue;
              return {
                pos: [TABLE_COLS[b], rowY(g), 0],
                title: `paper: ${m.name}, ${SAE_MODEL[b]}`,
                rows: [
                  { label: 'paper: score', value: m.s[b] },
                  ...SAE_MODEL.map((nm, q) => (q === b ? null : { label: `paper: ${nm}`, value: m.s[q] })).filter((r): r is Readout => r !== null),
                  { label: 'better', value: m.up ? 'higher' : 'lower' },
                ],
                note: `Table 4, PT-Large-2L-Aux at layer 12.${m.snr ? ' This result uses a 20-feature ablation. Mack et al. caution that this test has low signal-to-noise below two billion parameters.' : ''}`,
              };
            }
          }
          return null;
        }
        return null;
      },
    };
    return chapter;
  },
};

export default parity;
