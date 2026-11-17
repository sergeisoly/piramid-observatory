import * as THREE from 'three/webgpu';
import type { Chapter, ChapterFactory, Control, Hotspot, Probe, ProbeInput, Readout, Result, FigureSource } from '../engine/types';
import { resolveParams } from '../engine/params';
import { C, CSS, diverge, mixRGB, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { smooth, easeInOut, clamp, lerp, fmt } from '../lib/math';
import { Net, record, scoreTask, leakMap, offDiag, rmsOf, triangleStates, TRI_BONDS, type NetCfg, type TaskScore, type Pass } from './frustration.sim';
import { TileField } from './frustration.tiles';

/**
 * FRUSTRATION: the mechanism of Dmitry Vaintrob's post "A tale of three theories", told on a miniature of
 * its sparse denoiser trained in the browser, and the post's own evidence for it. Four slides:
 *   3.1  the task: one noisy input goes through the trained network (40 features, 32 neurons); beside it, the
 *        network's weights, most of them small and scattered, and bars of how well it denoises (no better than
 *        rescaling its input, on every seed tried). The question is the scatter of the trained weights (quenched,
 *        against the thermal noise of mean-field training)
 *   3.2  why sharing neurons forces interference: a layer narrower than the input cannot give every feature its own
 *        direction, so once every feature must be returned, some feature alone writes into other outputs; the
 *        interference map of the network lists all of it
 *   3.3  the post's example of frustration, reduced to one triangle of spins; flipping a spin moves the unsatisfied pair.
 *        Beside it, the network's spins: the signs of its large encoder weights, neurons × features
 *   3.4  compare the weight plots of the narrower denoiser and a wider denoiser (32 features, 40 neurons)
 *        with the same parameter count. Both architecture and task change. A second comparison within each
 *        network measures the effect of zeroing decoder weights paired with small encoder weights.
 * The four slides are computed in build(); the comparison also links to a separate view of the original source figure.
 */

const D_F = 40, H_N = 32; // features and neurons of the network in superposition (the no-superposition one swaps them)
const STEPS = 3000, LR = 0.02, THR = 0.6, THR_D = 0.3, EARLY = STEPS / 10;
const SCORE_N = 4000; // fresh samples for the task score of each run
const LEAK_LIM = 0.2; // a leak of this size gets the full color of the map

// ── timeline (local seconds) ──
const T1 = 0.8, T2 = 6.0, T3 = 11.5, T4 = 18.0, DUR = 24;
const FLIP1 = T3 + 2.4, FLIP2 = T3 + 4.2, FLIP_LEN = 0.7; // slide 3.3: the top magnet turns over, then the lower left one

// ── layout in world units, with a shared framing box in engine/framing.ts ──
type Rect = { x0: number; x1: number; y0: number; y1: number };
type V3 = [number, number, number];
const ROW = 0.19, Y_TOP = 4.45; // one row per feature, shared by the diagram and the leak map on 3.2 and 3.3
const rowY = (k: number) => Y_TOP - (k + 0.5) * ROW;
const NP = (D_F * ROW) / H_N; // the pitch of the neurons in the diagram
const neuronY = (r: number) => Y_TOP - (r + 0.5) * NP;
const XI0 = -12.0, XIN = -10.9, XH = -8.0, XOUT = -5.5, XO0 = -4.4, BAR = 1.9; // diagram columns, bar length per unit value
const P: Rect = { x0: 1.2, x1: 7.9, y0: -2.5, y1: 4.2 }; // the weight plot (3.1)
const INSET: Rect = { x0: 9.4, x1: 12.7, y0: 0.5, y1: 3.8 }; // its center, enlarged
/** 3.1: how well the network denoises, as three bars under the enlargement: labels from x0, bars from xb, BAR_U per unit. */
const DEN = { x0: 8.55, xb: 11.65, top: -1.95, pitch: 0.46, h: 0.26, unit: 10 };
const MAP_A2 = { x0: 0.2, top: Y_TOP, cell: ROW }; // the leak map on 3.2, rows level with the diagram's
// slide 3.4: two weight plots, each with its center enlarged: no superposition left, superposition right
const PL: Rect = { x0: -13.6, x1: -7.6, y0: -2.3, y1: 3.7 };
const IL: Rect = { x0: -6.6, x1: -3.6, y0: 0.7, y1: 3.7 };
const PR: Rect = { x0: 0.6, x1: 6.6, y0: -2.3, y1: 3.7 };
const IR: Rect = { x0: 7.6, x1: 10.6, y0: 0.7, y1: 3.7 };
const TRI = { cx: -8.4, cy: 1.3, r: 2.2, ring: 0.6 }; // the triangle (3.3)

const POS: RGB = C.amber, NEG: RGB = C.cyan; // the film's signed ramp: ochre +, teal −
const col = (css: string, s: string) => `<span style="color:${css}">${s}</span>`;
const n0 = (v: number) => v.toLocaleString('en-US');
const tickText = (v: number) => (v < -1e-9 ? '−' : '') + String(Math.abs(Math.round(v * 100) / 100));
/** A signed number with a fixed count of decimals and a true minus sign (−0.00 prints as 0.00). */
const signed = (v: number, digits = 2) => {
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
};
const niceLim = (v: number, q: number, lo: number) => Math.max(lo, Math.ceil((v * 1.06) / q - 1e-9) * q);
const per100 = (p: number) => Math.round(p * 100);

const POST_URL = 'https://www.lesswrong.com/posts/siu22scEfuKxpSgfK/a-tale-of-three-theories-sparsity-frustration-and';

/** Author-supplied evidence, kept separate from the miniature trained below. */
const AUTHORS_EXPERIMENT: FigureSource = {
  asset: 'figures/frustration/authors-paired-scatter.png',
  actionLabel: 'Published weight plots',
  title: 'Sparse denoiser weights with and without superposition',
  alt: 'Original paired encoder–decoder weight plots. The network without superposition is on the left, with tightly grouped weights; the network with superposition is on the right, with broader clusters. Both complete axes are preserved, with different axis ranges.',
  credit: 'Experiment by Dmitry Vaintrob and Lucas Teixeira, from “A tale of three theories: sparsity, frustration, and statistical field theory” (2026). The two panels use different ranges on both axes. Architecture and task change together, so these plots alone do not isolate the cause of the scatter. The background plots show the separate miniature computed for this presentation.',
  images: [
    { asset: 'figures/frustration/authors-paired-scatter.png', alt: 'Original published comparison with both complete encoder–decoder scatter plots side by side. Without superposition is on the left and with superposition is on the right. Both axis ranges differ between panels.', caption: 'Without superposition (left). With superposition (right). The two panels use different ranges on both axes.' },
    { asset: 'figures/frustration/authors-without-superposition.png', alt: 'Original encoder–decoder weight scatter without superposition, with tightly grouped weight clusters and the complete source axes.', caption: 'Without superposition' },
    { asset: 'figures/frustration/authors-with-superposition.png', alt: 'Original encoder–decoder weight scatter with superposition, with broader weight clusters and the complete source axes. Both axis ranges differ from the plot without superposition.', caption: 'With superposition' },
  ],
  sourceURL: POST_URL,
};

const controls: Control[] = [
  {
    key: 'seed', label: 'seed', kind: 'range', value: 6, min: 1, max: 20, step: 1,
    hint: 'Sets the random start and the data order of both networks.',
  },
];

interface RunResult {
  cfg: NetCfg;
  E: Float64Array; // h × d, E[i·d + j]: neuron i reads feature j
  D: Float64Array; // d × h, D[k·h + i]: neuron i writes feature k
  score: TaskScore & { shown: Pass };
  leak: Float64Array; // d × d, leak[j·d + k]: output k with feature j alone on
  off: Float64Array; // its off-diagonal entries
  nBig: number; // weights with |E| > THR
  sameSignAtEndpoints: number; // of those, how many end with the sign they started with
  bigEarly: number; // of those, how many were already large, with that sign, at step EARLY
  sdSmall: number; // root mean square of the small E
  maskedRmse: number; // test error after zeroing decoder weights paired with small encoder weights
  sdSmallD: number; // root mean square of D over the pairs with small E
  sanitized: boolean;
}

/** Train one network; keep its final weights, its leaks, its score on fresh samples and where its large weights started. */
async function trainRun(cfg: NetCfg, name: string, log: (m: string) => Promise<void>): Promise<RunResult> {
  const net = new Net(cfg);
  const tr = await record(net, STEPS, LR, [0, EARLY, STEPS], (s) => log(`${name}: step ${s} / ${STEPS}`));
  let sanitized = false;
  for (const a of [net.E, net.D]) for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) { a[i] = 0; sanitized = true; }
  const { d, h } = net, n = d * h;
  const maskedD = net.D.slice();
  let nBig = 0, sameSignAtEndpoints = 0, bigEarly = 0, sE = 0, sD = 0, nS = 0;
  for (let i = 0; i < h; i++)
    for (let j = 0; j < d; j++) {
      const p = i * d + j, e = net.E[p];
      if (Math.abs(e) > THR) {
        nBig++;
        if (Math.sign(tr.E[p]) === Math.sign(e)) sameSignAtEndpoints++; // snapshot 0: the start
        if (Math.abs(tr.E[n + p]) > THR && Math.sign(tr.E[n + p]) === Math.sign(e)) bigEarly++; // snapshot 1: step EARLY
      } else {
        maskedD[j * h + i] = 0;
        nS++;
        sE += e * e;
        sD += net.D[j * h + i] ** 2;
      }
    }
  const leak = leakMap(d, h, net.E, net.D);
  // the score uses its own random stream, so it never touches the training data
  const score = scoreTask(net, SCORE_N, cfg.seed + 9000);
  const masked = scoreTask({ cfg, d, h, E: net.E, D: maskedD }, SCORE_N, cfg.seed + 9000);
  return {
    maskedRmse: masked.rmse, cfg, E: net.E.slice(), D: net.D.slice(), score, leak, off: offDiag(leak, d),
    nBig, sameSignAtEndpoints, bigEarly, sdSmall: Math.sqrt(sE / Math.max(1, nS)), sdSmallD: Math.sqrt(sD / Math.max(1, nS)), sanitized,
  };
}

const frustration: ChapterFactory = {
  id: 'frustration',
  label: 'frustration · sparse denoisers with narrow and wide hidden layers',
  controls,
  async build(ctx, params) {
    const Pm = resolveParams(controls, params);
    const seed = Math.round(Pm.seed);

    // ───────────────────────── the computation ─────────────────────────
    // λ_on = 4 is the film's choice (the post re-balances the loss without giving a number); σ² = 0.04 makes the input
    // noisy enough that the network's output is clearly cleaner than its input
    const base = { d: D_F, h: H_N, sigma2: 0.04, lamOn: 4, kE: 1e-3, kD: 5e-3, sdE: 0.1, sdD: 0.1 };
    // the no-superposition network has the same number of weights: D_F and H_N swapped
    const cfgA: NetCfg = { ...base, seed: 100 + seed }, cfgN: NetCfg = { ...base, d: H_N, h: D_F, seed: 200 + seed };
    const A = await trainRun(cfgA, 'narrow network', ctx.log);
    await ctx.log('narrow network ✓  training the wide network');
    const N = await trainRun(cfgN, 'wide network', ctx.log);
    await ctx.log('runs ✓  building figure');

    const d = D_F, h = H_N, sA = A.score, sN = N.score;
    const leakA = rmsOf(A.off);
    // the post's comparison (3.4): the small write weights scatter only with superposition (4× on every seed tried)
    const contrast = A.sdSmallD > 4 * N.sdSmallD;
    let ownA = 0;
    for (let j = 0; j < d; j++) ownA += A.leak[j * d + j] / d;
    const tri = triangleStates();

    // the feature of slide 3.2: the one with the largest single leak in run 1
    let j0 = 0, j0max = -1;
    for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) if (k !== j && Math.abs(A.leak[j * d + k]) > j0max) { j0max = Math.abs(A.leak[j * d + k]); j0 = j; }
    const shown = sA.shown; // the first fresh scoring input, regardless of outcome
    const aAlone = new Float64Array(h);
    for (let i = 0; i < h; i++) { const t = Math.tanh(A.E[i * d + j0]); aAlone[i] = t * t * t; }

    // neurons are drawn in an order that keeps the diagram's lines short: by the mean row of the features they read
    // strongly (the numbering of neurons is arbitrary, so this changes nothing about the network)
    const bary = Array.from({ length: h }, (_, i) => {
      let s = 0, w = 0;
      for (let j = 0; j < d; j++) { const e = Math.abs(A.E[i * d + j]); if (e > THR) { s += j * e; w += e; } }
      return { i, y: w ? s / w : d / 2 };
    }).sort((a, b) => a.y - b.y);
    const nRow = new Int32Array(h);
    bary.forEach((b, rr) => (nRow[b.i] = rr));

    // the large weights of run 1, as lines of the diagram
    const eLines: { i: number; j: number; s: number }[] = [], dLines: { i: number; k: number; s: number }[] = [];
    let selectedReaders = 0;
    for (let i = 0; i < h; i++)
      for (let j = 0; j < d; j++) {
        const e = A.E[i * d + j];
        if (Math.abs(e) > THR) {
          eLines.push({ i, j, s: Math.sign(e) });
          if (j === j0) selectedReaders++;
        }
      }
    for (let k = 0; k < d; k++) for (let i = 0; i < h; i++) { const w = A.D[k * h + i]; if (Math.abs(w) > THR_D) dLines.push({ i, k, s: Math.sign(w) }); }
    // neurons that read two or more features strongly; of those, how many read them all with one sign (the post's footnote
    // asks for that), and how many would by chance if each sign were random (2^(1−k) for a neuron with k strong reads)
    let carriers = 0, rowAgree = 0, rowChance = 0;
    for (let i = 0; i < h; i++) {
      let c = 0, pos = 0;
      for (let j = 0; j < d; j++) { const e = A.E[i * d + j]; if (Math.abs(e) > THR) { c++; if (e > 0) pos++; } }
      if (c >= 2) { carriers++; rowChance += 2 ** (1 - c); if (pos === 0 || pos === c) rowAgree++; }
    }

    // weight plot limits, from run 1's final weights
    let maxE = 0, maxD = 0;
    for (let i = 0; i < h; i++) for (let j = 0; j < d; j++) { maxE = Math.max(maxE, Math.abs(A.E[i * d + j])); maxD = Math.max(maxD, Math.abs(A.D[j * h + i])); }
    const eLim = Math.min(8, niceLim(maxE, 0.5, 1.5)), dLim = Math.min(8, niceLim(maxD, 0.25, 1.0));
    const stepE = eLim >= 2 ? 1 : 0.5, stepD = dLim >= 1.5 ? 1 : 0.5;
    // the inset shows the small weights: four of their spreads either side of 0, on each axis
    const iE = niceLim(4 * A.sdSmall, 0.05, 0.05), iD = niceLim(4 * A.sdSmallD, 0.05, 0.05);
    // 3.4: both plots share their axes, and both enlargements theirs
    let maxE4 = 0, maxD4 = 0;
    for (const R of [A, N]) {
      const dd = R.cfg.d, hh = R.cfg.h;
      for (let i = 0; i < hh; i++) for (let j = 0; j < dd; j++) { maxE4 = Math.max(maxE4, Math.abs(R.E[i * dd + j])); maxD4 = Math.max(maxD4, Math.abs(R.D[j * hh + i])); }
    }
    const eLim4 = Math.min(8, niceLim(maxE4, 0.5, 1.5)), dLim4 = Math.min(8, niceLim(maxD4, 0.25, 1.0));
    const iE4 = niceLim(4 * Math.max(A.sdSmall, N.sdSmall), 0.05, 0.05), iD4 = niceLim(4 * Math.max(A.sdSmallD, N.sdSmallD), 0.05, 0.05);

    // ───────────────────────── words that depend on the runs ─────────────────────────
    // 3.1: the network in superposition denoises no better than rescaling its input (on 20 of 20 seeds, 0.7–3.5 % worse);
    // the network without superposition beats its own rescaled input (20 of 20, by about a third)
    const noBetter = sA.rmse >= sA.rmsScale;

    // ───────────────────────── scene objects ─────────────────────────
    const group = new THREE.Group();
    const thin = new LineField(2400); // frames, ticks, the diagram's weights
    const strokes = new LineField(120, { width: 3 }); // magnets, bonds, the highlight of a map column
    const inBars = new TileField(d), outBars = new TileField(d);
    const neurons = new PointField({ count: h, size: 0.24 });
    const dots = new PointField({ count: d * h, size: 0.17 }); // the weight plot
    const zoom = new PointField({ count: d * h, size: 0.075 }); // its enlarged center: small, light dots, so that the scatter shows
    const mapA = new TileField(d * d);
    const spins = new TileField(d * h); // 3.3: the signs of the large encoder weights, neuron i in row i, feature j in column j
    const den = new TileField(3); // 3.1: error per feature of the noisy input, the rescaled input and the network
    const den4 = new TileField(4); // 3.4: the rescaled input and the network, for each of the two networks
    // 3.4: both networks, each as a weight plot and its enlarged center
    const dotsN = new PointField({ count: d * h, size: 0.15 }), zoomN = new PointField({ count: d * h, size: 0.065 });
    const dotsS = new PointField({ count: d * h, size: 0.15 }), zoomS = new PointField({ count: d * h, size: 0.065 });
    group.add(mapA.object, spins.object, den.object, den4.object, inBars.object, outBars.object, thin.object, dots.object, zoom.object, dotsN.object, zoomN.object, dotsS.object, zoomS.object, neurons.object, strokes.object);

    // weight plot and inset: static
    const mapX = (rc: Rect, e: number, lim: number) => (rc.x0 + rc.x1) / 2 + (e * (rc.x1 - rc.x0)) / (2 * lim);
    const mapY = (rc: Rect, w: number, lim: number) => (rc.y0 + rc.y1) / 2 + (w * (rc.y1 - rc.y0)) / (2 * lim);
    /** One network's weight pairs (E_ij, D_ji) as a plot in `Pr`, and its small weights enlarged in `Ir`. */
    const paintWeights = (R: RunResult, pf: PointField, zf: PointField, Pr: Rect, Ir: Rect, eL: number, dL: number, iEl: number, iDl: number, inkBig = 1.2) => {
      const dd = R.cfg.d, hh = R.cfg.h;
      for (let i = 0; i < hh; i++)
        for (let j = 0; j < dd; j++) {
          const p = i * dd + j, e = R.E[p], w = R.D[j * hh + i], big = Math.abs(e) > THR;
          pf.setPos(p, mapX(Pr, e, eL), mapY(Pr, w, dL), 0);
          pf.setColor(p, big ? (e > 0 ? POS : NEG) : C.slate, big ? inkBig : 0.45);
          const inside = !big && Math.abs(e) < iEl && Math.abs(w) < iDl;
          zf.setPos(p, mapX(Ir, e, iEl), mapY(Ir, w, iDl), 0);
          zf.setColor(p, C.white, inside ? 0.55 : 0);
          zf.sizes[p] = inside ? 1 : 0;
        }
      pf.commit();
      zf.commit();
    };
    paintWeights(A, dots, zoom, P, INSET, eLim, dLim, iE, iD);
    // lighter ink for the large weights of 3.4, so that a tight cluster keeps its hue when many dots overprint
    paintWeights(N, dotsN, zoomN, PL, IL, eLim4, dLim4, iE4, iD4, 0.55);
    paintWeights(A, dotsS, zoomS, PR, IR, eLim4, dLim4, iE4, iD4, 0.55);
    /** The color of one map cell: the diagonal (each feature's own output) in the team ink, leaks on the signed ramp. */
    const cellColor = (L: Float64Array, j: number, k: number): [RGB, number] =>
      j === k ? [C.ice, 1.15] : [diverge(clamp(L[j * d + k] / LEAK_LIM, -1, 1)), 1];
    /** Lay out one map: column j = feature j on alone, row k = output k. */
    const placeMap = (tf: TileField, L: Float64Array, at: { x0: number; top: number; cell: number }) => {
      for (let j = 0; j < d; j++)
        for (let k = 0; k < d; k++) {
          const [c, a] = cellColor(L, j, k);
          tf.set(j * d + k, at.x0 + (j + 0.5) * at.cell, at.top - (k + 0.5) * at.cell, at.cell * 0.92, at.cell * 0.92, c, a);
        }
      tf.commit();
    };
    placeMap(mapA, A.leak, MAP_A2);
    // the spins, laid out where the interference map stood, columns level with its columns
    let spinPos = 0, spinNeg = 0;
    for (let i = 0; i < h; i++)
      for (let j = 0; j < d; j++) {
        const e = A.E[i * d + j], big = Math.abs(e) > THR;
        if (big) { if (e > 0) spinPos++; else spinNeg++; }
        spins.set(i * d + j, MAP_A2.x0 + (j + 0.5) * MAP_A2.cell, MAP_A2.top - (i + 0.5) * MAP_A2.cell, MAP_A2.cell * 0.9, MAP_A2.cell * 0.9, big ? (e > 0 ? POS : NEG) : C.slate, big ? 1.2 : 0.16);
      }
    spins.commit();
    // the three bars of 3.1: noisy input, rescaled input, the network
    const denVals = [sA.rmsIn, sA.rmsScale, sA.rmse];
    denVals.forEach((v, q) => den.set(q, DEN.xb + (v * DEN.unit) / 2, DEN.top - q * DEN.pitch, v * DEN.unit, DEN.h, q === 2 ? C.white : C.slate, q === 2 ? 1.0 : 0.7));
    den.commit();
    // 3.4: intact network and decoder mask, scored on identical inputs.
    const DEN4_DX = 2.25, DEN4_Y = [-1.55, -2.0];
    ([[N, IL], [A, IR]] as [RunResult, Rect][]).forEach(([R, Ir], q) =>
      [R.score.rmse, R.maskedRmse].forEach((val, r) =>
        den4.set(q * 2 + r, Ir.x0 + DEN4_DX + (val * DEN.unit) / 2, DEN4_Y[r], val * DEN.unit, DEN.h, r === 1 ? C.white : C.slate, r === 1 ? 1.0 : 0.7)));
    den4.commit();

    // the triangle of slide 3.3: vertex 0 on top, 1 lower left, 2 lower right
    const TV: V3[] = [90, 210, 330].map((deg) => [TRI.cx + TRI.r * Math.cos((deg * Math.PI) / 180), TRI.cy + TRI.r * Math.sin((deg * Math.PI) / 180), 0] as V3);
    const bondEnds: [V3, V3][] = TRI_BONDS.map(([a, b]) => {
      const dx = TV[b][0] - TV[a][0], dy = TV[b][1] - TV[a][1], len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len, g = TRI.ring + 0.1;
      return [[TV[a][0] + ux * g, TV[a][1] + uy * g, 0], [TV[b][0] - ux * g, TV[b][1] - uy * g, 0]];
    });
    /** Magnet angles at t: 0 up, 1 down, 2 up; magnet 0 turns over at FLIP1, magnet 1 at FLIP2. */
    const magnets = (t: number) => {
      const f1 = easeInOut(clamp((t - FLIP1) / FLIP_LEN)), f2 = easeInOut(clamp((t - FLIP2) / FLIP_LEN));
      return [Math.PI / 2 - Math.PI * f1, -Math.PI / 2 + Math.PI * f2, Math.PI / 2];
    };

    // ── visibility of each part at t (pure in t; shared by update, inspect and probe) ──
    const vis = (t: number) => {
      const diag = smooth(0.2, 1.0, t) * (1 - smooth(T3, T3 + 0.4, t));
      const alone = smooth(T2 + 0.3, T2 + 1.4, t); // 3.2: the noisy input morphs into one feature alone
      const plot = smooth(0.6, 1.4, t) * (1 - smooth(T2, T2 + 0.4, t));
      const inset = smooth(2.2, 3.0, t) * (1 - smooth(T2, T2 + 0.4, t));
      const map = smooth(T2 + 0.5, T2 + 1.3, t) * (1 - smooth(T3, T3 + 0.4, t));
      const spin = smooth(T3 + 0.3, T3 + 1.1, t) * (1 - smooth(T4, T4 + 0.4, t)); // 3.3: the network's spins
      const hi = smooth(T2 + 1.4, T2 + 2.0, t) * (1 - smooth(T3, T3 + 0.4, t)); // the column of feature j0 in the map
      const tri = smooth(T3 + 0.4, T3 + 1.2, t) * (1 - smooth(T4, T4 + 0.4, t));
      const bond = smooth(T3 + 1.2, T3 + 1.7, t);
      // 3.4: the network in superposition first (it is slide 3.1's), then the one without, then both enlargements
      const sup = smooth(T4 + 0.4, T4 + 1.2, t), nos = smooth(T4 + 1.2, T4 + 2.0, t), ins4 = smooth(T4 + 2.2, T4 + 3.0, t);
      return { diag, alone, plot, inset, map, spin, hi, tri, bond, sup, nos, ins4 };
    };

    const TICK = 27; // px from a tick end to the top of a centered label below it

    const chapter = {
      id: 'frustration',
      pillar: 'theory' as const,
      kicker: 'LEARNING THEORY',
      title: 'FRUSTRATION',
      source: 'D. Vaintrob · A tale of three theories · LessWrong 2026',
      links: [
        { text: 'A tale of three theories', url: POST_URL },
        { text: 'Vaintrob’s denoising experiments', url: 'https://github.com/mvaintrob/clean-denoising-experiments' },
      ],
      duration: DUR,
      group,
      results: [
        {
          label: 'weight scatter in wide and narrow hidden layers',
          measured: `decoder RMS for pairs with small encoder weights, wide vs narrow: ${fmt(N.sdSmallD)} vs ${fmt(A.sdSmallD)}`,
          reference: 'post: a few allowable weight values without superposition, and weights that look essentially random within each kind with it',
          headline: true,
          group: 'check',
          kind: 'miniature adapted from the post',
          claim: 'how widely decoder weights paired with small encoder weights scatter in two networks with equal parameter counts',
          say: contrast
            ? 'Decoder weights paired with small encoder weights cluster closer to zero in the wide network. The published plots show a similar pattern. Both the architecture and the denoising task change.'
            : 'The difference in decoder scatter between the two networks is weaker in this miniature than in the published comparison.',
        },
      ] as Result[],
      beats: [
        {
          at: T1,
          caption:
            'Dmitry Vaintrob and Lucas Teixeira asked why a denoiser’s small weights scattered around zero despite fresh data at every step. This smaller network tries to recover two active features from noisy inputs. ' +
            'On unweighted test RMSE, multiplying every input by the same factor ' +
            (noBetter ? 'does at least as well as the network.' : 'leaves more error than the network.'),
        },
        {
          at: T2,
          caption:
            // the reason is dimension: the map from features to outputs, L = D·tanh³(E), passes through h < d neurons, so it
            // has rank at most h and cannot be diagonal; the signs say how contributions through different neurons combine
            'The network shares neurons across features (<em>superposition</em>). Switch on the highlighted feature alone and a second output responds. This leakage is <em>interference</em>. Contributions through different neurons can cancel, but 32 hidden directions cannot return all 40 features to their own outputs without some leakage.',
        },
        {
          at: T3,
          caption:
            // the post's mapping: the spins are the signs of the large encoder weights. Its random-pattern model is about
            // networks of its size (1,024 neurons for 3,072 features); this miniature's pattern is far from random (notes),
            // so the caption presents Vaintrob's proposed mechanism.
            'Three spins that each favor the opposite orientation from both neighbors cannot all be satisfied. Flipping one can move the conflict to another pair. This is <em>frustration</em>. In the denoiser, the spins are the signs of large encoder weights. Some features share more same-sign paths than others. Vaintrob proposes that the small weights adjust to these unequal overlaps once the signs settle.',
        },
        {
          at: T4,
          caption:
            // the observed contrast, and what the simultaneous architecture/task change leaves open; follows `contrast`
            'With more neurons than features, each feature can have its own direction. ' +
            (contrast ? 'Decoder weights paired with small encoder weights cluster more tightly, as in the published comparison. Both width and task change here. ' : 'The contrast in decoder weights paired with small encoder weights is weaker here than in Vaintrob and Teixeira’s experiment. ') +
            (A.maskedRmse > A.score.rmse ? 'Within the narrow network, zeroing decoder weights paired with small encoder weights raises the denoising error.' : 'Within the narrow network, zeroing decoder weights paired with small encoder weights leaves denoising error unchanged or lower on these inputs.'),
        },
      ],
      // a flat figure seen head-on; the engine re-frames each slide
      shots: [
        { t: 0, pos: [0, 0.81, 26.14], look: [0, 0.81, 0], fov: 38 },
        { t: DUR, pos: [0, 0.81, 26.14], look: [0, 0.81, 0], fov: 38 },
      ],

      update(t: number, api: Parameters<Chapter['update']>[1]) {
        const v = vis(t);
        thin.clear();
        strokes.clear();
        const box = (rc: Rect, k: number, c: RGB = C.white) =>
          thin.polyline([[rc.x0, rc.y0, 0], [rc.x1, rc.y0, 0], [rc.x1, rc.y1, 0], [rc.x0, rc.y1, 0]], c, k, true);
        /** An enlargement's scale: −lim, 0 and lim on both axes, the x labels below, the y labels at the left. */
        const insetTicks = (Ir: Rect, eL: number, dL: number, ins: number, id: string) => {
          for (const q of [-1, 0, 1]) {
            const x = mapX(Ir, q * eL, eL), y = mapY(Ir, q * dL, dL);
            thin.push([x, Ir.y0, 0], [x, Ir.y0 - 0.1, 0], C.white, 0.8 * ins);
            thin.push([Ir.x0, y, 0], [Ir.x0 - 0.1, y, 0], C.white, 0.8 * ins);
            api.label(`${id}x${q}`, [x, Ir.y0 - 0.05, 0], tickText(q * eL), { align: 'center', alpha: ins, dy: TICK - 4, size: 'xs' });
            // the lowest y label would meet the lowest x label at the corner: it stands a little higher
            api.label(`${id}y${q}`, [Ir.x0 - 0.12, y + (q < 0 ? 0.3 : 0), 0], tickText(q * dL), { align: 'right', alpha: ins, dx: 4, size: 'xs' });
          }
        };

        // ════════ 3.1–3.2: the network as a diagram: input bars, neurons, output bars ════════
        {
          const k = v.diag, m = v.alone;
          const show = k > 0.004;
          inBars.uOpacity.value = k;
          outBars.uOpacity.value = k;
          neurons.uOpacity.value = k;
          inBars.object.visible = outBars.object.visible = neurons.object.visible = show;
          if (show) {
            // values: one noisy input on 3.1, morphing into feature j0 alone on 3.2
            for (let j = 0; j < d; j++) {
              const xv = lerp(shown.x[j], j === j0 ? 1 : 0, m);
              const yv = lerp(shown.y[j], A.leak[j0 * d + j], m);
              const xl = clamp(xv, -0.5, 1.3) * BAR, yl = clamp(yv, -0.52, 1.3) * BAR; // room for negative bars up to the node columns
              const on = shown.on.includes(j) ? 1 - m : 0, onAlone = j === j0 ? m : 0;
              // input: positive values grow left of the zero line, negative ones right
              inBars.set(j, XI0 - xl / 2, rowY(j), Math.max(0.004, Math.abs(xl)), ROW * 0.72, mixRGB(C.slate, C.ice, Math.max(on, onAlone)), lerp(0.55, 1.1, Math.max(on, onAlone)));
              // output: positive values grow right of zero, and interference uses the signed ramp
              const own = Math.max(on, onAlone), leakK = j !== j0 ? m : 0;
              const c = mixRGB(mixRGB(C.slate, C.ice, own), yv >= 0 ? POS : NEG, leakK * (1 - own));
              outBars.set(j, XO0 + yl / 2, rowY(j), Math.max(0.004, Math.abs(yl)), ROW * 0.72, c, lerp(0.55, 1.1, Math.max(own, leakK)));
            }
            inBars.commit();
            outBars.commit();
            for (let i = 0; i < h; i++) {
              const a = lerp(shown.a[i], aAlone[i], m), y = neuronY(nRow[i]);
              neurons.setPos(i, XH, y, 0);
              const act = clamp(Math.abs(a) / 0.6);
              neurons.setColor(i, act > 0.02 ? mixRGB(C.slate, a > 0 ? POS : NEG, act) : C.slate, lerp(0.45, 1.3, act));
            }
            neurons.commit();
            // zero lines and node columns
            thin.push([XI0, rowY(0) + ROW / 2, 0], [XI0, rowY(d - 1) - ROW / 2, 0], C.white, 0.5 * k);
            thin.push([XO0, rowY(0) + ROW / 2, 0], [XO0, rowY(d - 1) - ROW / 2, 0], C.white, 0.5 * k);
            // the large weights: input j → neuron i, neuron i → output k; strong where the neuron is active
            const actOf = (i: number) => clamp(Math.abs(lerp(shown.a[i], aAlone[i], m)) / 0.3);
            for (const L of eLines) {
              const a = actOf(L.i), from = j0 === L.j ? m : 0;
              const kk = k * lerp(0.1, 0.95, Math.max(a * (1 - m), from));
              thin.push([XIN, rowY(L.j), 0], [XH - 0.12, neuronY(nRow[L.i]), 0], L.s > 0 ? POS : NEG, kk);
            }
            for (const L of dLines) {
              const a = actOf(L.i), leakPath = L.k !== j0 ? a * m : 0;
              const c = L.s > 0 ? POS : NEG;
              thin.push([XH + 0.12, neuronY(nRow[L.i]), 0], [XOUT, rowY(L.k), 0], c, k * lerp(0.08, 0.95, Math.max(a * (1 - m), leakPath)));
            }
            // labels
            const kk = k;
            api.label('dIn', [XI0 - 1.0, Y_TOP + 0.3, 0], m < 0.5 ? 'noisy input' : 'one feature alone', { align: 'center', alpha: kk, size: 'md', help: m < 0.5 ? `<p>The model has ${d} input features and ${h} hidden neurons. Two features are on, and Gaussian noise is added to every feature. Bars give feature values and lines show large weights.</p>` : '<p>Only the selected feature is on, with no input noise. The input and output bars give feature values, and lines show large weights.</p>' });
            api.label('dH', [XH, Y_TOP + 0.3, 0], 'hidden neurons', { align: 'center', alpha: kk, size: 'md' });
            api.label('dOut', [XO0 + 0.9, Y_TOP + 0.3, 0], 'predicted features', { align: 'center', alpha: kk, size: 'md' });
            api.label('dF1', [-14.25, rowY(0), 0], '1', { align: 'right', alpha: kk, dx: 4 });
            api.label('dFN', [-14.25, rowY(d - 1), 0], String(d), { align: 'right', alpha: kk, dx: 4 });
            api.label('dFA', [-14.25, (rowY(0) + rowY(d - 1)) / 2, 0], 'features', { align: 'right', alpha: kk, dx: 4 });
            api.label('dKey', [XI0 - 2.0, rowY(d - 1) - 0.25, 0],
              `${col(CSS.ice, '■')} active feature&ensp;weight sign: ${col(CSS.amber, '+')} or ${col(CSS.cyan, '−')}`,
              { align: 'left', alpha: kk, dy: 6, color: CSS.slate });
            if (m > 0.01) api.label('dLeak', [XO0 + 0.4, rowY(d - 1) - 0.25, 0], `${col(CSS.amber, '+')} / ${col(CSS.cyan, '−')} interference`, { align: 'left', alpha: kk * m, dy: 6 });
          }
        }

        // ════════ 3.1: the weight plot and its enlarged center ════════
        if (v.plot <= 0.004) { den.uOpacity.value = 0; den.object.visible = false; }
        dots.uOpacity.value = v.plot;
        dots.object.visible = v.plot > 0.004;
        zoom.uOpacity.value = Math.min(v.plot, v.inset);
        zoom.object.visible = v.inset > 0.004;
        if (v.plot > 0.004) {
          const k = v.plot;
          box(P, 0.8 * k);
          thin.push([mapX(P, 0, eLim), P.y0, 0], [mapX(P, 0, eLim), P.y1, 0], C.white, 0.35 * k);
          thin.push([P.x0, mapY(P, 0, dLim), 0], [P.x1, mapY(P, 0, dLim), 0], C.white, 0.35 * k);
          for (let i = -Math.floor(eLim / stepE + 1e-6); i * stepE <= eLim + 1e-6; i++) {
            const e = i * stepE, x = mapX(P, e, eLim);
            if (Math.abs(e) > eLim - 0.05) continue;
            thin.push([x, P.y0, 0], [x, P.y0 - 0.13, 0], C.white, 0.8 * k);
            api.label(`px${i}`, [x, P.y0 - 0.13, 0], tickText(e), { align: 'center', alpha: k, dy: TICK });
          }
          for (let i = -Math.floor(dLim / stepD + 1e-6); i * stepD <= dLim + 1e-6; i++) {
            const w = i * stepD, y = mapY(P, w, dLim);
            if (Math.abs(w) > dLim - 0.05) continue;
            thin.push([P.x0, y, 0], [P.x0 - 0.13, y, 0], C.white, 0.8 * k);
            api.label(`py${i}`, [P.x0 - 0.13, y, 0], tickText(w), { align: 'right', alpha: k, dx: 4 });
          }
          api.label('pT', [P.x0, P.y1 + 0.5, 0], 'Encoder and decoder weights', { align: 'left', alpha: k, size: 'md', dx: -9, help: '<p>For neuron i and feature j, each dot pairs the encoder weight E<sub>ij</sub> with the decoder weight D<sub>ji</sub>. The encoder weight determines the dot’s color.</p>' });
          api.label('pX', [P.x1, P.y0 - 0.95, 0], 'encoder weight →', { align: 'right', alpha: k, dx: 4 });
          api.label('pY', [P.x0 + 0.12, P.y1 - 0.3, 0], '↑ decoder weight', { align: 'left', alpha: k, plate: true, dy: 8 });
          api.label('pKey', [P.x0, P.y0 - 1.85, 0], `${col(CSS.amber, '●')} ${col(CSS.cyan, '●')} large encoder weights, + and −&ensp;${col(CSS.slate, '●')} small encoder weights<br>Computed for this presentation`, { align: 'left', alpha: k, dx: -9 });
          // the small weights' patch on the plot, and the leaders to its enlargement
          const ins = v.inset * k;
          if (ins > 0.004) {
            const zr: Rect = { x0: mapX(P, -iE, eLim), x1: mapX(P, iE, eLim), y0: mapY(P, -iD, dLim), y1: mapY(P, iD, dLim) };
            box(zr, 0.9 * ins);
            box(INSET, 0.8 * ins);
            thin.push([zr.x1, zr.y1, 0], [INSET.x0, INSET.y1, 0], C.white, 0.35 * ins);
            thin.push([zr.x1, zr.y0, 0], [INSET.x0, INSET.y0, 0], C.white, 0.35 * ins);
            thin.push([mapX(INSET, 0, iE), INSET.y0, 0], [mapX(INSET, 0, iE), INSET.y1, 0], C.white, 0.25 * ins);
            thin.push([INSET.x0, mapY(INSET, 0, iD), 0], [INSET.x1, mapY(INSET, 0, iD), 0], C.white, 0.25 * ins);
            insetTicks(INSET, iE, iD, ins, 'i');
            api.label('iT', [INSET.x0, INSET.y1 + 0.35, 0], 'Small-weight pairs', { align: 'left', alpha: ins, dx: -9, help: `<p>The inset shows pairs with small encoder weights in the boxed region around zero. It uses its own axis scales so the scatter is visible.</p>` });
          }
          // how well it denoises: three bars on one scale, the network last
          const dk = v.inset * k;
          den.uOpacity.value = dk;
          den.object.visible = dk > 0.004;
          if (dk > 0.004) {
            api.label('dnT', [DEN.x0, DEN.top + 0.62, 0], 'Test RMSE', { align: 'left', alpha: dk, size: 'md', dx: -9, help: '<p>Bar length gives root mean square error (RMSE) on fresh test inputs, weighting all features equally. The baseline uses the scalar that minimizes expected unweighted squared error among rescalings of the input.</p>' });
            ['noisy input', 'rescaled input', 'this network'].forEach((name, q) => {
              api.label(`dn${q}`, [DEN.x0, DEN.top - q * DEN.pitch, 0], name, { align: 'left', alpha: dk, dx: -9, color: q === 2 ? CSS.white : CSS.slate });
              api.label(`dnValue${q}`, [DEN.xb + denVals[q] * DEN.unit + 0.18, DEN.top - q * DEN.pitch, 0], fmt(denVals[q], 3), { align: 'left', alpha: dk, dx: -4, size: 'xs' });
            });
            thin.push([DEN.xb, DEN.top + DEN.pitch / 2, 0], [DEN.xb, DEN.top - 2.5 * DEN.pitch, 0], C.white, 0.5 * dk);
          }
        }

        // ════════ 3.2–3.3: run 1's leak map ════════
        const at = MAP_A2, mk = v.map;
        mapA.uOpacity.value = mk;
        mapA.object.visible = mk > 0.004;
        if (mk > 0.004) {
          const side = d * at.cell;
          box({ x0: at.x0 - 0.03, x1: at.x0 + side + 0.03, y0: at.top - side - 0.03, y1: at.top + 0.03 }, 0.7 * mk);
          api.label('mT', [at.x0, at.top + 1.35, 0], 'Interference map', { align: 'left', alpha: mk, size: 'md', dx: -9, help: `<p>Each column switches on one feature at value 1 without input noise. Blue marks each feature’s own output. For other outputs, color shows the sign and magnitude of the response, reaching full intensity at ±${LEAK_LIM}. The mean diagonal response is ${fmt(ownA, 2)}. The root mean square off-diagonal response is ${fmt(leakA)}.</p>` });
          // axes of the map: columns = the feature that is on, rows = the output
          api.label('mC', [at.x0 + side / 2, at.top - side - 0.05, 0], 'feature that is on →', { align: 'below', alpha: mk, dy: 2 });
          api.label('mR', [at.x0 - 0.1, at.top - side / 2, 0], 'predicted feature ↓', { align: 'right', alpha: mk, dx: 4 });
          // 3.2: the column of feature j0, which the diagram shows
          if (v.hi > 0.01) {
            const x = at.x0 + (j0 + 0.5) * at.cell;
            strokes.polyline([[x - at.cell / 2 - 0.03, at.top + 0.03, 0], [x + at.cell / 2 + 0.03, at.top + 0.03, 0], [x + at.cell / 2 + 0.03, at.top - side - 0.03, 0], [x - at.cell / 2 - 0.03, at.top - side - 0.03, 0]], C.red, 0.9 * v.hi, true);
            api.label('mJ', [x, at.top + 0.03, 0], `feature ${j0 + 1}`, { align: 'center', alpha: v.hi, dy: -2, color: CSS.red, size: 'xs' });
          }
          if (mk > 0.01) {
            const kx = MAP_A2.x0 + d * MAP_A2.cell + 0.5;
            api.label('mK', [kx, MAP_A2.top - 0.2, 0],
              `${col(CSS.ice, '■')} own output (diagonal)<br>${col(CSS.amber, '■ +')} ${col(CSS.cyan, '■ −')} interference`,
              { align: 'left', alpha: mk });
            api.label('m-origin', [at.x0, at.top - side - 1.15, 0], 'Computed for this presentation', { align: 'left', alpha: mk });
          }
        }

        // ════════ 3.3: the network's spins, the signs of its large encoder weights ════════
        spins.uOpacity.value = v.spin;
        spins.object.visible = v.spin > 0.004;
        if (v.spin > 0.004) {
          const k = v.spin, sw = d * at.cell, sh = h * at.cell;
          box({ x0: at.x0 - 0.03, x1: at.x0 + sw + 0.03, y0: at.top - sh - 0.03, y1: at.top + 0.03 }, 0.7 * k);
          api.label('sT', [at.x0, at.top + 0.76, 0], 'Encoder weight signs', { align: 'left', alpha: k, size: 'md', dx: -9, help: `<p>Each row is a neuron and each column is a feature. Spin means the sign of a large encoder weight. Small encoder weights have no assigned spin.</p><p>At the checkpoint one tenth of the way through training, ${A.bigEarly} of the ${A.nBig} weights that finished large were already large and had their final sign.</p>` });
          api.label('sC', [at.x0 + sw / 2, at.top - sh - 0.05, 0], 'feature →', { align: 'below', alpha: k, dy: 2 });
          api.label('sR', [at.x0 - 0.1, at.top - sh / 2, 0], 'neuron ↓', { align: 'right', alpha: k, dx: 4 });
          api.label('sK', [at.x0 + sw + 0.5, at.top - 0.2, 0],
            `${col(CSS.amber, '■ ↑ +')}&ensp;${col(CSS.cyan, '■ ↓ −')}&ensp;large weight<br><span style="color:${CSS.slate};opacity:0.3">■</span> small weight, no spin`,
            { align: 'left', alpha: k });
          api.label('s-origin', [at.x0, at.top - sh - 1.15, 0], 'Computed for this presentation', { align: 'left', alpha: k });
        }

        // ════════ 3.4: the post's comparison, two weight plots on shared axes, each with its center enlarged ════════
        den4.uOpacity.value = Math.min(v.nos, v.ins4);
        den4.object.visible = den4.uOpacity.value > 0.004;
        const weightPanel = (R: RunResult, Pr: Rect, Ir: Rect, pf: PointField, zf: PointField, k: number, kIns: number, id: string, title: string) => {
          pf.uOpacity.value = k;
          pf.object.visible = k > 0.004;
          zf.uOpacity.value = Math.min(k, kIns);
          zf.object.visible = k * kIns > 0.004;
          if (k < 0.004) return;
          box(Pr, 0.8 * k);
          thin.push([mapX(Pr, 0, eLim4), Pr.y0, 0], [mapX(Pr, 0, eLim4), Pr.y1, 0], C.white, 0.35 * k);
          thin.push([Pr.x0, mapY(Pr, 0, dLim4), 0], [Pr.x1, mapY(Pr, 0, dLim4), 0], C.white, 0.35 * k);
          // numbered at the positive ends only, so that no two labels meet at a corner
          api.label(`${id}x`, [mapX(Pr, eLim4, eLim4), Pr.y0, 0], tickText(eLim4), { align: 'center', alpha: k, dy: TICK - 6 });
          api.label(`${id}y`, [Pr.x0, mapY(Pr, dLim4, dLim4), 0], tickText(dLim4), { align: 'right', alpha: k, dx: 4 });
          api.label(`${id}T`, [Pr.x0, Pr.y1 + 2.0, 0], title, { align: 'left', alpha: k, size: 'md', dx: -9, figure: id === 'S' ? AUTHORS_EXPERIMENT : undefined, help: `<p>This model has ${R.cfg.d} features and ${R.cfg.h} hidden neurons. Each dot pairs the weight from feature j to neuron i with the weight from neuron i back to feature j. Both full plots share the same axes.</p>` });
          api.label(`${id}X`, [Pr.x1, Pr.y0 - 0.85, 0], 'encoder weight →', { align: 'right', alpha: k, dx: 4 });
          api.label(`${id}Y`, [Pr.x0 + 0.1, Pr.y1 - 0.1, 0], '↑ decoder weight', { align: 'left', alpha: k, plate: true, dy: 8 });
          const ins = kIns * k;
          if (ins > 0.004) {
            const zr: Rect = { x0: mapX(Pr, -iE4, eLim4), x1: mapX(Pr, iE4, eLim4), y0: mapY(Pr, -iD4, dLim4), y1: mapY(Pr, iD4, dLim4) };
            box(zr, 0.9 * ins);
            box(Ir, 0.8 * ins);
            thin.push([zr.x1, zr.y1, 0], [Ir.x0, Ir.y1, 0], C.white, 0.3 * ins);
            thin.push([zr.x1, zr.y0, 0], [Ir.x0, Ir.y0, 0], C.white, 0.3 * ins);
            thin.push([mapX(Ir, 0, iE4), Ir.y0, 0], [mapX(Ir, 0, iE4), Ir.y1, 0], C.white, 0.25 * ins);
            thin.push([Ir.x0, mapY(Ir, 0, iD4), 0], [Ir.x1, mapY(Ir, 0, iD4), 0], C.white, 0.25 * ins);
            api.label(`${id}iT`, [Ir.x0, Ir.y1 + 0.35, 0], 'Small-weight pairs', { align: 'left', alpha: ins, dx: -9, help: `<p>The inset shows pairs with small encoder weights in the boxed region near zero. Its scale matches the other network's inset so the decoder scatter can be compared directly.</p>` });
            // the enlargement's scale: its positive ends numbered, as on the first slide
            insetTicks(Ir, iE4, iD4, ins, `${id}i`);
            // Denoising before and after the within-network intervention.
            api.label(`${id}dT`, [Ir.x0, DEN4_Y[0] + 0.45, 0], 'Test RMSE', { align: 'left', alpha: Math.min(ins, v.nos), dx: -9, color: CSS.slate, help: '<p>Each feature gets equal weight in the test RMSE. The trained and masked versions see the same inputs. The mask sets decoder weights to zero wherever the paired encoder weight is small. All other weights stay fixed.</p>' });
            ['trained', 'masked'].forEach((name, r) => {
              const value = r === 0 ? R.score.rmse : R.maskedRmse;
              api.label(`${id}d${r}`, [Ir.x0, DEN4_Y[r], 0], name, { align: 'left', alpha: Math.min(ins, v.nos), dx: -9, color: r === 1 ? CSS.white : CSS.slate });
              api.label(`${id}dValue${r}`, [Ir.x0 + DEN4_DX + value * DEN.unit + 0.18, DEN4_Y[r], 0], fmt(value, 4), { align: 'left', alpha: Math.min(ins, v.nos), dx: -4, size: 'xs' });
            });
            thin.push([Ir.x0 + DEN4_DX, DEN4_Y[0] + 0.25, 0], [Ir.x0 + DEN4_DX, DEN4_Y[1] - 0.25, 0], C.white, 0.5 * Math.min(ins, v.nos));
          }
        };
        weightPanel(A, PR, IR, dotsS, zoomS, v.sup, v.ins4, 'S', 'Narrow hidden layer');
        weightPanel(N, PL, IL, dotsN, zoomN, v.nos, v.ins4, 'N', 'Wide hidden layer');
        if (v.nos > 0.01)
          api.label('cK', [PL.x0, PL.y0 - 1.85, 0],
            `${col(CSS.amber, '●')} ${col(CSS.cyan, '●')} large encoder weights, + and −&ensp;${col(CSS.slate, '●')} small encoder weights<br>Computed for this presentation`,
            { align: 'left', alpha: v.nos, dx: -9 });

        // ════════ 3.3: the post's analogy, three magnets on a triangle ════════
        if (v.tri > 0.004) {
          const k = v.tri, ang = magnets(t), sv = ang.map((a) => Math.sin(a));
          for (let m = 0; m < 3; m++) {
            const ux = Math.cos(ang[m]), uy = sv[m], L = 0.34;
            const cc = mixRGB(NEG, POS, (uy + 1) / 2), hx = TV[m][0] + ux * L, hy = TV[m][1] + uy * L;
            strokes.push([TV[m][0] - ux * L, TV[m][1] - uy * L, 0], [hx, hy, 0], cc, 1.3 * k);
            strokes.push([hx, hy, 0], [hx - ux * 0.18 - uy * 0.13, hy - uy * 0.18 + ux * 0.13, 0], cc, 1.3 * k);
            strokes.push([hx, hy, 0], [hx - ux * 0.18 + uy * 0.13, hy - uy * 0.18 - ux * 0.13, 0], cc, 1.3 * k);
            thin.circle(TV[m], TRI.ring, [1, 0, 0], [0, 1, 0], C.white, 0.8 * k, 48);
          }
          TRI_BONDS.forEach(([a, b], q) => {
            const u = clamp((1 + sv[a] * sv[b]) / 2) * v.bond; // 1 = the two point the same way
            strokes.push(bondEnds[q][0], bondEnds[q][1], mixRGB(C.slate, C.red, u), lerp(0.9, 1.4, u) * k);
          });
          api.label('triT', [TRI.cx - 3.2, TV[0][1] + TRI.ring + 0.8, 0], 'Spins on a triangle<small>Illustration</small>', { align: 'left', alpha: k, size: 'md', help: '<p>Each pair favors opposite spins. A red bond marks an unsatisfied pair pointing the same way. Flipping either spin in this pair moves the conflict to another pair.</p>' });
          // the unsatisfied pair follows the magnets: bond 2–0, then 0–1, then 1–2
          TRI_BONDS.forEach(([a, b], q) => {
            const u = clamp((1 + sv[a] * sv[b]) / 2);
            const w = smooth(0.75, 0.95, u) * k * v.bond;
            if (w < 0.01) return;
            const [p0, p1] = bondEnds[q], mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2;
            const dx = mx - TRI.cx, dy = my - TRI.cy, l = Math.hypot(dx, dy) || 1;
            const lx = mx + (dx / l) * 0.45, ly = my + (dy / l) * 0.45;
            api.label(`triU${q}`, [lx, ly, 0], 'unsatisfied', { align: dy < -0.5 ? 'below' : dx > 0 ? 'left' : 'right', alpha: w, color: CSS.red });
          });
        }

        thin.commit();
        strokes.commit();
      },

      readouts(t: number): Readout[] {
        if (t < T1 + 1.0) return [];
        if (t < T2)
          return [
            { label: 'unweighted test RMSE for noisy input, rescaled input and network', value: `${fmt(sA.rmsIn)}, ${fmt(sA.rmsScale)}, ${fmt(sA.rmse)}` },
            { label: 'two largest values are the two features that are on: network vs input', value: `${per100(sA.top2)}% vs ${per100(sA.top2In)}%`, detail: true },
            { label: 'mean output at the two features that are on', value: fmt(sA.meanOn, 2), detail: true },
            { label: 'neurons that read two or more features strongly', value: `${carriers} of ${h}`, detail: true },
          ];
        if (t < T3)
          return [
            { label: 'RMS interference across all off-diagonal feature pairs', value: fmt(leakA) },
            { label: "a feature's own output, on average", value: fmt(ownA, 2) },
          ];
        if (t < T4)
          return [
            { label: 'spins: large encoder weights, + and −', value: `${spinPos} and ${spinNeg}` },
            { label: 'final large weights with the same sign at initialization and at the end', value: `${A.sameSignAtEndpoints} of ${A.nBig}`, detail: true },
            { label: 'final large weights already large with their final sign one tenth into training', value: `${A.bigEarly} of ${A.nBig}`, detail: true },
          ];
        return [
          { label: 'decoder RMS for pairs with small encoder weights, wide vs narrow', value: `${fmt(N.sdSmallD, 3)} vs ${fmt(A.sdSmallD, 3)}` },
          { label: 'encoder RMS for the same pairs, wide vs narrow', value: `${fmt(N.sdSmall, 3)} vs ${fmt(A.sdSmall, 3)}` },
          { label: 'test RMSE after masking, wide vs narrow', value: `${fmt(N.maskedRmse, 4)} vs ${fmt(A.maskedRmse, 4)}`, note: `intact: ${fmt(sN.rmse, 4)} vs ${fmt(sA.rmse, 4)}` },
        ];
      },

      // ── notes layer ──
      details(t: number) {
        if (t < T2) return {
          title: 'What the training objective rewards',
          body: '<p>Most features are off, so predicting zero can score well under ordinary squared error. Vaintrob gives active features extra weight during training. This miniature weights their squared errors four times as heavily. The authors’ training code also supports separate penalties on encoder and decoder weight size. The miniature chooses its own penalty strengths, which can affect how tightly weights cluster.</p>' +
            (A.sanitized ? '<p>This run produced non-finite weights, which were replaced with zero. Its reported values are unreliable.</p>' : ''),
        };
        if (t >= T4) return {
          title: 'What the weight mask tests',
          body: `<p>In each trained network, the mask sets a decoder weight to zero wherever its paired encoder weight has absolute value at most ${THR}. The weight measurements report the root mean square about zero over all pairs selected by this encoder threshold. Both versions see the same test inputs. The mask measures the combined contribution of those decoder weights to this network’s output.</p><p>Each input has exactly two active features. Swapping the dimensions changes the probability that a feature is active from 2/${D_F} to 2/${H_N}. The two networks have equal parameter counts and train on different data distributions. A wide hidden layer allows separate neurons for each feature, though some runs still share neurons across features.</p>` +
            '<p>Vaintrob and Teixeira published the weight comparison. The decoder mask is a separate test in this smaller model. Testing whether frozen sign conflicts cause the scatter would require changing those conflicts while holding the architecture and task fixed.</p>' +
            (A.sanitized || N.sanitized ? '<p>A run produced non-finite weights, which were replaced with zero. Its reported values are unreliable.</p>' : ''),
        };
        return undefined;
      },

      inspect(t: number): Hotspot[] {
        const v = vis(t), hs: Hotspot[] = [];
        if (v.diag > 0.3 && v.alone < 0.5)
          hs.push({
            id: 'task', pos: [XI0 - 2.2, Y_TOP + 0.3, 0], title: 'Denoising against a scalar baseline',
            body: `<p>The diagram shows the first test input. Across all ${n0(SCORE_N)} test inputs, the narrow network’s error is ${noBetter ? 'at least as high as' : 'lower than'} the error from simply multiplying each input by ${fmt(sA.scale, 2)}. This scalar minimizes expected squared error among scalar rescalings when all features count equally. Training gives the active features extra weight.</p>`,
          });
        if (v.plot > 0.3)
          hs.push({
            id: 'weights', pos: [P.x0 - 0.35, P.y1 + 0.5, 0], title: 'Randomness that stays fixed',
            body: '<p>Vaintrob distinguishes fluctuations that keep changing at positive temperature from quenched disorder that stays fixed over the timescale of interest. He proposes that the signs learned early in training act as a fixed background for later weight adjustments. On that account, disorder can arise inside the network even with unlimited data from a symmetric distribution.</p>',
          });
        if (v.diag > 0.3 && v.alone >= 0.5)
          hs.push({
            id: 'alone', pos: [XI0 - 2.2, Y_TOP + 0.3, 0], title: 'Adding the paths through neurons',
            body:
              `<p>With feature $j$ alone at value 1 and no input noise, output $k$ is $L_{jk} = \\sum_i D_{ki}\\tanh^3(E_{ij})$. The signs of the two weights determine each path's sign. Opposite contributions can cancel in the sum.</p>` +
              `<p>Feature ${j0 + 1} was selected because it has the largest interference on any single output in this run. For this feature, ${selectedReaders} ${selectedReaders === 1 ? 'encoder weight exceeds' : 'encoder weights exceed'} ${THR} in absolute value. The single-feature responses form a matrix of rank at most 32 because the output is linear in 32 hidden activations. A diagonal matrix with 40 nonzero entries would have rank 40.</p>`,
          });
        if (v.tri > 0.3)
          hs.push({
            id: 'triangle', pos: [TRI.cx - 3.55, TV[0][1] + TRI.ring + 0.8, 0], title: 'One triangle and a whole lattice',
            body: `<p>Of the eight possible spin arrangements, ${tri.nGround} have the lowest energy. They all leave ${tri.minUnsat} pair unsatisfied. The disordered ground states discussed in Vaintrob's post belong to an extended triangular lattice.</p>`,
          });
        if (v.spin > 0.3)
          hs.push({
            id: 'spins', pos: [MAP_A2.x0 - 0.35, MAP_A2.top + 0.76, 0], title: 'Limits of the random-sign model',
            body:
              `<p>In the post's simplified model, the locations of nonzero signs are chosen independently, while all nonzero signs within a neuron's row agree. The near-optimality claim applies in a suitable high-dimensional regime. Even the post's network, with 1,024 neurons and 3,072 features, shows structure beyond that approximation.</p>` +
              `<p>Here ${rowAgree} of ${carriers} neurons that read several features strongly use one sign for all of them. Independent random signs would give about ${Math.round(rowChance)}. This smaller network provides a limited illustration of the sign model.</p>`,
          });
        if (v.sup > 0.3)
          hs.push({
            id: 'sup', pos: [PR.x0 - 0.35, PR.y1 + 2.0, 0], title: 'Why frozen disorder could help interpretation',
            body: '<p>Vaintrob argues that disorder can make structures at different scales more independent, which could support analyses that average over fine detail while retaining larger-scale effects. His account links the small weight perturbations to frozen sign structure. The two weight plots alone do not establish that link.</p>',
          });
        return hs;
      },

      // Notes layer: the cell or dot under the pointer (hidden while interactivity is frozen).
      probe(t: number, p: ProbeInput): Probe | null {
        if (!p.plane) return null;
        const [wx, wy] = p.plane, v = vis(t);
        const maps: [{ x0: number; top: number; cell: number }, Float64Array, number, string][] = [[MAP_A2, A.leak, v.map, 'this network']];
        for (const [at, L, k, name] of maps) {
          const side = d * at.cell;
          if (k < 0.3 || wx < at.x0 || wx >= at.x0 + side || wy > at.top || wy <= at.top - side) continue;
          const j = clamp(Math.floor((wx - at.x0) / at.cell), 0, d - 1), kk = clamp(Math.floor((at.top - wy) / at.cell), 0, d - 1);
          return {
            pos: [at.x0 + (j + 0.5) * at.cell, at.top - (kk + 0.5) * at.cell, 0],
            title: j === kk ? `Feature ${j + 1}, its own output, ${name}` : `Feature ${j + 1} into output ${kk + 1}, ${name}`,
            rows: [{ label: j === kk ? 'own output' : 'interference', value: signed(L[j * d + kk], 3) }],
          };
        }
        if (v.spin > 0.3 && wx >= MAP_A2.x0 && wx < MAP_A2.x0 + d * ROW && wy <= MAP_A2.top && wy > MAP_A2.top - h * ROW) {
          const j = Math.floor((wx - MAP_A2.x0) / ROW), i = Math.floor((MAP_A2.top - wy) / ROW);
          const e = A.E[i * d + j];
          return {
            pos: [MAP_A2.x0 + (j + 0.5) * ROW, MAP_A2.top - (i + 0.5) * ROW, 0],
            title: `Neuron ${i + 1} and feature ${j + 1}`,
            rows: [
              { label: 'encoder weight', value: signed(e, 3) },
              { label: 'spin', value: Math.abs(e) > THR ? (e > 0 ? '↑ +' : '↓ −') : 'unassigned (small weight)' },
            ],
          };
        }

        // The same trained weight pair remains inspectable in the full plot and its
        // enlargement, including both networks in the final comparison.
        const weightPlots: [RunResult, Rect, number, number, number, boolean, string][] = [
          [A, P, eLim, dLim, v.plot, false, 'Narrow hidden layer'],
          [A, INSET, iE, iD, Math.min(v.plot, v.inset), true, 'Narrow hidden layer'],
          [N, PL, eLim4, dLim4, v.nos, false, 'Wide hidden layer'],
          [N, IL, iE4, iD4, Math.min(v.nos, v.ins4), true, 'Wide hidden layer'],
          [A, PR, eLim4, dLim4, v.sup, false, 'Narrow hidden layer'],
          [A, IR, iE4, iD4, Math.min(v.sup, v.ins4), true, 'Narrow hidden layer'],
        ];
        for (const [R, rect, eL, dL, visible, inset, name] of weightPlots) {
          if (visible < 0.3 || wx < rect.x0 - 0.3 || wx > rect.x1 + 0.3 || wy < rect.y0 - 0.3 || wy > rect.y1 + 0.3) continue;
          const dd = R.cfg.d, hh = R.cfg.h;
          let best = -1, bd = 100;
          for (let q = 0; q < dd * hh; q++) {
            const i = Math.floor(q / dd), j = q % dd, e = R.E[q], w = R.D[j * hh + i];
            if (inset && (Math.abs(e) > THR || Math.abs(e) >= eL || Math.abs(w) >= dL)) continue;
            const screen = p.project(mapX(rect, e, eL), mapY(rect, w, dL), 0);
            const distance = (screen[0] - p.x) ** 2 + (screen[1] - p.y) ** 2;
            if (distance < bd) { bd = distance; best = q; }
          }
          if (best < 0) continue;
          const i = Math.floor(best / dd), j = best % dd, e = R.E[best], w = R.D[j * hh + i];
          return {
            pos: [mapX(rect, e, eL), mapY(rect, w, dL), 0],
            title: `${name} · neuron ${i + 1} and feature ${j + 1}`,
            rows: [
              { label: 'reads the feature, $E_{ij}$', value: signed(e, 3) },
              { label: 'writes it back, $D_{ji}$', value: signed(w, 3) },
            ],
          };
        }
        return null;
      },
    };
    return chapter as Chapter;
  },
};

export default frustration;
