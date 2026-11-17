import * as THREE from 'three/webgpu';
import type { Chapter, ChapterFactory, FrameApi, Hotspot, LabelOpts, NoteDetails, Probe, Readout, Result } from '../engine/types';
import { resolveParams } from '../engine/params';
import { C, CSS, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { MatrixField } from '../kit/shapes';
import { mulberry32, smooth, easeInOut, clamp, lerp, fmt } from '../lib/math';
import { G, M, TARGETS, makeData, langevin, singleNeuronTheory, plan, type RunData, type TargetId } from './meanfield.sim';

/**
 * MEAN FIELD. Eight slides on one figure:
 *   the neurons of a width-N two-layer network as dots in (w1, w2, a)            (3D scatter, left)
 *   maps on the 8×8 input grid: target, remaining error, a third map             (heatmaps, right)
 * A panel is on screen only on the slides whose text talks about it, so that each slide holds one comparison:
 *   S1  the scatter alone, turning slowly
 *   S2  the target map beside the scatter, as tall as its box
 *   S3  an equation of three maps, error = target − scaled sum of ramps. The ramps form a pile whose top card is the ramp
 *        of one neuron, ringed in the accent color and tied to its card by a leader
 *   S4  training: the scatter sorts into groups whose colors match the signs on the floor and turns to the top;
 *        the ring and the top card follow the same neuron into its group, and the error fades
 *   S5  the error freezes: matched flat input-weight panels separate the network and new neurons; the target and
 *        pile leave, and the two large maps compare the frozen error with the error the new neurons rebuild
 *   S6  the control uses the same flat panels and scales for a network trained under the other scaling and
 *        new neurons trained in its frozen error with the original coupling and readout scales
 *   S7  the error predicted at infinite width beside the frozen one; matched panels compare the network with a
 *        sample of the same size drawn from the solution, without changing the input-weight scales
 *   S8  a second independently initialized and trained network beside the first, with both average residuals
 * Each pair of maps compared on a slide has its own color scale (see `scaleAt`); the bar shows its end value.
 * The size of the output weights is read off the scatter's height axis, and the network error off the Measured
 * panel; neither has a chart, which would repeat what those already show.
 * Five Langevin runs and a numerical solution of the single-neuron theory supply the data (see meanfield.sim.ts).
 * The rebuilt and frozen error maps are compared by their uncentered cosine similarity (`cosine`).
 */

type V3 = [number, number, number];

// ── timeline (seconds): slide starts, then the events inside the slides ──
const S1 = 0.8, S2 = 4.3, S3 = 9.3, S4 = 13.3, S5 = 27.0, S6 = 35.0, S7 = 43.0, S8 = 48.0;
const DUR = 51;
const KF = 0.7; // display keyframe spacing: each dot follows a spline through its true states
const T_LAB = S2 + 0.3; // the target map labels
const T_MAPS = S3 + 0.5; // the error map and the one-neuron map, once the target has made room for them
const T_PICK = S3 + 0.7; // the ringed neuron
const T_RUN = 14.0, D_RUN = 10.6; // training: step 0 … last step
const T_END = T_RUN + D_RUN;
const T_FREEZE = S5; // the error is frozen at its average
const T_CAV = S5 + 1.4, D_CAV = 6.0; // new neurons train independently in the frozen error
// The control network enters already trained. The red dots show new neurons training in its frozen error.
const T_WRONG = S6, T_WR = S6 + 1.6, D_WR = 5.6;
const T_OUT = S7; // the red neurons leave
const T_THEORY = S7 + 0.6; // the infinite-width solution
/**
 * Time spent in prior-only dynamics before training starts, in units of η per step
 * (170 main-run steps and 40 frozen-error steps at η = 0.01).
 */
const PRE_MAIN = 1.7, PRE_TEST = 0.4;

// ── page layout: pixel coordinates on a 1600×900 frame seen by the reference camera ──
// The camera stands closer than a plain head-on view would need, with a wider lens, so that the scatter has some
// perspective; the z = 0 plane, where the maps lie, keeps the same scale (WPP).
const REF_D = 21, REF_FOV = (2 * Math.atan((28 * Math.tan((19 * Math.PI) / 180)) / 21) * 180) / Math.PI;
const WPP = (REF_D * Math.tan((REF_FOV * Math.PI) / 360)) / 450; // world units per pixel on the z = 0 plane
const sp = (sx: number, sy: number): V3 => [(sx - 800) * WPP, -(sy - 450 + 40.5) * WPP, 0];
const SC_PX: [number, number] = [366, 436]; // center of the scatter panel
const SC_INTRO_X = 600; // the scatter stands alone here on the first slide
const POP_SIZE = 244, POP_X = [219, 511] as const; // matched input-weight projections, S5–S8
/**
 * The maps use the width of the frame right of the scatter, in a row centered on the scatter's height, so that the
 * eye moves level between the neurons and the maps.
 * Three arrangements, each filling that width:
 *   S2      the target alone, as tall as the scatter's box (MB)
 *   S3–S4   an equation of three smaller maps (MS): error = target − scaled sum of ramps
 *   S5–S8   two maps as tall as the scatter's box, comparing the frozen error with another error
 */
const HM_Y = 436, MB = 382, MS = 224, XL = 664;
const EQ_GAP = 40, OP_GAP = 96, PAIR_GAP = 56;
const STACK_N = 8, STACK_D = 2.8; // the pile behind the one-neuron map: cards, and their offset in px at size MS
type Slot = [number, number]; // center x, size (layout pixels)
const L2_TGT: Slot = [XL + MB / 2, MB];
const L3_ERR: Slot = [XL + MS / 2, MS], L3_TGT: Slot = [XL + 1.5 * MS + EQ_GAP, MS], L3_THD: Slot = [XL + 2.5 * MS + EQ_GAP + OP_GAP, MS];
const L5_ERR: Slot = [XL + MB / 2, MB], L5_THD: Slot = [XL + 1.5 * MB + PAIR_GAP, MB];
const BAR_N = 32, BAR_COLS = 2, BAR_GAP = 14; // the color bar stands BAR_GAP px right of the last map on screen
const barBeside = (s: Slot, extra = 0): Slot => [s[0] + s[1] / 2 + extra + BAR_GAP + (BAR_COLS * s[1]) / BAR_N / 2, s[1]];
/** The leader from the followed neuron to its ramp runs this far below the maps (px), under their tick labels. */
const LEAD_DY = 44;
const B2: Slot = barBeside(L2_TGT), B3: Slot = barBeside(L3_THD, STACK_N * STACK_D), B5: Slot = barBeside(L5_THD);

// ── scatter scales (world units) ──
const SW = 1.2; // per unit of w
const WAX = 3; // w axes run from −3 to 3
const HBOX = 3.0; // half-height of the axis box
const H_FIT = 382 * WPP; // projected height of the axis box on the page (world units)
const TOP_MUL = 0.95, TOP_UP = 4; // seen from the top the box is drawn a little smaller and higher (pixels)

/** A population of neurons with a recorded run and a screen-time → step map. */
interface Pop {
  run: RunData;
  eta: number;
  t0: number; // screen time of step 0
  dur: number; // screen seconds from step 0 to the last step
  a0: number; // training time covered by the linear part of the map
  p: number; // exponent of the accelerating part
  Ttot: number; // total training time of the run
  phase: Float32Array; // per-dot keyframe phase in [0, 1)
  theta: Float32Array; // n × 3 displayed (w1, w2, a) at the current t
}

function makePop(run: RunData, eta: number, t0: number, dur: number, a0Max: number, half: number, seed: number): Pop {
  const Ttot = run.steps * eta;
  const a0 = Math.min(a0Max, 0.25 * Ttot);
  // choose p so that the map has covered `half` time units at mid-window
  const want = Math.min(half, 0.5 * Ttot);
  const p = clamp(Math.log(Math.max(1e-6, want - a0 / 2) / Math.max(1e-6, Ttot - a0)) / Math.log(0.5), 1.3, 5);
  const rnd = mulberry32(seed);
  const phase = new Float32Array(run.n);
  for (let i = 0; i < run.n; i++) phase[i] = rnd();
  return { run, eta, t0, dur, a0, p, Ttot, phase, theta: new Float32Array(run.n * 3) };
}

/** Display step (fractional) at screen time t: still at step 0 before the run, then a slow start that accelerates. */
function stepAt(pop: Pop, t: number) {
  const tau = t - pop.t0;
  if (tau <= 0) return 0;
  const u = Math.min(1, tau / pop.dur);
  const L = pop.a0 * u + (pop.Ttot - pop.a0) * Math.pow(u, pop.p);
  return clamp(L / pop.eta, 0, pop.run.steps);
}

const P4 = new Float32Array(12);
function stateAt(run: RunData, i: number, step: number, out: Float32Array, o: number) {
  const x = (step + run.pre) / run.every;
  let j = Math.floor(x);
  if (j > run.nSnap - 2) j = run.nSnap - 2;
  if (j < 0) j = 0;
  const f = clamp(x - j);
  const a = (j * run.n + i) * 3, b = a + run.n * 3, s = run.snaps;
  out[o] = s[a] + (s[b] - s[a]) * f;
  out[o + 1] = s[a + 1] + (s[b + 1] - s[a + 1]) * f;
  out[o + 2] = s[a + 2] + (s[b + 2] - s[a + 2]) * f;
}

/** Catmull–Rom through the dot's own states sampled every KF seconds of screen time. */
function pose(pop: Pop, i: number, t: number, out: Float32Array, o: number) {
  const ph = pop.phase[i];
  const kf = t / KF - ph;
  const k0 = Math.floor(kf), u = kf - k0;
  for (let j = 0; j < 4; j++) stateAt(pop.run, i, stepAt(pop, (k0 + j - 1 + ph) * KF), P4, j * 3);
  const u2 = u * u, u3 = u2 * u;
  const c0 = -0.5 * u3 + u2 - 0.5 * u, c1 = 1.5 * u3 - 2.5 * u2 + 1, c2 = -1.5 * u3 + 2 * u2 + 0.5 * u, c3 = 0.5 * u3 - 0.5 * u2;
  out[o] = c0 * P4[0] + c1 * P4[3] + c2 * P4[6] + c3 * P4[9];
  out[o + 1] = c0 * P4[1] + c1 * P4[4] + c2 * P4[7] + c3 * P4[10];
  out[o + 2] = c0 * P4[2] + c1 * P4[5] + c2 * P4[8] + c3 * P4[11];
}

/** Linear interpolation of a per-snapshot 64-vector (the error) at a fractional step. */
function fieldAt(run: RunData, step: number, out: Float32Array) {
  const x = (step + run.pre) / run.every;
  let j = Math.floor(x);
  if (j > run.nSnap - 2) j = run.nSnap - 2;
  if (j < 0) j = 0;
  const f = clamp(x - j), a = j * M, b = a + M, s = run.field;
  for (let k = 0; k < M; k++) out[k] = s[a + k] + (s[b + k] - s[a + k]) * f;
}

/** Diverging tone curve of the maps: MatrixField brightness goes as value², so compress first. */
const tone = (v: number) => 0.63 * Math.sign(v) * Math.pow(Math.min(1, Math.abs(v)), 0.8);

/** Density of the key ink where it marks a negative sign: dark enough to stand apart from blue, teal and slate. */
const NEG_K = 0.8;
const meanfield: ChapterFactory = {
  id: 'meanfield',
  label: 'mean field · training, new neurons, the equation at infinite width',
  controls: [
    { key: 'beta', label: '$\\beta$  inverse temperature', kind: 'range', value: 100, min: 25, max: 400, step: 25, hint: 'Higher $\\beta$ sharpens the groups and lowers the network error. Above 100, $\\eta$ shrinks as $1/\\beta$.' },
    {
      key: 'N', label: '$N$  width', kind: 'select', value: 256,
      options: [32, 64, 128, 256, 512, 1024].map((v) => ({ value: v, label: v.toLocaleString('en-US') })),
      hint: 'Under mean-field scaling the fluctuations of $r(x)$ grow as $N$ falls. The fixed point of that scaling does not depend on $N$.',
    },
    {
      // option labels stay plain text: a <select> cannot typeset
      key: 'target', label: '$y(x)$  target', kind: 'select', value: 'x1x2',
      options: (Object.keys(TARGETS) as TargetId[]).map((k) => ({ value: k, label: TARGETS[k].label })),
      hint: 'Count the neurons for which $\\operatorname{sign}(a) = \\operatorname{sign}(y(w))$.',
    },
    {
      key: 'scaling', label: '$s$  readout scaling', kind: 'select', value: 'mf',
      options: [{ value: 'mf', label: 'mean-field  1/N' }, { value: 'lazy', label: 'Gaussian process  1/√N' }],
      hint: 'GP stands for Gaussian process. This comparison changes both readout scaling and the strength of an individual neuron’s response to the error. With $s = 1/\\sqrt N$ the cloud stays near its prior while $2\\beta/\\sqrt N$ is small.',
    },
    { key: 'seed', label: 'seed', kind: 'range', value: 1, min: 1, max: 20, step: 1, hint: 'Changes the initial weights and training noise. Compare how much the measured patterns vary across runs.' },
  ],

  async build(ctx, params) {
    const P = resolveParams(meanfield.controls, params);
    const N = Math.round(+P.N), beta = +P.beta, seed = Math.round(+P.seed);
    const target = (P.target in TARGETS ? P.target : 'x1x2') as TargetId;
    const lazy = P.scaling === 'lazy';
    const d = makeData(target);
    const T = TARGETS[target];
    const isProd = target === 'x1x2';
    const pl = plan(N, beta, lazy, PRE_MAIN, PRE_TEST);
    const K = pl.K;

    // ── 1. training: N neurons coupled through the error they make together ──
    const mainRun = langevin(d, { seed, n: N, pre: pl.preMain, steps: pl.stepsMain, every: pl.every, eta: pl.eta, b: pl.b, out: pl.out, frozen: null });
    const chunk = Math.max(40, Math.round(250000 / N));
    for (let done = false, c = 0; !done; c++) {
      done = mainRun.advance(chunk);
      await ctx.log(`training ${N} neurons with noise: step ${Math.min(pl.stepsMain, Math.max(0, (c + 1) * chunk - pl.preMain))} / ${pl.stepsMain}`);
    }
    const gasRun = mainRun.data;
    // A new initialization and noise stream test whether the same task gives similar population statistics.
    // Only the last state and the time-averaged residual are needed for this independent repeat.
    const repeatRun = langevin(d, {
      seed: seed + 600, n: N, pre: pl.preMain, steps: pl.stepsMain,
      every: pl.preMain + pl.stepsMain, eta: pl.eta, b: pl.b, out: pl.out, frozen: null,
    });
    while (!repeatRun.advance(chunk)) await ctx.log('Training a second network from a new initialization');
    const repeat = repeatRun.data;
    const repeatWeights = repeat.snaps.subarray(N * 3);

    const rBar = gasRun.rAvg; // the error averaged over the second half of training: the frozen error of the test

    // ── 2. the theory's equation on a 96 × 96 grid of input weights, solved for infinite width ──
    const theory = singleNeuronTheory(d, pl.b, pl.out * N);
    const fp = theory.fixedPoint(d.Y);
    // Under mean-field scaling the equation has no N in it, and its solution is the N → ∞ error. Under lazy
    // scaling b and the readout scale depend on N: the solution is a finite-N quantity. The figure shows it only for mean field.
    const fpOk = fp.ok && !lazy;
    const firstTheorySample = fpOk ? theory.sample(fp.r, N, seed + 7) : null;
    let rmsStar = 0, rmsField = 0;
    for (let k = 0; k < M; k++) {
      rmsStar += (rBar[k] - fp.r[k]) ** 2; rmsField += rBar[k] * rBar[k];
    }
    rmsStar = Math.sqrt(rmsStar / M); rmsField = Math.sqrt(rmsField / M);
    await ctx.log(`equation at infinite width: loss ${fmt(fp.loss, 4)} (Newton, ${fp.iters} it), mean per-step loss ${fmt(gasRun.avg.loss, 4)}`);

    // ── 3. the test: K new neurons trained alone in the network's frozen error ──
    const outK = (pl.out * N) / K;
    const cavRun = langevin(d, { seed: seed + 100, n: K, pre: pl.preTest, steps: pl.stepsTest, every: pl.every, eta: pl.eta, b: pl.b, out: outK, frozen: rBar });
    for (let done = false; !done; ) {
      done = cavRun.advance(1100);
      await ctx.log(`${K} new neurons in the frozen error`);
    }
    // ── 4. the control, the film's reconstruction of the post's "more Gaussian process-like model": the same
    // network trained the same way under the other scaling (at the default, readout 1/√N with the coupling to the error
    // weakened to match, the kernel regime's scaling). Then K new neurons trained alone in its frozen error with the
    // coupling of the test above, so the control asks whether that error fits the loop of the test's scaling. ──
    const pk = plan(N, beta, !lazy, PRE_MAIN, PRE_TEST);
    const kerRun = langevin(d, { seed: seed + 300, n: N, pre: pk.preMain, steps: pk.stepsMain, every: pk.every, eta: pk.eta, b: pk.b, out: pk.out, frozen: null });
    for (let done = false, c = 0; !done; c++) {
      done = kerRun.advance(chunk);
      await ctx.log(`training ${N} neurons with the ${lazy ? 'mean-field' : 'kernel'} scaling: step ${Math.min(pk.stepsMain, Math.max(0, (c + 1) * chunk - pk.preMain))} / ${pk.stepsMain}`);
    }
    const ker = kerRun.data;
    const rK = ker.rAvg; // the control network's error, averaged over the second half of training
    const wrRun = langevin(d, { seed: seed + 400, n: K, pre: pl.preTest, steps: pl.stepsTest, every: pl.every, eta: pl.eta, b: pl.b, out: outK, frozen: rK });
    for (let done = false; !done; ) {
      done = wrRun.advance(1100);
      await ctx.log(`${K} new neurons in the error of the ${lazy ? 'mean-field' : 'GP'}-scaled network`);
    }
    const fresh = cavRun.data, wrong = wrRun.data;
    /** Uncentered cosine similarity of two error maps: 1 for the same pattern, near 0 for unrelated ones. Both are errors around zero. */
    const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) => {
      let ab = 0, aa = 0, bb = 0;
      for (let k = 0; k < M; k++) { ab += a[k] * b[k]; aa += a[k] ** 2; bb += b[k] ** 2; }
      return ab / Math.sqrt(Math.max(1e-30, aa * bb));
    };
    const loss0 = gasRun.loss[gasRun.pre];
    const eGas = gasRun.avg.ea2, eCav = fresh.avg.ea2, eWr = wrong.avg.ea2, eKer = ker.avg.ea2;
    /** Typical size of an output weight (root mean square), the measure the slides, the notes and the board use. */
    const rG = Math.sqrt(eGas), rC = Math.sqrt(eCav), rW = Math.sqrt(eWr), rK0 = Math.sqrt(eKer);

    // ── the sign rule: sign of the output weight = sign of the target in the neuron's input direction ──
    const ruleOf = (w1: number, w2: number, a: number) => { const s = T.fn(w1, w2); return a !== 0 && s !== 0 && Math.sign(a) === Math.sign(s); };
    const ruleCount = (run: RunData) => {
      let n = 0;
      const o = (run.nSnap - 1) * run.n * 3;
      for (let i = 0; i < run.n; i++) if (ruleOf(run.snaps[o + i * 3], run.snaps[o + i * 3 + 1], run.snaps[o + i * 3 + 2])) n++;
      return n;
    };
    const ruleG = ruleCount(gasRun), ruleRepeat = ruleCount(repeat);
    let rule0 = 0; // at step 0 of training, for the notes
    {
      const st = new Float32Array(3);
      for (let i = 0; i < N; i++) { stateAt(gasRun, i, 0, st, 0); if (ruleOf(st[0], st[1], st[2])) rule0++; }
    }
    await ctx.log(`error ${fmt(loss0, 3)}→${fmt(gasRun.avg.loss, 4)} · output weights ${fmt(eGas, 2)} | ${fmt(eCav, 2)} | kernel ${fmt(eKer, 2)} → ${fmt(eWr, 1)} · sign rule ${ruleG} of ${N}`);

    // ── how the run came out: every sentence below that depends on it picks its branch from these ──
    // Computed in advance, on 20 random starts at the default settings: the sign rule held for 92–97 % of the neurons,
    // the test's cosine was 0.79–0.99 and the control's −0.26 to 0.04, so every default branch below was taken on all 20.
    const shareG = ruleG / N;
    /** The match between the rebuilt and frozen errors, measured separately for the test and control. */
    const cosC = cosine(fresh.rAvg, rBar), cosW = cosine(wrong.rAvg, rK);
    const samePattern = cosC >= 0.7, otherPattern = cosW < 0.7;
    /** The post's own comparison: the new neurons read as one network, their network error against the trained network's. */
    const lossC = fresh.avg.loss / Math.max(1e-12, gasRun.avg.loss);
    // ── How well the two networks fit the target (S6), measured by the mean squared values in their frozen error maps.
    // In the November 17, 2026 comparison of 20 random starts at default settings:
    // the kernel-scaled network's frozen error 0.036–0.047 against 0.025–0.026, 1.4–1.8 times larger on all 20; it is
    // close to a faded copy of the target (cosine 0.94–0.97 with it) and to the Gaussian-process posterior mean at the
    // same noise (cosine ≥ 0.99); training four times longer leaves it unchanged (5 starts). The new mean-field neurons
    // overshoot the target in it: the rebuilt error's component along the target is negative on all 20. ──
    const mseOf = (v: ArrayLike<number>) => { let s = 0; for (let k = 0; k < M; k++) s += v[k] * v[k]; return s / M; };
    const relativeError = (rebuilt: Float64Array, frozen: Float64Array) => {
      let squaredDifference = 0;
      for (let k = 0; k < M; k++) squaredDifference += (rebuilt[k] - frozen[k]) ** 2;
      return Math.sqrt(squaredDifference / (M * mseOf(frozen)));
    };
    const mseB = mseOf(rBar), mseK = mseOf(rK);
    const nearLoss = fpOk && Math.abs(mseB / fp.loss - 1) < 0.1;
    // ── color scales: every pair of maps compared on a slide shares one scale. For the target and the live error
    // (S2–S4) it is the round number at or just above the target's largest value. For the small errors compared
    // from S5 on it is the round number above nine in ten of the pair's values: the few larger values take the end
    // color, and the bar's end label then says "≥". The bar is relabeled when the pair changes. ──
    const absSorted = (v: ArrayLike<number>) => Array.from({ length: M }, (_, k) => Math.abs(v[k])).sort((x, y) => x - y);
    const maxAbs = (v: ArrayLike<number>) => absSorted(v)[M - 1];
    const q90 = (v: ArrayLike<number>) => absSorted(v)[Math.floor(0.9 * M)];
    const NICE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5];
    const nice = (v: number) => NICE.find((s) => s >= v * 0.999) ?? Math.ceil(v);
    const pairScale = (a: ArrayLike<number>, b: ArrayLike<number>): [number, boolean] => {
      const s = nice(Math.max(q90(a), q90(b)));
      return [s, Math.max(maxAbs(a), maxAbs(b)) > s * 1.001];
    };
    const scY = nice(maxAbs(d.Y)); // the target, the live error and one neuron's ramp (S2–S4)
    const [scF, clipF] = pairScale(rBar, fresh.rAvg); // the frozen error and its rebuild (S5)
    const [scW, clipW] = pairScale(rK, wrong.rAvg); // the control network's error and its rebuild (S6)
    const [scRepeat, clipRepeat] = pairScale(rBar, repeat.rAvg); // the original and repeated networks (S8)
    const [scT, clipT] = fpOk ? pairScale(rBar, fp.r) : [scF, clipF]; // the frozen error and the theory's (S7)
    await ctx.log(`color scales: target ${scY}, frozen ${scF}, wrong ${scW}, theory ${scT}`);

    // ── axis ranges chosen from the data ──
    const quant = (run: RunData, q: number) => {
      const v: number[] = [];
      const o = (run.nSnap - 1) * run.n * 3;
      for (let i = 0; i < run.n; i++) v.push(Math.abs(run.snaps[o + i * 3 + 2]));
      v.sort((x, y) => x - y);
      return v[Math.min(run.n - 1, Math.floor(q * run.n))];
    };
    const tickStep = (r: number) => (r <= 6 ? 1 : r <= 12 ? 2 : r <= 30 ? 5 : r <= 60 ? 10 : 20);
    const aGas = Math.max(quant(gasRun, 0.995), quant(fresh, 0.995), 2.5);
    const aTop = clamp(Math.ceil(aGas), 3, 60), aStep = tickStep(aTop);
    const aLab = Math.max(aStep, Math.floor((0.85 * aTop) / (2 * aStep)) * 2 * aStep);

    // ── populations ──
    const gas = makePop(gasRun, pl.eta, T_RUN, D_RUN, 1.0, 1.7, 31);
    const cav = makePop(fresh, pl.eta, T_CAV, D_CAV, 0.7, 3.2, 32);
    const wr = makePop(wrong, pl.eta, T_WR, D_WR, 0.7, 3.4, 33);
    // The control network is shown trained. Its step map starts before the chapter.
    const kp = makePop(ker, pk.eta, -1000, 1, 1.0, 1.7, 34);

    const group = new THREE.Group();
    const sc = new THREE.Group(); // the turning 3D scatter
    const SC0 = sp(SC_PX[0], SC_PX[1]), SCI = sp(SC_INTRO_X, SC_PX[1]);
    sc.position.set(SC0[0], SC0[1], 0);
    group.add(sc);
    const WB = WAX * SW;
    // Viewer rotation belongs to the scatter, whose local origin is its center. Flat maps stay put.
    const scatterRotation = new THREE.Quaternion(), scatterTurn = new THREE.Quaternion();
    const scatterRight = new THREE.Vector3(1, 0, 0), scatterUp = new THREE.Vector3(0, 1, 0);
    const parentRotation = new THREE.Quaternion(), authoredNormal = new THREE.Vector3(), turnedNormal = new THREE.Vector3();
    const resetScatter = () => scatterRotation.identity();
    const scatterChanged = () => Math.abs(scatterRotation.w) < 1 - 1e-9;
    const sizeN = Math.pow(256 / N, 0.25);

    // axis box. `floor`: bottom square, light grid and the w ticks (the whole frame in the top-down view).
    // `cage`: the other edges and the zero lines of the a = 0 plane.
    const floor = new LineField(64), cage = new LineField(32);
    floor.polyline([[-WB, -HBOX, WB], [WB, -HBOX, WB], [WB, -HBOX, -WB], [-WB, -HBOX, -WB]], C.white, 0.75, true);
    for (let k = -WAX + 1; k < WAX; k++) {
      floor.push([k * SW, -HBOX, -WB], [k * SW, -HBOX, WB], C.white, k === 0 ? 0.3 : 0.17);
      floor.push([-WB, -HBOX, k * SW], [WB, -HBOX, k * SW], C.white, k === 0 ? 0.3 : 0.17);
    }
    for (let k = -WAX; k <= WAX; k++) {
      floor.push([k * SW, -HBOX, WB], [k * SW, -HBOX, WB + 0.16], C.white, 0.9); // w1 ticks (edge w2 = −3)
      floor.push([WB, -HBOX, -k * SW], [WB + 0.16, -HBOX, -k * SW], C.white, 0.9); // w2 ticks (edge w1 = +3)
    }
    floor.commit();
    cage.polyline([[-WB, HBOX, WB], [WB, HBOX, WB], [WB, HBOX, -WB], [-WB, HBOX, -WB]], C.white, 0.35, true);
    for (const [x, z] of [[-WB, WB], [WB, WB], [WB, -WB], [-WB, -WB]]) cage.push([x, -HBOX, z], [x, HBOX, z], C.white, x < 0 && z < 0 ? 0.75 : 0.35);
    cage.push([-WB, 0, 0], [WB, 0, 0], C.white, 0.22);
    cage.push([0, 0, -WB], [0, 0, WB], C.white, 0.22);
    cage.commit();
    const aTk = new LineField(96); // a ticks on the edge (w1 = −3, w2 = +3); rewritten when the range changes
    sc.add(floor.object, cage.object, aTk.object);

    const gasPts = new PointField({ count: N, size: 0.2 * sizeN, bokeh: 0 });
    // each neuron's shadow on the floor, straight below it: it gives the cloud depth and shows the input direction
    // of a neuron without reading its height. Seen from the top, the shadows lie under the dots.
    const shadow = new PointField({ count: N, size: 0.12 * sizeN, bokeh: 0 });
    /**
     * Height of a marker in the scatter's frame. As the view turns to the top, every marker comes down onto the
     * floor: seen from above, a marker held at its height would stand nearer the lens than the floor's ticks,
     * and the perspective would push it outwards, past the edge of the box.
     */
    const heightOf = (a: number, sa: number, range: number, top: number) => lerp(clamp(a, -range, range) * sa, -HBOX + 0.02, top);
    sc.add(shadow.object, gasPts.object);
    // All paired projections use the same range, including the theory sample and the independent repeat.
    const populationPair = new THREE.Group();
    populationPair.name = 'meanfield-population-comparison';
    let inputExtent = WAX;
    for (const run of [gasRun, fresh, ker, wrong])
      for (let j = 0; j < run.snaps.length; j += 3)
        inputExtent = Math.max(inputExtent, Math.abs(run.snaps[j]), Math.abs(run.snaps[j + 1]));
    for (const sample of [firstTheorySample, repeatWeights]) if (sample)
      for (let j = 0; j < sample.length; j += 3)
        inputExtent = Math.max(inputExtent, Math.abs(sample[j]), Math.abs(sample[j + 1]));
    const inputRange = Math.ceil(inputExtent * 1.05);
    const popAt = (panel: number, w1: number, w2: number) => sp(POP_X[panel] + (w1 / inputRange) * POP_SIZE / 2, HM_Y - (w2 / inputRange) * POP_SIZE / 2);
    const populationFrames = new LineField(32);
    for (let panel = 0; panel < 2; panel++) {
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => popAt(panel, x * inputRange, y * inputRange));
      populationFrames.polyline(corners, C.white, 0.7, true);
      populationFrames.push(popAt(panel, -inputRange, 0), popAt(panel, inputRange, 0), C.white, 0.22);
      populationFrames.push(popAt(panel, 0, -inputRange), popAt(panel, 0, inputRange), C.white, 0.22);
      for (const value of [-inputRange, 0, inputRange]) {
        const x = popAt(panel, value, -inputRange), y = popAt(panel, -inputRange, value);
        populationFrames.push(x, [x[0], x[1] - 4 * WPP, 0], C.white, 0.7);
        populationFrames.push(y, [y[0] - 4 * WPP, y[1], 0], C.white, 0.7);
      }
    }
    populationFrames.commit();
    const baseProjection = new PointField({ count: N, size: 7 * WPP * sizeN, bokeh: 0 });
    const newProjection = new PointField({ count: K, size: 7 * WPP, bokeh: 0 });
    const theoryProjection = new PointField({ count: N, size: 7 * WPP * sizeN, bokeh: 0 });
    baseProjection.object.name = 'trained-population';
    newProjection.object.name = 'independent-population';
    theoryProjection.object.name = 'theory-population';
    populationPair.add(populationFrames.object, baseProjection.object, newProjection.object, theoryProjection.object);
    populationPair.visible = false;
    group.add(populationPair);
    const pairedPopulations = (t: number) => t >= S5;
    const comparisonSample = (t: number) => t >= S8 ? repeatWeights : firstTheorySample;
    const projectPopulation = (points: PointField, weights: Float32Array, panel: number, ink?: RGB) => {
      for (let i = 0; i < points.count; i++) {
        const q = popAt(panel, weights[i * 3], weights[i * 3 + 1]);
        points.setPos(i, q[0], q[1], q[2]);
        points.setColor(i, ink ?? (weights[i * 3 + 2] >= 0 ? C.ice : C.white), 0.9);
      }
      points.commit(true, true, false);
    };
    const ring = new LineField(48, { width: 1.8 }); // marks the neuron whose output the third map shows
    const leader = new LineField(8, { width: 1.2 }); // ties the ringed neuron to its map
    group.add(ring.object, leader.object);

    // ── maps on the input grid: heat[0] the target, heat[1] the network's error, heat[2] the third map (one neuron's
    // ramp, a rebuilt error or the theory's). Each map, its frame and its ticks are drawn at size MB about their
    // center and placed and scaled by their objects, so that a map can change slot and size (see `geo`). ──
    const ROW_Y = sp(0, HM_Y)[1];
    const CELL = (MB * WPP) / G;
    const hh = (MB * WPP) / 2 + 0.03;
    const heat = [0, 1, 2].map(() => {
      const m = new MatrixField({ rows: G, cols: G, cell: CELL, gap: 0.08, height: 0, neg: C.white, pos: C.ice });
      m.object.rotation.x = Math.PI / 2;
      m.object.position.set(0, ROW_Y, 0);
      m.uOpacity.value = 0;
      group.add(m.object);
      return m;
    });
    const setHeat = (m: MatrixField, vals: ArrayLike<number>, gain = 1) => {
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) m.values[(G - 1 - j) * G + i] = tone(vals[j * G + i] * gain);
      m.commit();
    };
    /** Place a map-sized object at slot s (center x and size in layout pixels). */
    const place = (o: THREE.Object3D, s: Slot) => {
      o.position.x = sp(s[0], 0)[0];
      o.scale.setScalar(s[1] / MB);
    };
    // the color bar, its frame and ticks: one group that stands right of the last map on screen
    const barG = new THREE.Group();
    barG.position.y = ROW_Y;
    group.add(barG);
    const bar = new MatrixField({ rows: BAR_N, cols: BAR_COLS, cell: (MB * WPP) / BAR_N, gap: 0, height: 0, neg: C.white, pos: C.ice });
    const barFrame = new LineField(16);
    const bw = (BAR_COLS * MB * WPP) / BAR_N / 2 + 0.02;
    {
      bar.object.rotation.x = Math.PI / 2;
      for (let r = 0; r < BAR_N; r++) for (let q = 0; q < BAR_COLS; q++) bar.values[r * BAR_COLS + q] = tone(1 - (2 * (r + 0.5)) / BAR_N);
      bar.commit();
      bar.uOpacity.value = 0;
      barFrame.polyline([[-bw, -hh, 0], [bw, -hh, 0], [bw, hh, 0], [-bw, hh, 0]], C.white, 0.7, true);
      for (const v of [-1, -0.5, 0, 0.5, 1]) barFrame.push([bw, v * (hh - 0.03), 0], [bw + 0.16, v * (hh - 0.03), 0], C.white, 0.9);
      barFrame.commit();
      barFrame.uIntensity.value = 0;
      barG.add(bar.object, barFrame.object);
    }
    const mapFrames = [0, 1, 2].map(() => {
      const f = new LineField(16);
      f.polyline([[-hh, -hh, 0], [hh, -hh, 0], [hh, hh, 0], [-hh, hh, 0]], C.white, 0.7, true);
      f.commit();
      f.object.position.y = ROW_Y;
      f.uIntensity.value = 0;
      group.add(f.object);
      return f;
    });
    // The pile behind the one-neuron map (S3–S4): the top and right edges of the cards under it, one card per
    // neuron in spirit (the pile shows a few). The network's output is the average of the whole pile.
    const pile = new LineField(2 * STACK_N);
    {
      const o = (STACK_D * MB) / MS * WPP;
      for (let j = 1; j <= STACK_N; j++) {
        const k = 0.62 * (1 - (j - 1) / STACK_N);
        pile.polyline([[-hh + j * o, hh + j * o, 0], [hh + j * o, hh + j * o, 0], [hh + j * o, -hh + j * o, 0]], C.white, k);
      }
      pile.commit();
      pile.object.position.y = ROW_Y;
      pile.uIntensity.value = 0;
      group.add(pile.object);
    }
    // the frame of the top card in the accent color of the followed neuron
    const topCard = new LineField(8, { width: 1.6 });
    topCard.polyline([[-hh, -hh, 0], [hh, -hh, 0], [hh, hh, 0], [-hh, hh, 0]], C.amber, 1.0, true);
    topCard.commit();
    topCard.object.position.y = ROW_Y;
    topCard.uIntensity.value = 0;
    group.add(topCard.object);
    const keyFrame = new LineField(24); // the ticks of the leftmost map
    {
      for (const v of [-1, 0, 1]) {
        keyFrame.push([v * hh, -hh, 0], [v * hh, -hh - 0.16, 0], C.white, 0.9); // x1 ticks
        keyFrame.push([-hh, v * hh, 0], [-hh - 0.16, v * hh, 0], C.white, 0.9); // x2 ticks
      }
    }
    keyFrame.commit();
    keyFrame.object.position.y = ROW_Y;
    keyFrame.uIntensity.value = 0;
    group.add(keyFrame.object);

    // The one neuron whose output the third map shows from S3. It is chosen for a clear ramp when it
    // appears (|a| near 1, |w| near 1.5), and so that it would stay easy to follow through training: at four
    // later moments an output weight that is not small, has the sign it ends with and input weights inside
    // the axis box; and at the end a group member with a moderate output weight.
    let pick = 0;
    {
      const st = new Float32Array(3), fin = new Float32Array(3);
      const checks = [0.15, 0.35, 0.7, 1].map((f) => stepAt(gas, T_RUN + f * D_RUN));
      let bs = -Infinity;
      for (let i = 0; i < N; i++) {
        stateAt(gasRun, i, gasRun.steps, fin, 0);
        stateAt(gasRun, i, 0, st, 0);
        let score =
          (Math.abs(st[2]) > 0.5 ? 1 : 0) - Math.abs(Math.abs(st[2]) - 1.0) - Math.abs(Math.hypot(st[0], st[1]) - 1.5) * 0.7 +
          (fin[2] * T.fn(fin[0], fin[1]) > 0 ? 1 : 0) - Math.abs(Math.abs(fin[2]) - 1.6) * 0.5;
        for (const sk of checks) {
          stateAt(gasRun, i, sk, st, 0);
          score += (Math.abs(st[2]) > 0.6 ? 0.5 : 0) + (st[2] * fin[2] > 0 ? 0.5 : 0) + (Math.max(Math.abs(st[0]), Math.abs(st[1])) < 2.4 ? 0.5 : 0);
        }
        if (score > bs) { bs = score; pick = i; }
      }
    }

    // ── per-frame scratch ──
    const tmp = new Float32Array(3);
    const rIn = new Float32Array(M), rOut = new Float32Array(M), one = new Float32Array(M);
    const col: RGB = [0, 0, 0];
    const signCol = (v: number, scale: number, out: RGB) => {
      const sg = 0.5 + 0.5 * Math.tanh(v / scale);
      out[0] = NEG_K * C.white[0] + (C.ice[0] - NEG_K * C.white[0]) * sg;
      out[1] = NEG_K * C.white[1] + (C.ice[1] - NEG_K * C.white[1]) * sg;
      out[2] = NEG_K * C.white[2] + (C.ice[2] - NEG_K * C.white[2]) * sg;
    };
    const vec = new THREE.Vector3();
    /** Scatter-local coordinates (world units along w1, a, −w2) → world. */
    const scP = (x: number, y: number, z: number): V3 => {
      vec.set(x, y, z).applyMatrix4(sc.matrixWorld);
      return [vec.x, vec.y, vec.z];
    };

    /** Pose a population at time t and write its markers (a and w clamped to the axis box). */
    const drawPop = (pop: Pop, pts: PointField | null, t: number, sa: number, range: number, top: number, color: (a: number, out: RGB) => void, k: (i: number) => number) => {
      const n = pop.run.n, th = pop.theta;
      for (let i = 0; i < n; i++) {
        pose(pop, i, t, tmp, 0);
        th[i * 3] = tmp[0]; th[i * 3 + 1] = tmp[1]; th[i * 3 + 2] = tmp[2];
        pts?.setPos(i, clamp(tmp[0], -WAX, WAX) * SW, heightOf(tmp[2], sa, range, top), -clamp(tmp[1], -WAX, WAX) * SW);
        color(tmp[2], col);
        pts?.setColor(i, col, k(i));
      }
      pts?.commit(true, true, false);
    };
    // ── text ──
    const f2 = (x: number) => fmt(x, 2), f3 = (x: number) => fmt(x, 3), f4 = (x: number) => fmt(x, 4);
    const n0 = (v: number) => v.toLocaleString('en-US');
    const sTex = lazy ? '\\tfrac{1}{\\sqrt N}' : '\\tfrac1N'; // readout scale of the width-N network
    const sNTex = lazy ? `\\tfrac{1}{\\sqrt{${N}}}` : `\\tfrac{1}{${N}}`; // the same scale with the numeric width substituted for N
    const bTex = lazy ? '2\\beta/\\sqrt N' : '2\\beta';
    const targetW = isProd ? 'the product' : 'the target';

    // The sign statistic uses the target evaluated along the neuron's input-weight direction.
    const capSign = shareG >= 0.6
      ? `By the end, ${shareG >= 0.9 ? 'nearly all' : 'most'} output weights have the sign of ${targetW} along their neuron’s input-weight direction.`
      : 'Here the sign of the target along a neuron’s input-weight direction is a weak predictor of its output weight.';
    const capTest = 'Vaintrob and Greenspan trained fresh neurons independently on the error of the full network. They reported a similar cloverleaf of input weights and a similar loss. ' +
      `This reconstruction compares the errors input by input. The new neurons leave ${samePattern ? 'a similar' : 'a different'} pattern.`;
    const capFail = 'Does the choice of frozen error matter? In the published experiment, a background closer to a Gaussian process gave a different neuron population. ' +
      `Here the comparison network uses ${lazy ? 'mean-field' : 'Gaussian-process'} scaling. New neurons use the ${lazy ? 'Gaussian-process' : 'mean-field'} response and readout scales. They rebuild ${otherPattern ? 'a different' : 'a similar'} error pattern.`;
    const theoryLead = 'At infinite width, each candidate error determines an equilibrium distribution of neurons. The self-consistency equation selects the error left by that distribution’s average prediction.';
    const capTheory = !fpOk
      ? lazy ? 'The selected readout and coupling depend on width. Switch to mean-field scaling to compare the finite network with its infinite-width prediction.' : `${theoryLead} The numerical solver did not converge at these settings.`
      : `${theoryLead} Solving it on this input grid gives ${rmsStar < 0.2 * rmsField ? 'an error close to the trained network’s.' : 'a visibly different error from the trained network’s.'}`;
    /** The self-consistency equation (S7): the error equals the target minus the mean output of neurons drawn from ρ_r. */
    const EQ_LOOP = String.raw`r(x) = y(x) - \mathbb{E}_{(w,a)\sim\rho_r}\big[a\,\varphi(w\cdot x)\big]`;
    const beats = [
      { at: S1, caption: 'In the neural tangent kernel (NTK) regime, individual weights stay near their starting values during training. Dmitry Vaintrob and Lauren Greenspan study a regime where neurons can learn new features. Can a theory predict which weights become common across the population?' },
      { at: S2, caption: `Each hidden neuron responds to a direction in the plane of two inputs. Two input weights set that direction, and an output weight sets how much the neuron contributes. ${lazy ? 'The network divides their sum by the square root of its width.' : 'The network averages these contributions.'} Its target is ${isProd ? 'the product of the inputs.' : `$y = ${T.tex}$.`}` },
      { at: S3, caption: 'Training changes each neuron according to what the whole network still gets wrong. In a wide network, changing just one neuron barely affects that error. Each neuron can therefore treat the error as its background, even as all the neurons together reshape it.' },
      { at: S4, caption: `A neuron’s output weight tends to increase when it responds to inputs where the network predicts too little, and to decrease where the network predicts too much. ${capSign}` },
      { at: S5, caption: capTest },
      { at: S6, caption: capFail },
      // the equation in the formula corner, only where the caption writes the cycle as one equation (and it was solved)
      { at: S7, caption: capTheory, ...(fpOk ? { eq: EQ_LOOP, eqNote: '$\\rho_r$: how a neuron settles alone in the error $r$; the equation sets $r$ to the target minus the average output of such neurons' } : {}) },
      { at: S8, eq: '', caption: 'Does the population pattern survive a fresh start? A second network learned the same task from new random weights and a different sequence of training noise. ' +
        (Math.min(ruleG, ruleRepeat) / N >= 0.8 ? 'In both runs, most neurons follow the same relation between input direction and output sign.' : 'The fraction following the sign rule is ' + Math.round(100 * ruleG / N) + '% in the first run and ' + Math.round(100 * ruleRepeat / N) + '% in the second.') },

    ];

    // The closing board: the rebuilt test and its control are a check of the miniature, with the match of the rebuilt
    // error as the number. The weight sizes stay on the notes page: the control's larger weights follow from its coupling.
    const testSay = `Independent neurons trained against the network’s frozen error rebuild ${samePattern ? 'a similar' : 'a different'} error pattern. Given the comparison network’s error they rebuild ${otherPattern ? 'a different' : 'a similar'} pattern.`;
    const results: Result[] = [
      {
        headline: true, group: 'check', say: testSay, kind: 'rebuilt test with a control',
        label: `match of the error new neurons rebuild with the error they were given (cosine similarity): the network's own vs the ${lazy ? 'other' : 'GP-scaled'} network's`,
        measured: `${f2(cosC)} vs ${f2(cosW)}`.replace(/-/g, "−"),
        reference: '1 for maps with identical shape and positive relative scale',
        claim: 'cosine similarity of two error maps on the 64 inputs, 1 for the same pattern',
      },
      {
        group: 'check', kind: 'agreement with theory',
        say: !fpOk ? 'The self-consistency equation was not solved at these settings.' : nearLoss ? `The self-consistency equation, solved at infinite width, gives nearly the mean squared error of the ${N}-neuron network’s averaged error map.` : `The self-consistency equation, solved at infinite width, gives a mean squared error more than a tenth away from that of the ${N}-neuron network’s averaged error map.`,
        label: 'mean squared error of the averaged map and the infinite-width prediction',
        measured: `${f4(mseB)} vs ${fpOk ? f4(fp.loss) : 'not solved'}`,
        reference: 'stationary infinite-width solution on the same input grid',
        claim: 'mean squared error on the 64 inputs, using the time-averaged finite-network error and the stationary prediction',
      },
    ];

    // ── camera: frontal and nearly still. The scatter is turned as an object (see `view` in update),
    // so the flat panels always face the lens. ──
    const shots = [
      { t: 0, pos: [0, 0, REF_D + 0.5] as V3, look: [0, 0, 0] as V3, fov: REF_FOV },
      { t: 9, pos: [0, 0, REF_D] as V3, look: [0, 0, 0] as V3, fov: REF_FOV },
      { t: DUR, pos: [0, 0, REF_D] as V3, look: [0, 0, 0] as V3, fov: REF_FOV },
    ];

    const dot = (css: string) => `<span style="color:${css}">●</span>`;
    const ringG = (css: string) => `<span style="display:inline-block;width:0.75em;height:0.75em;border:1.7px solid ${css};border-radius:50%;box-sizing:border-box;vertical-align:-0.08em"></span>`;
    const lab = (api: FrameApi, key: string, pos: V3, text: string, alpha: number, o: LabelOpts = {}) => {
      if (alpha > 0.02) api.label(key, pos, text, { ...o, alpha });
    };
    const sm: LabelOpts = { size: 'sm', color: CSS.white };
    const md: LabelOpts = { size: 'md', color: CSS.white };
    /** Text whose box starts at (sx, sy) in layout pixels (left-aligned, vertically centered). */
    const at = (sx: number, sy: number) => sp(sx - 12, sy);

    /** Where the scatter sits at time t: alone in the middle of the page on the first slide, then in its slot. */
    const layoutA = (t: number) => {
      const intro = 1 - easeInOut(clamp((t - S2) / 2.0));
      return { scX: lerp(SC0[0], SCI[0], intro), scY: SC0[1] };
    };
    /** Text line `row` (0 nearest the box) above the axis box, `dx` layout pixels from the panel's left edge. */
    const rowPos = (t: number, dx: number, row: number): V3 => {
      const { scX, scY } = layoutA(t);
      return [scX + (dx - SC_PX[0] - 12) * WPP, scY + H_FIT / 2 + (26 + 22 * row) * WPP, 0];
    };
    const LG_X = 97; // x of the entries of the rows above the box (layout pixels at the panel's slot)

    // ── what is on screen when: shared by update, readouts and the notes ──
    const labK = (t: number) => smooth(T_LAB, T_LAB + 1.0, t);
    /** The scatter's title and the names of its floor axes, from the first slide on. */
    const introK = (t: number) => smooth(S1 + 0.6, S1 + 1.6, t);
    /** The target map appears at S2 and leaves when the error freezes at S5. */
    const tgtK = (t: number) => 1 - smooth(T_FREEZE, T_FREEZE + 0.35, t);
    /** 0 → 1 as the target alone gives way to the equation of three maps (S3). */
    const k3 = (t: number) => easeInOut(clamp((t - S3) / 0.8));
    /** 0 → 1 as the equation gives way to the pair of large maps (S5), once the target and the pile have left. */
    const k5 = (t: number) => easeInOut(clamp((t - T_FREEZE - 0.2) / 0.9));
    const mix3 = (a: Slot, b: Slot, k: number): Slot => [lerp(a[0], b[0], k), lerp(a[1], b[1], k)];
    /** Center x and size of map q, or of the color bar (q = 3) or the leftmost map (q = 4), at time t. */
    const geo = (q: number, t: number): Slot =>
      q === 0 ? mix3(L2_TGT, L3_TGT, k3(t))
      : q === 1 ? mix3(L3_ERR, L5_ERR, k5(t))
      : q === 2 ? mix3(L3_THD, L5_THD, k5(t))
      : q === 3 ? mix3(mix3(B2, B3, k3(t)), B5, k5(t))
      : mix3(mix3(L2_TGT, L3_ERR, k3(t)), L5_ERR, k5(t));
    /**
     * The color scale shared by the maps on screen: 1 for the target and the live error, then one scale per
     * compared pair. Returns the gain applied to the values, the bar's end value and how visible its labels are
     * (they dip while the scale changes).
     */
    const SCALE_KEYS: [number, number, boolean, number][] = [[T_FREEZE, scF, clipF, 0.9], [T_WRONG, scW, clipW, 1.2], [T_OUT, scT, clipT, 1.2], [S8, scRepeat, clipRepeat, 0.3]];
    const scaleAt = (t: number) => {
      let g = 1 / scY, end = scY, clip = false, vis = 1, prev = scY, prevC = false;
      for (const [at, s, c, dur] of SCALE_KEYS) {
        const k = smooth(at, at + dur, t);
        g += (1 / s - g) * k;
        if (k >= 0.5) { end = s; clip = c; }
        if (s !== prev || c !== prevC) vis *= Math.abs(2 * k - 1);
        prev = s; prevC = c;
      }
      return { g, end, clip, vis };
    };
    const signK = (t: number) => smooth(T_RUN - 0.2, T_RUN + 0.8, t);
    /** Blend toward the control network's frozen error (S6). */
    const wrongMix = (t: number) => smooth(T_WRONG, T_WRONG + 1.2, t) * (1 - smooth(T_OUT, T_OUT + 1.2, t));
    /**
     * Select the line under the maps and its opacity. S3 describes the loop. S5 shows one pass with new neurons.
     */
    const loopAt = (t: number): [number, number] => {
      // each line stays up to the end of its slide, where the film holds for reading, and leaves with the next one
      const a = smooth(S3 + 1.2, S3 + 2.0, t) * (1 - smooth(S4, S4 + 0.4, t));
      const b = smooth(T_CAV + D_CAV - 0.4, T_CAV + D_CAV + 0.4, t) * (1 - smooth(S6, S6 + 0.4, t));
      return a > 0.01 ? [0, a] : [1, b];
    };
    /**
     * Opacity of the ring, its leader and the one-neuron map. These follow one neuron from S3 through the end
     * of training at S4, including the turn of the scatter.
     */
    const oneShow = (t: number) => smooth(T_PICK, T_PICK + 0.6, t) * (1 - smooth(T_FREEZE, T_FREEZE + 0.5, t));
    const leadShow = (t: number) => oneShow(t) * smooth(T_PICK + 0.4, T_PICK + 1.0, t);
    /** The third map: the one neuron's output, the teal rebuild, the red rebuild, the theory's error. */
    const modes = (t: number) => {
      const one = oneShow(t);
      const cv = smooth(T_CAV - 0.4, T_CAV + 0.3, t) * (1 - smooth(T_WRONG, T_WRONG + 0.6, t));
      const w = smooth(T_WR - 0.4, T_WR + 0.3, t) * (1 - smooth(T_OUT, T_OUT + 0.6, t));
      const th = t >= S8 ? 1 : fpOk ? smooth(T_THEORY, T_THEORY + 0.8, t) : 0;
      return { one, cv, w, th };
    };
    const thirdMode = (t: number): 'one' | 'cav' | 'wr' | 'th' | 'none' => {
      const m = modes(t);
      // Rendering and probing share this selection, including partially visible transitions.
      if (m.th > 0.01) return 'th';
      if (m.w > 0.01) return 'wr';
      if (m.cv > 0.01) return 'cav';
      if (m.one > 0.01) return 'one';
      return 'none';
    };
    /** The middle map: 0 = live error, 1 = frozen error. `wrongMix` blends in the control network's error. */
    const frozenK = (t: number) => smooth(T_FREEZE, T_FREEZE + 0.9, t);
    /** The frozen-error map and its probe must use the same interpolated field. */
    const networkErrorAt = (t: number, out: Float32Array) => {
      fieldAt(gasRun, stepAt(gas, t), out);
      const fz = frozenK(t), wm = wrongMix(t);
      for (let k = 0; k < M; k++) {
        const own = lerp(out[k], rBar[k], fz);
        out[k] = lerp(own, rK[k], wm);
      }
    };
    /** End each rebuilt-error replay at the same training average that the comparison reports. */
    const rebuiltErrorAt = (control: boolean, t: number, out: Float32Array) => {
      const run = control ? wrong : fresh, pop = control ? wr : cav;
      const end = control ? T_WR + D_WR : T_CAV + D_CAV;
      fieldAt(run, stepAt(pop, t), out);
      const av = smooth(end - 0.6, end, t);
      for (let k = 0; k < M; k++) out[k] = lerp(out[k], run.rAvg[k], av);
    };
    // ── for the notes layer (read-only; `probeR` is scratch) ──
    const probeR = new Float32Array(M);
    /** Left edge and top edge of map q (layout pixels). */
    const hx = (q: number, t: number) => { const g = geo(q, t); return g[0] - g[1] / 2; };
    const hy = (q: number, t: number) => HM_Y - geo(q, t)[1] / 2;

    const chapter: Chapter = {
      id: 'meanfield',
      pillar: 'theory',
      kicker: 'LEARNING THEORY',
      title: 'MEAN FIELD',
      source: 'Vaintrob & Greenspan · Mean field sequence · LessWrong 2026',
      links: [{ text: 'Mean field sequence', url: 'https://www.lesswrong.com/posts/rduzFkTKx5pGKWKcL/mean-field-sequence-an-introduction' }],
      duration: DUR,
      group,
      beats,
      shots,
      results,
      interaction: {
        pick(t, p) {
          if (t < S1 || t >= DUR || pairedPopulations(t)) return null;
          // The convex outline of the projected box follows re-framing and the shared 2D viewport.
          // Outside it, wheel and pan continue to use the flat figure controls.
          type XY = [number, number];
          const points: XY[] = [];
          for (const x of [-WB, WB]) for (const y of [-HBOX, HBOX]) for (const z of [-WB, WB]) {
            const q = p.project(...scP(x, y, z));
            points.push([q[0], q[1]]);
          }
          points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
          const cross = (a: XY, b: XY, c: XY) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
          const half = (list: XY[]) => {
            const h: XY[] = [];
            for (const q of list) {
              while (h.length > 1 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
              h.push(q);
            }
            h.pop();
            return h;
          };
          const hull = [...half(points), ...half([...points].reverse())];
          const cursor: XY = [p.x, p.y];
          return hull.length >= 3 && hull.every((q, i) => cross(q, hull[(i + 1) % hull.length], cursor) >= 0) ? 'scatter' : null;
        },
        draggable: (region) => region === 'scatter',
        drag(t, region, dx, dy) {
          if (region !== 'scatter' || t < S1 || t >= DUR || pairedPopulations(t)) return;
          scatterTurn.setFromAxisAngle(scatterUp, dx * 0.006);
          scatterRotation.premultiply(scatterTurn);
          scatterTurn.setFromAxisAngle(scatterRight, dy * 0.006);
          scatterRotation.premultiply(scatterTurn).normalize();
        },
        reset: resetScatter,
        changed: scatterChanged,
      },

      update(t, api) {
        const fz = frozenK(t), wm = wrongMix(t), lk = labK(t);
        const paired = pairedPopulations(t), controlPair = paired && t >= S6 && t < S7;
        sc.visible = !paired;
        populationPair.visible = paired;

        // ── view of the scatter: oblique → top-down (w1, w2) after training ──
        const ez = (a: number, b: number) => easeInOut(clamp((t - a) / (b - a)));
        const authoredTop = ez(T_END, T_END + 2.0);
        // A slow turn of the scatter from the first slide to the end of training (about two degrees a second while the
        // cloud is introduced), with the view rising a little.
        const azObl = 58 - 34 * ez(0, S4 + 1) - 8 * ez(S4 + 1, T_END);
        const elev = 11 + 8 * ez(0, S4);
        const tilt = lerp(elev, 90, authoredTop) * (Math.PI / 180), az = azObl * (1 - authoredTop) * (Math.PI / 180);
        sc.rotation.set(tilt, az, 0);
        authoredNormal.set(0, 1, 0).applyQuaternion(sc.quaternion);
        sc.quaternion.premultiply(scatterRotation);
        turnedNormal.set(0, 1, 0).applyQuaternion(sc.quaternion);
        // The authored top view flattens marker heights to avoid perspective expansion. A local
        // tilt restores their real heights smoothly, together with the box and height-axis labels.
        const top = scatterChanged() ? authoredTop * (1 - smooth(0, 0.35, authoredNormal.angleTo(turnedNormal))) : authoredTop;
        // keep the projected height of the axis box constant while it turns (a little less from the top)
        sc.scale.setScalar((lerp(1, TOP_MUL, authoredTop) * H_FIT) / (2 * WB * (Math.abs(Math.cos(az)) + Math.abs(Math.sin(az))) * Math.sin(tilt) + 2 * HBOX * Math.cos(tilt)));
        const { scX, scY } = layoutA(t);
        sc.position.set(scX, scY + TOP_UP * WPP * authoredTop, 0);
        const pa = (dx: number, row: number) => rowPos(t, dx, row);
        group.updateMatrixWorld(true);
        group.getWorldQuaternion(parentRotation).invert();
        scatterRight.setFromMatrixColumn(api.camera.matrixWorld, 0).applyQuaternion(parentRotation).normalize();
        scatterUp.setFromMatrixColumn(api.camera.matrixWorld, 1).applyQuaternion(parentRotation).normalize();
        const oblique = (1 - top) * (1 - top);
        cage.uIntensity.value = 1 - top;

        // Only the first four slides use the 3D box; later comparisons use unclipped input weights.
        const range = aTop, SA = HBOX / range;
        {
          let n = 0;
          for (let v = -aTop; v <= aTop; v += aStep)
            aTk.seg(n++, [-WB, v * SA, -WB], [-WB - 0.16, v * SA, -WB], C.white, 0.9 * (1 - top));
          aTk.count = n;
          aTk.commit();
        }

        const gasK = 0.95 * Math.min(1, Math.pow(sizeN, 1.6));
        drawPop(gas, gasPts, t, SA, range, top, (a, out) => signCol(a, 0.4, out), () => gasK);
        if (controlPair) drawPop(kp, null, t, SA, range, top, () => {}, () => 0);
        let ruleNow = 0;
        for (let i = 0; i < N; i++) if (ruleOf(gas.theta[i * 3], gas.theta[i * 3 + 1], gas.theta[i * 3 + 2])) ruleNow++;
        // shadows: drawn while the scatter is seen at an angle, gone from the top, where they would lie under the dots
        const shK = gasK * oblique;
        shadow.object.visible = shK > 0.01;
        if (shK > 0.01) {
          const P = gasPts.positions, fy = -HBOX + 0.01;
          for (let i = 0; i < N; i++) {
            shadow.setPos(i, P[i * 3], fy, P[i * 3 + 2]);
            shadow.setColor(i, C.white, 0.2 * shK);
          }
          shadow.commit(true, true, false);
        }
        // Pose the compared populations independently of the 3D box's display clamps.
        if (paired) {
          projectPopulation(baseProjection, (controlPair ? kp : gas).theta, 0, t >= S8 ? undefined : C.slate);
          const sample = comparisonSample(t);
          newProjection.object.visible = t < S7;
          theoryProjection.object.visible = t >= S7 && !!sample;
          if (t < S7) {
            const pop = controlPair ? wr : cav;
            drawPop(pop, null, t, SA, range, top, () => {}, () => 0);
            projectPopulation(newProjection, pop.theta, 1, C.red);
          } else if (sample) projectPopulation(theoryProjection, sample, 1, t >= S8 ? undefined : C.red);
        }

        // ── the error handed to the middle map ──
        networkErrorAt(t, rIn);

        // ── maps ──
        const m0 = smooth(S2 + 0.6, S2 + 1.6, t), m12 = smooth(T_MAPS, T_MAPS + 1.0, t);
        const tg = m0 * tgtK(t);
        const { g: gain, end: scEnd, clip: scClip, vis: scVis } = scaleAt(t);
        const G0 = geo(0, t), G1 = geo(1, t), G2 = geo(2, t), GB = geo(3, t), GL = geo(4, t);
        place(heat[0].object, G0); place(mapFrames[0].object, G0);
        place(heat[1].object, G1); place(mapFrames[1].object, G1);
        place(heat[2].object, G2); place(mapFrames[2].object, G2); place(pile.object, G2); place(topCard.object, G2);
        place(barG, GB); place(keyFrame.object, GL);
        setHeat(heat[0], d.Y, gain);
        heat[0].uOpacity.value = tg;
        mapFrames[0].uIntensity.value = tg;
        keyFrame.uIntensity.value = m0;
        bar.uOpacity.value = m0;
        barFrame.uIntensity.value = m0;
        heat[1].uOpacity.value = m12;
        mapFrames[1].uIntensity.value = m12;
        setHeat(heat[1], rIn, gain);
        const md3 = modes(t), third = thirdMode(t);
        let h3 = 0, rampMax = 0;
        if (third === 'th') { setHeat(heat[2], t >= S8 ? repeat.rAvg : fp.r, gain); h3 = md3.th; }
        else if (third === 'wr') {
          // the live rebuild during the run, eased into its average over the second half at the end
          rebuiltErrorAt(true, t, rOut);
          setHeat(heat[2], rOut, gain); h3 = md3.w;
        } else if (third === 'cav') {
          rebuiltErrorAt(false, t, rOut);
          setHeat(heat[2], rOut, gain); h3 = md3.cv;
        } else if (third === 'one') {
          const i = pick, w1 = gas.theta[i * 3], w2 = gas.theta[i * 3 + 1], a = gas.theta[i * 3 + 2];
          for (let k = 0; k < M; k++) { const z = w1 * d.X1[k] + w2 * d.X2[k]; one[k] = z > 0 ? a * Math.tanh(z) : 0; rampMax = Math.max(rampMax, Math.abs(one[k])); }
          setHeat(heat[2], one, gain);
          h3 = md3.one;
        }
        heat[2].uOpacity.value = h3 * m12;
        // the third slot's frame: in the accent color while it holds the followed neuron's ramp (S3–S4), then, once
        // the slot has moved, with whichever map it holds, kept faintly between two of them
        topCard.uIntensity.value = m12 * md3.one;
        mapFrames[2].uIntensity.value = m12 * (1 - md3.one) * smooth(T_FREEZE + 1.0, T_FREEZE + 1.4, t) * Math.max(0.35, h3);
        pile.uIntensity.value = m12 * md3.one;

        // ring around the followed neuron and a leader from it to its ramp. The ring stays on the neuron while it
        // trains and while the scatter turns to the top. Both are drawn on the plane of the maps (z = 0), where the
        // neuron is seen from the camera, so that the leader meets the ring in any view.
        ring.clear();
        leader.clear();
        const ringK = paired ? 0 : oneShow(t), leadK = paired ? 0 : leadShow(t);
        if (ringK > 0.02) {
          const i = pick;
          const c = scP(clamp(gas.theta[i * 3], -WAX, WAX) * SW, heightOf(gas.theta[i * 3 + 2], SA, range, top), -clamp(gas.theta[i * 3 + 1], -WAX, WAX) * SW);
          const cam = api.camera.position, f = (cam.z - 0.05) / Math.max(1e-3, cam.z - c[2]);
          const cx = cam.x + (c[0] - cam.x) * f, cy = cam.y + (c[1] - cam.y) * f;
          const R = 0.27;
          ring.circle([cx, cy, 0.05], R, [1, 0, 0], [0, 1, 0], C.amber, 1.7 * ringK, 32);
          if (leadK > 0.02) {
            const yL = sp(0, HM_Y + G1[1] / 2 + LEAD_DY)[1], xL = sp(G1[0] - G1[1] / 2 - 34, 0)[0];
            const x3 = sp(G2[0] - G2[1] / 2 + 18, 0)[0], yB = sp(0, HM_Y + G2[1] / 2 + 6)[1];
            leader.polyline([[cx + R, cy, 0], [xL, cy, 0], [xL, yL, 0], [x3, yL, 0], [x3, yB, 0]], C.amber, 1.1 * leadK);
          }
        }
        ring.commit();
        leader.commit();

        if (!paired) {
          // ── labels: the scatter ──
          // on the first slide the title stands just above the box; it rises to its row when the color key comes in
          lab(api, 'a-title', pa(LG_X, 2), 'Hidden neuron weights', introK(t), {
            ...md,
            help: '<p>Each dot is a neuron. Its floor coordinates are the input weights $w_1$ and $w_2$. In the tilted view, its height gives the output weight $a$. Drag to rotate the cloud.</p>' +
              (shadow.object.visible ? '<p>Gray dots project the neurons onto the input-weight plane.</p>' : '') +
              (signK(t) > 0.5 ? `<p>The + and − floor marks give the sign of $${T.tex}$ in each input-weight direction.</p>` : ''),
          });
          // both entries in one label, so that the second follows the first at any label size
          lab(api, 'lg-ap', pa(LG_X, 1), `${dot(CSS.ice)} positive output weight&emsp;${dot(CSS.white)} negative`, introK(t), sm);
          lab(api, 'lg-pk', pa(LG_X, 0), `${ringG(CSS.amber)} followed neuron`, oneShow(t), { ...sm, color: CSS.amber });
          // the axis names stand on the lines of their tick labels, between the ticks
          const below: LabelOpts = { ...sm, align: 'center', dy: 30 };
          lab(api, 'ax-w1', scP(0, -HBOX, WB + 0.16), '$w_1$', introK(t), below);
          lab(api, 'tk-w1a', scP(-2 * SW, -HBOX, WB + 0.16), '−2', lk, below);
          lab(api, 'tk-w1b', scP(2 * SW, -HBOX, WB + 0.16), '2', lk, below);
          lab(api, 'ax-w2', scP(WB + 0.16, -HBOX, 0), '$w_2$', introK(t), { ...sm, dx: -6 });
          lab(api, 'tk-w2a', scP(WB + 0.16, -HBOX, 2 * SW), '−2', lk, { ...sm, dx: -6 });
          lab(api, 'tk-w2b', scP(WB + 0.16, -HBOX, -2 * SW), '2', lk, { ...sm, dx: -6 });
          // the floor's own labels hang under the lower end of its front edge, clear of the tick labels
          const fl = scP(-WB, -HBOX, WB + 0.16), fr = scP(WB, -HBOX, WB + 0.16);
          const fLow: V3 = [(fl[0] + fr[0]) / 2, Math.min(fl[1], fr[1]), Math.max(fl[2], fr[2])];
          // Count the displayed neurons that follow the sign rule during training (S4).
          lab(api, 'rule', fLow, `neurons that follow the sign rule: ${ruleNow} of ${N}`, smooth(T_RUN, T_RUN + 0.6, t) * (1 - smooth(T_FREEZE, T_FREEZE + 0.5, t)), { ...sm, align: 'left', dx: -199, dy: 73 });
          // plus and minus on the floor, where the target is positive and negative in that input direction
          {
            const k = 0.9 * signK(t);
            if (k > 0.02) {
              const dirs: [number, number][] = [[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 0], [0, 1], [-1, 0], [0, -1]];
              let mx = 0;
              for (const [u, v] of dirs) mx = Math.max(mx, Math.abs(T.fn(u / Math.hypot(u, v), v / Math.hypot(u, v))));
              dirs.forEach(([u, v], j) => {
                const y = T.fn(u / Math.hypot(u, v), v / Math.hypot(u, v));
                if (Math.abs(y) < 0.5 * mx) return;
                lab(api, `sg${j}`, scP(u * 2.55 * SW, -HBOX, -v * 2.55 * SW), y > 0 ? '+' : '−', k, { size: 'lg', color: y > 0 ? CSS.ice : CSS.white, align: 'middle' });
              });
            }
          }
          const aR: LabelOpts = { ...sm, align: 'right', dx: 6 };
          // The height title sits above its edge, clear of the tick labels.
          lab(api, 'ax-a', scP(-WB, HBOX + 0.12, -WB), 'Output weight $a$', lk * oblique, { ...sm, align: 'center', dy: -7 });
          lab(api, 'tk-aa', scP(-WB - 0.16, aLab * SA, -WB), String(aLab), lk * oblique, aR);
          lab(api, 'tk-ab', scP(-WB - 0.16, -aLab * SA, -WB), '−' + aLab, lk * oblique, aR);
          lab(api, 'tk-a0', scP(-WB - 0.16, 0, -WB), '0', lk * oblique, aR);
        } else {
          lab(api, 'a-title', pa(LG_X, 2), 'Input-weight distributions', 1, {
            ...md,
            help: (t >= S8 ? '<p>Both panels color each neuron by the sign of its output weight.</p>' : '<p>The panels share the same input-weight axes. Matching clouds can have different output weights.</p>') +
              (t >= S8 ? '<p>Both panels contain independently trained networks. Each used the same data grid, target, width, noise strength and training time.</p>' : t < S7 ? '<p>Each point is a neuron from the corresponding recorded training run.</p>' : fpOk ? `<p>Each panel contains ${n0(N)} neurons. The right panel samples the solved distribution. This sample is computed without training.</p>` : '<p>No solved distribution is available at these settings, so the right panel is empty.</p>'),
          });
          const titles = [controlPair ? `${lazy ? 'Mean-field' : 'GP'}-scaled network` : 'Trained network', t >= S8 ? 'Independent repeat' : t < S7 ? 'New neurons' : fpOk ? 'Equilibrium sample' : 'No solved distribution'];
          for (let panel = 0; panel < 2; panel++) {
            const left = POP_X[panel] - POP_SIZE / 2, bottom = HM_Y + POP_SIZE / 2;
            lab(api, `pop-head-${panel}`, sp(left, HM_Y - POP_SIZE / 2 - 28), titles[panel], 1, {
              ...md, color: panel && t < S8 ? CSS.red : CSS.slate,
            });
            lab(api, `pop-w1-${panel}`, sp(POP_X[panel] + POP_SIZE / 4, bottom), '$w_1$', 1, { ...sm, align: 'below', dy: 7 });
            if (panel === 0) lab(api, `pop-w2-${panel}`, sp(left - 29, HM_Y), '$w_2$', 1, { ...sm, align: 'right' });
            for (const value of [-inputRange, 0, inputRange]) {
              const number = String(value).replace('-', '−');
              lab(api, `pop-x-${panel}-${value}`, popAt(panel, value, -inputRange), number, 1, { ...sm, align: 'below', dy: 7 });
              lab(api, `pop-y-${panel}-${value}`, popAt(panel, -inputRange, value), number, 1, { ...sm, align: 'right', dx: -7 });
            }
          }
          if (t >= S8) {
            for (let panel = 0; panel < 2; panel++) {
              const nSign = panel ? ruleRepeat : ruleG;
              lab(api, `repeat-rule-${panel}`, sp(POP_X[panel], HM_Y + POP_SIZE / 2 + 78), `${nSign} of ${N} follow the sign rule`, 1, { ...sm, align: 'center' });
            }
            lab(api, 'repeat-key', sp(97, HM_Y + POP_SIZE / 2 + 115), `${dot(CSS.ice)} positive output weight&emsp;${dot(CSS.white)} negative`, 1, sm);
          }
        }

        // ── labels: the maps ──
        const t1 = 15; // title offset in px above a map's top edge
        const top0 = hy(0, t), top1 = hy(1, t), top2 = hy(2, t), topL = HM_Y - GL[1] / 2;
        lab(api, 'b-head', at(GL[0] - GL[1] / 2, topL - 58), 'Input grid', m0, {
          ...sm,
          help: '<p>Each square corresponds to an input. Maps compared side by side share a color scale. Values beyond the ends of that scale use the end color, marked by ≥ or ≤ on the bar.</p>',
        });
        lab(api, 'h1', at(hx(0, t), top0 - t1), `Target <span style="font-weight:400">$y = ${T.tex}$</span>`, tg, md);
        const errorMeaning = '<p>Error is the target minus the network’s output at each input.</p>';
        const frozenMeaning = '<p>The frozen map averages the error over the second half of training.</p>';
        // Fade through empty space so successive headings at the same anchor stay legible.
        const liveTitle = m12 * (1 - smooth(0, 0.5, fz));
        const frozenTitle = smooth(0.5, 1, fz) * (1 - smooth(0, 0.5, wm));
        const controlTitle = smooth(0.5, 1, wm);
        lab(api, 'h2a', at(hx(1, t), top1 - t1), 'Training error', liveTitle, { ...md, help: errorMeaning });
        lab(api, 'h2b', at(hx(1, t), top1 - t1), t >= S8 ? 'Frozen error, first run' : 'Frozen error', frozenTitle, { ...md, help: errorMeaning + frozenMeaning });
        lab(api, 'h2c', at(hx(1, t), top1 - t1), `Frozen error (${lazy ? 'mean-field' : 'GP'} scaling)`, controlTitle, {
          ...md,
          help: errorMeaning + frozenMeaning +
            (lazy ? '' : `<p>Mean squared error of the averaged map is ${f3(mseK)} with GP scaling and ${f3(mseB)} with mean-field scaling.</p>`),
        });
        const pileUp = ((STACK_N * STACK_D * G2[1]) / MS) * md3.one; // the pile's height above the top card
        lab(api, 'h3a', at(hx(2, t), top2 - t1 - pileUp), 'Neuron outputs', m12 * md3.one, {
          ...md,
          help: '<p>The top card shows the highlighted neuron. Its activation is zero on one side of $w\\cdot x=0$ and ramps up on the other. The output weight scales the ramp and sets its sign.</p>',
        });
        const rebuiltTitle = Math.max(md3.cv, md3.w) * (1 - smooth(0, 0.5, md3.th));
        lab(api, 'h3b', at(hx(2, t), top2 - t1), 'Rebuilt error', rebuiltTitle, {
          ...md,
          help: '<p>Each input’s rebuilt error is its target minus the new neurons’ combined output. At the end of the replay, the map shows this error averaged over the second half of their training.</p>',
        });
        lab(api, 'h3d', at(hx(2, t), top2 - t1), t >= S8 ? 'Frozen error, repeat' : 'Infinite-width error', smooth(0.5, 1, md3.th), {
          ...md,
          help: t >= S8 ? frozenMeaning : '<p>This map uses the average prediction over the entire equilibrium distribution. The red dots show one sample from that distribution.</p>',
        });
        // The equation at S3–S4 subtracts the scaled sum of the ramps from the target.
        const eqK = m12 * Math.min(tgtK(t), md3.one);
        const opAt = (xa: number, xb: number): V3 => sp((xa + xb) / 2, HM_Y);
        lab(api, 'op-eq', opAt(G1[0] + G1[1] / 2, G0[0] - G0[1] / 2), '$=$', eqK, { size: 'lg', color: CSS.white, align: 'middle' });
        lab(api, 'op-sum', opAt(G0[0] + G0[1] / 2, G2[0] - G2[1] / 2), `$-\\,${sNTex.replace('tfrac', 'frac')}\\sum_i$`, eqK, { size: 'lg', color: CSS.white, align: 'middle' });
        // One line for the loop beneath the maps at S3 and S5.
        {
          const [which, k] = loopAt(t);
          if (k > 0.02) {
            const x0 = G1[0] - G1[1] / 2, x1 = G2[0] + G2[1] / 2 + (which === 0 ? STACK_N * STACK_D : 0);
            const yb = HM_Y + Math.max(G1[1], G2[1]) / 2 + (which === 0 ? LEAD_DY + 14 : 46);
            const line = [
              'Error → individual neurons → combined output → error',
              'Frozen error → independently trained neurons → rebuilt error',
            ][which];
            lab(api, 'loop', [sp((x0 + x1) / 2, 0)[0], sp(0, yb)[1], 0], line, k, { ...sm, align: 'below' });
          }
        }
        const cL = sp(GL[0], HM_Y), hL = (hh * GL[1]) / MB;
        lab(api, 'x1', [cL[0], cL[1] - hL, 0], '$x_1$', m0, { ...sm, align: 'center', dy: 32 });
        lab(api, 'x1a', [cL[0] - hL, cL[1] - hL, 0], '−1', m0, { ...sm, align: 'center', dy: 32 });
        lab(api, 'x1b', [cL[0] + hL, cL[1] - hL, 0], '1', m0, { ...sm, align: 'center', dy: 32 });
        lab(api, 'x2', [cL[0] - hL, cL[1], 0], '$x_2$', m0, { ...sm, align: 'right', dx: 6 });
        lab(api, 'x2b', [cL[0] - hL, cL[1] + hL, 0], '1', m0, { ...sm, align: 'right', dx: 6, dy: 8 });
        // the color bar's end values: the scale of the pair of maps on screen
        // the followed neuron's ramp outgrows the target's scale as its output weight grows in training: its largest
        // values then take the end color, and the bar says so
        const clipNow = scClip || (md3.one > 0.5 && rampMax > scY * 1.001);
        const bk = GB[1] / MB, bx = sp(GB[0], 0)[0] + (bw + 0.16) * bk, bh = (hh - 0.03) * bk, endS = String(scEnd);
        lab(api, 'cb1', [bx, ROW_Y + bh, 0], (clipNow ? '≥ ' : '') + endS, m0 * scVis, { ...sm, dx: -4 });
        lab(api, 'cb0', [bx, ROW_Y, 0], '0', m0, { ...sm, dx: -4 });
        lab(api, 'cb2', [bx, ROW_Y - bh, 0], (clipNow ? '≤ ' : '') + '−' + endS, m0 * scVis, { ...sm, dx: -4 });
      },

      readouts(t) {
        // Each slide's numbers, shown on the notes page of that slide: the ones its text names, each row naming its
        // measure. Map differences compare time averages; per-step loss also includes fluctuations.
        const rows: Readout[] = [];
        const size = 'output-weight RMS over the second half of training';
        const stepLoss = 'mean per-step squared error over the second half of training';
        const cosName = 'cosine similarity of rebuilt and frozen error maps (1 = same pattern)';
        if (t >= T_END && t < S5) rows.push({ label: 'neurons that follow the sign rule at the end of training', value: `${ruleG} of ${N}`, note: `${rule0} at the start of training` });
        if (t >= S5 && t < S6)
          rows.push(
            { label: cosName, value: f2(cosC) },
            { label: 'difference between rebuilt and frozen maps, relative RMS', value: f2(relativeError(fresh.rAvg, rBar)), note: '0 means equal at every input' },
            { label: `${size}: network, then new neurons`, value: `${f2(rG)} · ${f2(rC)}`, note: 'prior RMS 1' },
            { label: `${stepLoss}, network and new neurons`, value: `${f4(gasRun.avg.loss)} · ${f4(fresh.avg.loss)}`, note: `${fmt(lossC, 2)} times the network's` },
          );
        if (t >= S6 && t < S7)
          rows.push(
            { label: 'difference between rebuilt and frozen maps, relative RMS', value: f2(relativeError(wrong.rAvg, rK)), note: '0 means equal at every input' },
            { label: cosName, value: f2(cosW), note: `${f2(cosC)} for the network's own error` },
            { label: `${size}: the ${lazy ? 'mean-field' : 'GP'}-scaled network, then the new neurons in its error`, value: `${f2(rK0)} · ${f2(rW)}`, note: 'prior RMS 1' },
            { label: `mean squared error of the averaged maps, ${lazy ? 'mean-field' : 'GP'} scaling and ${lazy ? 'GP' : 'mean-field'} scaling`, value: `${f4(mseK)} · ${f4(mseB)}`, note: `${f3(d.varY)} for an output of zero` },
          );
        if (t >= S8) return [
          { label: 'neurons following the sign rule, first run and repeat', value: `${ruleG} and ${ruleRepeat} of ${N}` },
          { label: 'cosine similarity of the two frozen errors', value: fmt(cosine(rBar, repeat.rAvg), 5) },
          { label: `${stepLoss}, first run and repeat`, value: `${f4(gasRun.avg.loss)} and ${f4(repeat.avg.loss)}` },
        ];
        if (t >= S7)
          rows.push({ label: 'mean squared error, averaged map and infinite-width prediction', value: `${f4(mseB)} · ${fpOk ? f4(fp.loss) : 'not solved'}`, note: `${f3(d.varY)} for a network that outputs zero` });
        return rows;
      },

      // Notes are specific to the comparison on this slide; details carry only useful extra depth.
      details(t): NoteDetails | undefined {
        if (t >= S4 && t < S5) return {
          title: 'How the error biases an output weight',
          body:
            '<p>The added training noise approximates sampling from a Bayesian posterior, which balances fitting the data against a prior preference for smaller weights. This is the learning assumption used for this calculation.</p>' +
            `<p>At continuous-time equilibrium, with the input weights $w$ and error $r$ held fixed, the output weight $a$ has a Gaussian distribution with mean $b\\,c_r(w)$ and variance 1. Here $c_r(w) = \\langle r\\,\\varphi(w\\cdot x)\\rangle$ averages the error times the activation across inputs, and $b = ${bTex}$ sets the strength of that response. The activation $\\varphi$ is nonnegative. Multiplying it by $a$ gives the neuron's signed output.</p>`,
        };
        if (t >= S5 && t < S6) return {
          title: 'Freezing an average error',
          body:
            '<p>Vaintrob and Greenspan used the completed network’s error as the target for each new neuron. Here training noise makes the error fluctuate, so the reconstruction freezes its average over the second half of training.</p>' +
            '<p>Each new neuron sees that same fixed error and uses the same noise strength as the trained network. A new neuron never changes the error it sees, even through its own output. This approximates an individually negligible contribution in a very wide network. The final map averages the error of the combined population over the second half of its training.</p>',
        };
        if (t >= S6 && t < S7) return {
          title: 'Matching the size of the errors',
          body: lazy
            ? '<p>The selected network uses Gaussian-process scaling, so the comparison network uses mean-field scaling. The new neurons keep the selected network’s response strength and readout scaling. This checks whether the other network\'s error satisfies that particular cycle.</p>'
            : '<p>Across twenty runs at the default settings, the mean square of the frozen GP error was 1.4 to 1.8 times that of the mean-field error. Rescaling each GP error to the root mean square of its paired mean-field error preserved its spatial pattern. With the same initial weights and training noise, the rebuilt maps still had cosine similarities below 0.45 to the rescaled errors.</p>' +
              '<p>In a separate comparison, fresh neurons used the GP network’s response strength and readout scaling with its original frozen error. Cosine similarities ranged from 0.53 to 0.98.</p>',
        };
        return undefined;
      },
      inspect(t): Hotspot[] {
        const rowA = (dx: number, row: number) => rowPos(t, dx, row);
        if (t < S2) return [{
          id: 'neurons',
          pos: rowA(LG_X - 16, 2),
          title: 'What an infinite-width limit assumes',
          body: '<p>A finite network can contain mechanisms resembling different infinite-width limits. Vaintrob and Greenspan use these limits as sources of simplified mechanisms for understanding real networks.</p>',
        }];
        if (t < S3) return [{
          id: 'maps',
          pos: [...sp(hx(0, t) - 24, hy(0, t) - 24)] as V3,
          title: 'The scale of the experiment',
          body: `<p>Vaintrob and Greenspan train 2,048 hidden neurons on the product of two inputs, with infinite data. This reconstruction trains ${N} neurons on an ${G} × ${G} grid${isProd ? '' : ', with a different target selected here'}. Its error is measured on those ${M} inputs.</p>`,
        }];
        if (t < S4) return [{
          id: 'ring',
          pos: [...sp(hx(2, t) - 24, hy(2, t) - 24)] as V3,
          title: 'The shape of one neuron\'s output',
          body: '<p>The activation $\\varphi(z) = \\max(0, \\tanh z)$ is zero for $z \\le 0$ and approaches 1 for large positive $z$. A neuron\'s output is $a\\,\\varphi(w\\cdot x)$, so its ramp levels off toward $a$. A negative output weight makes it slope downward.</p>',
        }];
        if (t < S5) {
          const fl = scP(-WB, -HBOX, WB + 0.16), fr = scP(WB, -HBOX, WB + 0.16);
          return [{
            id: 'signs',
            pos: [(fl[0] + fr[0]) / 2, Math.min(fl[1], fr[1]), Math.max(fl[2], fr[2])] as V3,
            dx: -206,
            dy: 51,
            title: 'Why some neurons have the other sign',
            body: '<p>Training noise spreads output weights around the value favored by the error. Near the boundaries between positive and negative regions, that preference is weak enough for some neurons to take the other sign. The rule therefore describes a distribution, with exceptions.</p>',
          }];
        }
        if (t < S6) return [{
          id: 'fresh',
          pos: rowA(LG_X - 16, 2),
          title: 'What a matching pattern establishes',
          body: '<p>The authors reported similar loss for the jointly and independently trained populations. Here, the cosine similarity compares two residual maps as lists of error values. It equals 1 whenever one is a positive multiple of the other, so matching patterns can still have different error sizes. The mean squared errors and typical output weights provide separate comparisons.</p>',
        }];
        if (t < S7) return [{
          id: 'wrong',
          pos: rowA(LG_X - 16, 2),
          title: 'How the control was constructed',
          body:
            '<p>The post leaves the recipe for its “more Gaussian process-like” control unstated. This reconstruction changes the network\'s readout scaling and adjusts each neuron\'s response to the error to match. The red neurons keep both the response strength and the readout scaling used in the preceding test.</p>' +
            '<p>The comparison asks whether the other network\'s error satisfies that same cycle. It leaves open whether the gray network has a self-consistent description under its own scaling.</p>',
        }];
        if (t < S8) return [{
          id: 'theory',
          pos: rowA(LG_X - 16, 0),
          title: fpOk ? 'What this comparison checks' : 'The missing infinite-width comparison',
          body: fpOk
            ? '<p>The post states the self-consistency condition in general form. The reconstruction solves it numerically for this grid. The right-hand map uses the mean over the solved distribution. The right-hand scatter shows a finite sample from it. The left-hand map averages the trained network’s errors over the second half of training.</p>'
            : lazy
              ? '<p>With Gaussian-process scaling, the response strength in this equation still depends on the finite width. It therefore supplies no infinite-width mean-field prediction for this setting.</p>'
              : '<p>The numerical calculation did not converge at these settings, so there is no solved distribution to compare with the trained network.</p>',
        }];
        return [{
          id: 'answer',
          pos: rowA(LG_X - 16, 0),
          title: 'What can repeat across training runs?',
          body: '<p>Both networks trained from scratch on the same target and input grid. The displayed weights are their final states. Each error map averages over the second half of its own run. The sign count and the error maps compare population statistics without pairing individual neurons across runs. Two runs check these quantities at the selected settings and leave the rest of the joint weight distribution untested.</p>',
        }];
      },
      probe(t, p) {
        if (pairedPopulations(t)) {
          const control = t >= S6 && t < S7, sample = comparisonSample(t);
          const fields = t < S7 ? [baseProjection, newProjection] : sample ? [baseProjection, theoryProjection] : [baseProjection];
          const populations = [(control ? kp : gas).theta, t < S7 ? (control ? wr : cav).theta : sample];
          let nearest = 100, panel = -1, index = -1;
          for (let q = 0; q < fields.length; q++) {
            const positions = fields[q].positions;
            for (let i = 0; i < fields[q].count; i++) {
              const v = p.project(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
              const distance = (v[0] - p.x) ** 2 + (v[1] - p.y) ** 2;
              if (distance < nearest) { nearest = distance; panel = q; index = i; }
            }
          }
          if (panel >= 0) {
            const positions = fields[panel].positions, weights = populations[panel]!;
            return {
              pos: [positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]],
              title: panel ? t >= S8 ? 'Neuron of the independent repeat' : t < S7 ? 'Independently trained neuron' : 'Neuron sampled from the infinite-width solution' : control ? `${lazy ? 'Mean-field' : 'GP'}-scaled network neuron` : 'Trained network neuron',
              rows: [
                { label: 'index $i$', value: String(index) },
                { label: '$w_1$', value: f3(weights[index * 3]) },
                { label: '$w_2$', value: f3(weights[index * 3 + 1]) },
                { label: 'output weight $a$', value: f3(weights[index * 3 + 2]) },
              ],
            };
          }
        } else {
          // ── the scatter: nearest drawn dot (neurons first, the theory's samples only when no neuron is near) ──
          const e = sc.matrixWorld.elements;
          const toW = (P: Float32Array, i: number): V3 => {
            const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
            return [e[0] * x + e[4] * y + e[8] * z + e[12], e[1] * x + e[5] * y + e[9] * z + e[13], e[2] * x + e[6] * y + e[10] * z + e[14]];
          };
          let bestD = 10 * 10, bestPop = -1, bestI = -1;
          const scan = (P: Float32Array, n: number, q: number, on: (i: number) => boolean, tol: number) => {
            for (let i = 0; i < n; i++) {
              if (!on(i)) continue;
              const w = toW(P, i), s = p.project(w[0], w[1], w[2]);
              if (s[2] < -1 || s[2] > 1) continue;
              const dd = (s[0] - p.x) ** 2 + (s[1] - p.y) ** 2;
              if (dd < bestD && dd < tol) { bestD = dd; bestPop = q; bestI = i; }
            }
          };
          if (gasPts.object.visible) scan(gasPts.positions, N, 0, () => true, 100);
          if (bestPop >= 0) {
            const P = gasPts.positions;
            let w1: number, w2: number, a: number;
            const th = gas.theta; w1 = th[bestI * 3]; w2 = th[bestI * 3 + 1]; a = th[bestI * 3 + 2];
            // the error this dot sees at t
            let field: ArrayLike<number>, fieldName: string;
            if (t < T_FREEZE) { fieldAt(gasRun, stepAt(gas, t), probeR); field = probeR; fieldName = 'the network\'s error now'; }
            else { field = rBar; fieldName = 'the frozen error'; }
            let c = 0, act = 0;
            for (let k = 0; k < M; k++) {
              const z = w1 * d.X1[k] + w2 * d.X2[k];
              if (z > 0) { c += field[k] * Math.tanh(z); act++; }
            }
            c /= M;
            const rows: Readout[] = [
              { label: 'index $i$', value: String(bestI) },
              { label: '$(w_1, w_2)$', value: `(${f2(w1)}, ${f2(w2)})` },
              { label: '$a$', value: f3(a) },
              { label: 'inputs with $w\\cdot x > 0$', value: `${act} of ${M}` },
              { label: 'error it sees', value: fieldName },
              { label: '$c_r(w) = \\langle r\\,\\varphi(w\\cdot x)\\rangle$', value: fmt(c, 3) },
              { label: '$b\\,c_r(w)$, equilibrium mean of $a$ if this error is frozen', value: f3(pl.b * c) },
            ];
            const out: Probe = {
              pos: toW(P, bestI),
              title: 'Neuron of the network',
              rows,
              note: `At continuous-time equilibrium in a fixed error, $a$ given $w$ is Gaussian with mean $b\\,c_r(w)$, $b = ${bTex}$, and variance 1.`,
            };
            return out;
          }

        }

        // ── the maps: the cell under the pointer ──
        const pl0 = p.plane;
        const m0 = smooth(S2 + 0.6, S2 + 1.6, t);
        if (pl0 && m0 > 0.5) {
          for (let q = 0; q < 3; q++) {
            if (q > 0 && t < T_MAPS + 0.5) break;
            if (q === 0 && tgtK(t) < 0.5) continue;
            const gq = geo(q, t), half = (gq[1] * WPP) / 2;
            const c0 = sp(gq[0], HM_Y);
            const u = (pl0[0] - (c0[0] - half)) / (2 * half), v = (pl0[1] - (c0[1] - half)) / (2 * half);
            if (u < 0 || u >= 1 || v < 0 || v >= 1) continue;
            const i = Math.floor(u * G), j = Math.floor(v * G), k = j * G + i;
            const pos: V3 = [c0[0] - half + ((i + 0.5) / G) * 2 * half, c0[1] - half + ((j + 0.5) / G) * 2 * half, 0.02];
            const xRow: Readout = { label: 'input $x = (x_1, x_2)$', value: `(${f3(d.X1[k])}, ${f3(d.X2[k])})` };
            if (q === 0) return { pos, title: `Target $y = ${T.tex}$`, rows: [xRow, { label: '$y(x)$', value: f3(d.Y[k]) }] };
            if (q === 1) {
              networkErrorAt(t, probeR);
              const fz = frozenK(t), wm = wrongMix(t);
              const shown = probeR[k];
              const title = wm > 0.5 ? 'Error of the gray network, frozen' : fz > 0.5 ? 'The frozen error' : 'What the network still gets wrong';
              return { pos, title, rows: [xRow, { label: 'error', value: f3(shown) }, { label: '$y(x)$', value: f3(d.Y[k]) }, { label: 'network output $y - $ error', value: f3(d.Y[k] - shown) }] };
            }
            const third = thirdMode(t);
            if (third === 'none') return null;
            if (third === 'one') {
              const ii = pick;
              const w1 = gas.theta[ii * 3], w2 = gas.theta[ii * 3 + 1], a = gas.theta[ii * 3 + 2], z = w1 * d.X1[k] + w2 * d.X2[k];
              return {
                pos, title: 'Output of the followed neuron',
                rows: [xRow, { label: '$w\\cdot x$', value: f3(z) }, { label: '$a$', value: f3(a) }, { label: '$a\\,\\varphi(w\\cdot x)$', value: f3(z > 0 ? a * Math.tanh(z) : 0) }],
                note: `The network adds this times $${sTex}$.`,
              };
            }
            if (third === 'th') return { pos, title: t >= S8 ? 'Frozen error of the second network' : 'Error predicted at infinite width', rows: [xRow, { label: 'error', value: f3(t >= S8 ? repeat.rAvg[k] : fp.r[k]) }, { label: 'frozen error of the run', value: f3(rBar[k]) }] };
            const cav3 = third === 'cav';
            rebuiltErrorAt(!cav3, t, probeR);
            const input = cav3 ? rBar[k] : rK[k];
            return {
              pos, title: 'Error rebuilt by the new neurons',
              rows: [xRow, { label: t >= (cav3 ? T_CAV + D_CAV : T_WR + D_WR) ? 'second-half mean error' : 'displayed rebuilt error', value: f3(probeR[k]) }, { label: cav3 ? 'frozen error' : 'the gray network\'s error', value: f3(input) }, { label: 'difference', value: f3(probeR[k] - input) }],
            };
          }
        }
        return null;
      },
    };
    return chapter;
  },
};

export default meanfield;
