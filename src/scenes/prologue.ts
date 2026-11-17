import * as THREE from 'three/webgpu';
import { attribute, uniform } from 'three/tsl';
import type { Chapter, ChapterFactory, Hotspot } from '../engine/types';
import { C, CSS, PILLAR_HUE, THEME, type RGB } from '../engine/palette';
import { CORNERS, VERT, CENTER, INK, teamTriangle, type Corner } from '../engine/mark';
import { TEAM_LABEL, TEAM_ROLE, TEAM_CHAPTERS, TITLES, LOOP_SIDE } from '../engine/cards';
import { ORDER } from './index';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { InkFill } from '../kit/ink';
import { mulberry32, gaussian, smooth, easeInOut, clamp } from '../lib/math';

/**
 * PROLOGUE: the program. Seven slides, each a step the viewer can name.
 *  0.1 a small network: circles are neurons, lines are weights; it trains;
 *  0.2 two of its weights in a row, a and b, and a plot with one axis for each. The network folds into a dot.
 *      The chain under the network shows what the two weights compute: output = b × a × input. The dot slides
 *      along a curve on which a × b stays the same, so the network's output stays the same;
 *  0.3 training from a random start moves the dot; a second run starts elsewhere and ends elsewhere;
 *  0.3b the plot tilts and the error on the data rises out of it as a surface, and thousands of runs slide downhill
 *      into the same curved valley while the view turns slowly (a slide of its own, so that the two runs are held
 *      before the surface replaces them);
 *  0.4 the view returns to the plot from above, the valley floor is drawn (the networks that fit the data): the
 *      product a × b determines the output; exact position along its curve leaves the output unchanged;
 *  0.5 the flat map of the three teams, built corner by corner, then what passes along each side in the direction
 *      of the loop, the way back, how to read the colors, and coarse-graining in the center;
 *  0.6 four plain research questions, before chapter 1 introduces the objects used to test them.
 * The first caption asks the question the finale returns to: which differences between two trained runs change
 * what the network computes.
 * A dot means one thing from 0.2 to 0.4: one network, placed by the values of weight a and weight b.
 *
 * The runs and the surface are computed here. They belong to the smallest network in which the two weights sit
 * in a row, weight a into a neuron and weight b out of it, so that the output is b·a·input. Each run trains a and
 * b by stochastic gradient descent on the same data, from its own random start and in its own random order of
 * the data. The error depends on a·b alone, so it is lowest on the whole curve a·b = (least-squares slope of the
 * data): the networks that fit the data. Gradient descent on it keeps a² − b² almost fixed, so the start decides
 * where along the curve a run ends; the order of the data scatters it a little across the curve. The network on
 * the left is drawn: its other weights are random numbers.
 *
 * The plot lives in its own group (`pivot`), which the scene tilts and turns for 0.3b and back for 0.4; its height
 * axis is scaled by `lift`, so the surface, the paths and the dots rise out of the flat plot together and settle back.
 */
const DUR = 56;
// T4 starts slide 0.3b: the plot turns from the two runs to the surface and the cloud
const T1 = 0.4, T2 = 5.6, T3 = 12.6, T4 = 22.2, T5 = 32.4, T6 = 40.0, T7 = 49.0;
type Win = [number, number];
// 0.1: the network draws itself and trains (its lines change darkness); this training is run 1
const DRAW: Win = [0.5, 2.4], TRAIN1: Win = [2.8, 4.9];
// 0.2: the network moves left and the plot appears; the chain of the two weights; the network takes the weights of
// the first point (PRE), folds into its dot (FOLD0), and the dot slides along the curve to the second point
const AXES: Win = [5.7, 6.5], CHAIN: Win = [6.3, 7.0], PRE: Win = [6.4, 7.0], FOLD0: Win = [7.2, 8.1], GLIDE: Win = [8.9, 11.2];
// 0.3: back to run 1's random start; it folds into its dot, and training moves the dot; then new random weights,
// a second fold, a second training
const REWIND: Win = [12.8, 13.4], FLY1: Win = [13.5, 14.4], RUN1: Win = [14.5, 16.5];
const RESET: Win = [16.8, 17.3], FLY2: Win = [17.4, 18.3], RUN2: Win = [18.4, 20.4];
// the dots of slides 0.2 and 0.3 are drawn larger, and settle to the size of the others as the other runs arrive
const BIG = 4.2;
// 0.3b: the network leaves; the plot tilts and the error surface rises; the other runs slide to their places one
// after another, faster and faster; the view keeps turning slowly until the slide ends
const NET_OFF: Win = [22.3, 23.0], TILT: Win = [22.4, 24.8], RAIN: Win = [24.3, 30.3], FLIGHT = 1.0, FADE = 1.2;
// 0.4: the view returns to the plot from above, the dots step back, and the valley floor and spread appear
const UNTILT: Win = [32.6, 34.8], DIM: Win = [32.8, 33.8], CURVE: Win = [34.8, 36.4], ALONG: Win = [36.8, 38.3];
// 0.5: the plot leaves; the triangle's outline, then one corner after another; then the sides of the loop one after
// another, and the way back
const LEAVE: Win = [40.1, 40.8], OUTLINE: Win = [40.8, 41.4], CORNER0 = 41.3, CORNER_GAP = 0.8;
const SIDE0 = 44.0, SIDE_GAP = 0.8, BACK = 46.6;
// 0.6: the triangle leaves; the chain draws itself one step after another, each check with its question beside it
const LEAVE6: Win = [49.1, 49.7], STEP0 = 49.9, STEP_GAP = 0.55;

const ease = (w: Win, t: number) => easeInOut(smooth(w[0], w[1], t));
const DEG = Math.PI / 180;

// ── the network: in the middle for slide 0.1, then on the left of the plot ──
const LAYERS = [4, 6, 6, 3];
const LX = [-4.2, -1.4, 1.4, 4.2];
const DY = 1.25, RING = 0.34;
const NET0 = 0, NET1 = -12.4, SLIDE: Win = [5.6, 6.4];
const netX = (t: number) => NET0 + (NET1 - NET0) * easeInOut(smooth(SLIDE[0], SLIDE[1], t));
// the chain under the network: input neuron → × a → neuron → × b → output neuron
const CHX = [-3.3, 0, 3.3], CHY = -5.4;

// ── the plot: weight a to the right, weight b upward, both from 0 to WMAX; the error upward out of the page ──
const WMAX = 1.05;
const PX0 = -3.2, PW = 11, PY0 = -5.2, PH = 10.4;
const CX = PX0 + PW / 2, CY = PY0 + PH / 2; // the point the plot turns about
const HS = 7; // page units of height per unit of root-mean-square error
// the 3-D view of 0.3b: the plot tilts back by THETA and turns from PHI0 to PHI1 about its height axis
const THETA = 58 * DEG, PHI0 = 28 * DEG, PHI1 = 52 * DEG, DROP3 = -1.2; // DROP3: how far the tilted plot moves down

// ── the runs on the plot (see the comment at the top) ──
const NR = 10000; // runs
const NDATA = 64, SLOPE = 0.25, LABEL_NOISE = 0.2; // the data: y = 0.25·x plus noise, the same 64 examples for every run
const START_SD = 0.5, START_MIN = 0.03, START_MAX = 1; // a random start: each weight is the size of a normal draw with this spread, drawn again until it lies in [START_MIN, START_MAX]
const PICK_LO = 0.08, PICK_HI = 0.92, PICK_PATH = 0.3; // how runs 1 and 2 are picked (see below)
const LR = 0.03, EPOCHS = 8;
const LANDED = 0.02; // a run has reached the curve once |a·b − fit| falls below this
const K = 20; // segments of each drawn path
const DATA_SEED = 5, RUN_SEED0 = 1000; // the data's seed; run r uses RUN_SEED0 + r for its start and its order
// the two networks of slide 0.2: weight a of each; weight b is chosen so that a·b is the fit
const PA1 = 0.35, PA2 = 0.85;
// the surface: a G × G grid of (a, b); grid lines on it every GL-th row
const G = 64, GL = 4;

// ── the triangle (centered on the origin), oriented as the emblem: Data on top, theory bottom right, tools bottom left ──
const SIDE = 10.5;
const tri = (p: [number, number]): RGB => [(p[0] - CENTER[0]) * SIDE, (CENTER[1] - p[1]) * SIDE, 0];
const TEAM_ORDER: Corner[] = ['data', 'theory', 'tools'];
// the sides of the loop, in the direction it runs: theory → tools, tools → data, data → theory
const LOOP: [Corner, Corner][] = [['theory', 'tools'], ['tools', 'data'], ['data', 'theory']];

// ── 0.6 four plain questions; the next chapter introduces the objects used to test them ──
const CH_X = -3.5, CH_Y0 = 3.7, CH_DY = 1.72;
const chainY = (k: number) => CH_Y0 - k * CH_DY;
const QUESTIONS = [
  ['Recovery', 'Can we recover a known pattern from the network?'],
  ['Training', 'Was the pattern already readable at initialization?'],
  ['Intervention', 'Does changing the pattern change a prediction?'],
  ['Approximation', 'How much could the details we leave out<br>change that prediction?'],
];
const URL_INTRO = 'https://www.lesswrong.com/posts/nbSJhbLERTZFeNxY7/introducing-piramid-physics-informed-research-for-ambitious';
/** The second ink for label lines that carry meaning: no lighter than the axes (styles.css, --ink-2). */
const INK2 = 'var(--ink-2)';

interface Seg { ax: number; ay: number; bx: number; by: number; ring: boolean; d: number }

/** A shaded surface: a fixed triangle mesh whose vertices carry their own amount of ink. */
class SurfaceInk {
  readonly object: THREE.Mesh;
  readonly uIntensity = uniform(0);
  constructor(positions: Float32Array, colors: Float32Array, index: Uint32Array) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    g.setIndex(new THREE.BufferAttribute(index, 1));
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = (attribute('color', 'vec3') as any).mul(this.uIntensity);
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = false;
    m.side = THREE.DoubleSide;
    m.blending = THREE.AdditiveBlending;
    this.object = new THREE.Mesh(g, m);
    this.object.frustumCulled = false;
  }
}

const prologue: ChapterFactory = {
  id: 'prologue',
  label: 'prologue · the program',
  async build() {
    const rnd = mulberry32(11);
    const g = gaussian(rnd);
    const group = new THREE.Group();

    // ── 0.1 the network: rings for neurons, lines for weights ──
    const nodes: [number, number][][] = LAYERS.map((n, l) => Array.from({ length: n }, (_, k) => [LX[l], ((n - 1) / 2 - k) * DY] as [number, number]));
    const segs: Seg[] = [];
    const xmin = LX[0] - RING, xmax = LX[LX.length - 1] + RING;
    const delay = (x: number) => 1 - (x - xmin) / (xmax - xmin); // the right of the network leaves for the dot first
    const RSEG = 28;
    nodes.flat().forEach(([x, y]) => {
      for (let i = 0; i < RSEG; i++) {
        const a0 = (i / RSEG) * Math.PI * 2, a1 = ((i + 1) / RSEG) * Math.PI * 2;
        segs.push({ ax: x + RING * Math.cos(a0), ay: y + RING * Math.sin(a0), bx: x + RING * Math.cos(a1), by: y + RING * Math.sin(a1), ring: true, d: delay(x) });
      }
    });
    const firstWeight = segs.length;
    // The two weights of the plot sit in a row: weight a from the top input neuron into the top neuron of the first
    // hidden layer, weight b from that neuron on. The weight labeled in slide 0.1 is the bottom horizontal line.
    let segA = -1, segB = -1, segW = -1;
    for (let l = 0; l + 1 < LAYERS.length; l++)
      for (const p of nodes[l])
        for (const q of nodes[l + 1]) {
          const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
          if (l === 0 && p === nodes[0][0] && q === nodes[1][0]) segA = segs.length;
          if (l === 1 && p === nodes[1][0] && q === nodes[2][0]) segB = segs.length;
          if (l === 1 && p === nodes[1][5] && q === nodes[2][5]) segW = segs.length;
          segs.push({ ax: p[0] + ux * RING, ay: p[1] + uy * RING, bx: q[0] - ux * RING, by: q[1] - uy * RING, ring: false, d: delay((p[0] + q[0]) / 2) });
        }
    const NW = segs.length - firstWeight;
    // The other weights, before and after training, for run 1 (…a) and run 2 (…b); 0..1 is how dark the line is.
    // They are drawn, not trained.
    const draw = (s: number) => Float32Array.from({ length: NW }, () => Math.min(1, Math.abs(g()) * s));
    const w0a = draw(0.45), w1a = draw(0.6), w0b = draw(0.45), w1b = draw(0.6);
    const A = segA - firstWeight, B = segB - firstWeight;

    // ── the data and the error ──
    const rd = mulberry32(DATA_SEED), gd = gaussian(rd);
    const X = Float64Array.from({ length: NDATA }, () => gd());
    const Y = X.map((x) => SLOPE * x + LABEL_NOISE * gd());
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < NDATA; i++) { sxy += X[i] * Y[i]; sxx += X[i] * X[i]; syy += Y[i] * Y[i]; }
    const FIT = sxy / sxx; // a·b of the networks that fit the data best: the curve every run ends near
    /** Root-mean-square error on the data of the network whose two weights multiply to p. */
    const rmse = (p: number) => Math.sqrt(Math.max(0, (sxx * p * p - 2 * sxy * p + syy) / NDATA));
    const RMIN = rmse(FIT);
    /** Height of the surface over (a, b), in page units: the error above its lowest value. */
    const hgt = (a: number, b: number) => HS * (rmse(a * b) - RMIN);

    // ── the runs on the plot ──
    const STEPS = EPOCHS * NDATA;
    const sa = new Float32Array(NR), sb = new Float32Array(NR), fa = new Float32Array(NR), fb = new Float32Array(NR);
    const pa = new Float32Array(NR * (K + 1)), pb = new Float32Array(NR * (K + 1)); // each run's drawn path, K+1 points evenly spaced along it
    const plen = new Float32Array(NR);
    {
      const ra = new Float64Array(STEPS + 2), rb = new Float64Array(STEPS + 2), qa = new Float64Array(STEPS + 2), qb = new Float64Array(STEPS + 2), cum = new Float64Array(STEPS + 2);
      const order = Int32Array.from({ length: NDATA }, (_, i) => i);
      for (let r = 0; r < NR; r++) {
        const rs = mulberry32(RUN_SEED0 + r), gs = gaussian(rs);
        const start = () => { for (;;) { const v = Math.abs(gs()) * START_SD; if (v >= START_MIN && v <= START_MAX) return v; } };
        let a = start(), b = start();
        sa[r] = a; sb[r] = b;
        let n = 0, landed = false;
        ra[n] = a; rb[n] = b; n++;
        for (let e = 0; e < EPOCHS; e++) {
          for (let i = NDATA - 1; i > 0; i--) { const j = (rs() * (i + 1)) | 0, s = order[i]; order[i] = order[j]; order[j] = s; }
          for (let m = 0; m < NDATA; m++) {
            const x = X[order[m]], err = a * b * x - Y[order[m]];
            const ga = 2 * err * x * b, gb = 2 * err * x * a;
            a -= LR * ga; b -= LR * gb;
            if (!landed) { ra[n] = a; rb[n] = b; n++; if (Math.abs(a * b - FIT) < LANDED) landed = true; }
          }
        }
        fa[r] = a; fb[r] = b;
        // the drawn path: the way to the curve, lightly smoothed, then the last stretch to the trained weights
        for (let i = 0; i < n; i++) {
          let s0 = 0, s1 = 0, c = 0;
          for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) { s0 += ra[j]; s1 += rb[j]; c++; }
          qa[i] = i === 0 ? ra[0] : s0 / c; qb[i] = i === 0 ? rb[0] : s1 / c;
        }
        qa[n] = a; qb[n] = b;
        const m = n + 1;
        cum[0] = 0;
        for (let i = 1; i < m; i++) cum[i] = cum[i - 1] + Math.hypot(qa[i] - qa[i - 1], qb[i] - qb[i - 1]);
        const L = cum[m - 1];
        plen[r] = L;
        let j = 0;
        for (let k = 0; k <= K; k++) {
          const s = (L * k) / K;
          while (j < m - 2 && cum[j + 1] < s) j++;
          const span = cum[j + 1] - cum[j], f = span > 1e-12 ? clamp((s - cum[j]) / span) : 0;
          pa[r * (K + 1) + k] = qa[j] + (qa[j + 1] - qa[j]) * f;
          pb[r * (K + 1) + k] = qb[j] + (qb[j + 1] - qb[j]) * f;
        }
      }
    }
    // runs 1 and 2 of slide 0.3: two runs that start well inside the curve, on long paths, and end far apart along
    // it: the ones whose end points have a² − b² closest to the 8th and the 92nd percentile of all runs
    const dEnd = (r: number) => fa[r] * fa[r] - fb[r] * fb[r];
    const dSorted = Array.from({ length: NR }, (_, r) => dEnd(r)).sort((p, q) => p - q);
    const dLo = dSorted[Math.floor(PICK_LO * (NR - 1))], dHi = dSorted[Math.floor(PICK_HI * (NR - 1))];
    let R1 = -1, R2 = -1;
    for (let r = 0; r < NR; r++) {
      if (!(sa[r] >= 0.06 && sb[r] >= 0.06 && sa[r] * sb[r] <= 0.08 && plen[r] >= PICK_PATH)) continue;
      if (R1 < 0 || Math.abs(dEnd(r) - dLo) < Math.abs(dEnd(R1) - dLo)) R1 = r;
      if (R2 < 0 || Math.abs(dEnd(r) - dHi) < Math.abs(dEnd(R2) - dHi)) R2 = r;
    }
    // how the end points spread along the curve: the middle SPAN of their values of weight a
    const SPAN = 0.9, SPAN_PCT = `${Math.round(SPAN * 100)}%`;
    const aSorted = Array.from(fa).sort((p, q) => p - q);
    const aLo = aSorted[Math.floor(((1 - SPAN) / 2) * (NR - 1))], aHi = aSorted[Math.floor(((1 + SPAN) / 2) * (NR - 1))];
    // how far the end points scatter across the curve (root mean square of the distance to it, in weight units)
    let across = 0, startsAbove = 0;
    for (let r = 0; r < NR; r++) {
      const v = (fa[r] * fb[r] - FIT) / Math.hypot(fa[r], fb[r]);
      across += v * v;
      if (sa[r] * sb[r] > FIT) startsAbove++;
    }
    across = Math.sqrt(across / NR);
    let negative = 0;
    for (let r = 0; r < NR; r++) if (fa[r] < 0 || fb[r] < 0) negative++;
    // the two networks of slide 0.2, both on the curve
    const PB1 = FIT / PA1, PB2 = FIT / PA2;

    // ── where things stand on the page ──
    const px = (a: number) => PX0 + (a / WMAX) * PW, py = (b: number) => PY0 + (b / WMAX) * PH;
    const at = (t: number, q: [number, number], dx = 0, dy = 0): RGB => [netX(t) + q[0] + dx, q[1] + dy, 0];
    const tmp: [number, number] = [0, 0], tmp2: [number, number] = [0, 0], tmpP: [number, number] = [0, 0], tmp1: [number, number] = [0, 0], tmpR2: [number, number] = [0, 0], tmpG: [number, number] = [0, 0];
    /** Point u (0..1) of run r's drawn path, in weight units. */
    const pathAt = (r: number, u: number, out: [number, number]) => {
      const x = clamp(u) * K, j = Math.min(K - 1, Math.floor(x)), f = x - j, o = r * (K + 1) + j;
      out[0] = pa[o] + (pa[o + 1] - pa[o]) * f;
      out[1] = pb[o] + (pb[o + 1] - pb[o]) * f;
      return out;
    };
    /** A point of the plot: page position of weights (a, b), on the surface. */
    const onSurf = (a: number, b: number): RGB => [px(a), py(b), hgt(a, b)];
    const startOf = (r: number): RGB => onSurf(pa[r * (K + 1)], pb[r * (K + 1)]);
    const endOf = (r: number): RGB => onSurf(pa[r * (K + 1) + K], pb[r * (K + 1) + K]);
    /** The point of slide 0.2's glide at fraction g: along the curve a·b = FIT, evenly in log a. */
    const glideAt = (g: number, out: [number, number]) => { out[0] = PA1 * Math.pow(PA2 / PA1, g); out[1] = FIT / out[0]; return out; };

    // the weights a and b of the drawn network at time t
    const abAt = (t: number, out: [number, number]) => {
      if (t < T2) return pathAt(R1, ease(TRAIN1, t), out); // 0.1: run 1 trains
      if (t < REWIND[0]) {
        // 0.2: from run 1's trained weights to the first point of the curve, then along the curve
        const e = ease(PRE, t);
        pathAt(R1, 1, out);
        const ea = out[0], eb = out[1];
        glideAt(ease(GLIDE, t), out);
        out[0] = ea + (out[0] - ea) * e; out[1] = eb + (out[1] - eb) * e;
        return out;
      }
      if (t < RESET[0]) {
        // 0.3, run 1: back to its random start, then training
        if (t >= RUN1[0]) return pathAt(R1, ease(RUN1, t), out);
        const e = ease(REWIND, t);
        glideAt(1, out);
        const ga = out[0], gb = out[1];
        pathAt(R1, 0, out);
        out[0] = ga + (out[0] - ga) * e; out[1] = gb + (out[1] - gb) * e;
        return out;
      }
      // run 2: new random weights, then training
      const e = ease(RESET, t), s = ease(RUN2, t);
      pathAt(R2, s, out);
      const e1a = pa[R1 * (K + 1) + K], e1b = pb[R1 * (K + 1) + K], s2a = pa[R2 * (K + 1)], s2b = pb[R2 * (K + 1)];
      out[0] += (e1a - s2a) * (1 - e);
      out[1] += (e1b - s2b) * (1 - e);
      return out;
    };
    // the other weights of the drawn network: run 1 trains (0.1); they stay (0.2); back to run 1's start and
    // training again (0.3); new random weights and run 2's training
    const otherAt = (j: number, t: number) =>
      t < T2
        ? w0a[j] + (w1a[j] - w0a[j]) * ease(TRAIN1, t)
        : t < RESET[0]
          ? w1a[j] + (w0a[j] - w1a[j]) * ease(REWIND, t) + (w1a[j] - w0a[j]) * ease(RUN1, t)
          : w1a[j] + (w0b[j] - w1a[j]) * ease(RESET, t) + (w1b[j] - w0b[j]) * ease(RUN2, t);
    const dark = (w: number) => 0.12 + 0.88 * clamp(w);

    const net = new LineField(segs.length, { width: 1.6 });
    const fly = new LineField(segs.length, { width: 1.6 }); // the copy of the network that folds into its dot
    for (let i = 0; i < segs.length; i++) { net.push([0, 0, 0], [0, 0, 0], C.white, 0); fly.push([0, 0, 0], [0, 0, 0], C.white, 0); }
    net.commit(); fly.commit();
    group.add(net.object, fly.object);
    // weight a and weight b, drawn heavier from slide 0.2 on
    const heavy = new LineField(2, { width: 4.2 });
    heavy.push([0, 0, 0], [0, 0, 0], C.white, 0); heavy.push([0, 0, 0], [0, 0, 0], C.white, 0);
    heavy.commit();
    group.add(heavy.object);
    // the chain: three neurons in a row, the two weights between them (heavy, as dark as the weights) and arrowheads
    const chainRings = new LineField(3 * RSEG, { width: 1.6 });
    const chainLines = new LineField(2, { width: 4.2 });
    const chainHeads = new LineField(4, { width: 1.6 });
    for (let i = 0; i < 3 * RSEG; i++) chainRings.push([0, 0, 0], [0, 0, 0], C.white, 0);
    for (let i = 0; i < 2; i++) chainLines.push([0, 0, 0], [0, 0, 0], C.white, 0);
    for (let i = 0; i < 4; i++) chainHeads.push([0, 0, 0], [0, 0, 0], C.white, 0);
    group.add(chainRings.object, chainLines.object, chainHeads.object);

    // ── the plot: everything in it is written in page coordinates with the height as z, inside `inner` ──
    const pivot = new THREE.Group(), inner = new THREE.Group();
    const viewCenter = new THREE.Group(), viewTurn = new THREE.Group(), uncenter = new THREE.Group();
    pivot.name = 'prologue-error-plot';
    viewCenter.name = 'prologue-surface-center';
    viewTurn.name = 'prologue-surface-view';
    inner.position.set(-CX, -CY, 0);
    uncenter.add(inner);
    viewTurn.add(uncenter);
    viewCenter.add(viewTurn);
    pivot.add(viewCenter);
    group.add(pivot);
    let surfaceYaw = 0, surfacePitch = 0, surfaceZoom = 1, surfaceEngaged = false;
    const surfaceStage = (t: number) => t >= T4 && t < T5;
    const resetSurface = () => { surfaceYaw = 0; surfacePitch = 0; surfaceZoom = 1; surfaceEngaged = false; };
    const desiredView = new THREE.Quaternion(), desiredEuler = new THREE.Euler();
    /** How far the plot has gone into 3-D at time t (0 flat, 1 tilted with the surface up), and its turn. */
    const tiltAt = (t: number) => ease(TILT, t) * (1 - ease(UNTILT, t));
    const turnAt = (t: number) => PHI0 + (PHI1 - PHI0) * easeInOut(smooth(TILT[1] - 0.6, T5, t));
    const pose = (t: number) => {
      // A first gesture during the reveal inspects the complete 3-D pose immediately. The stored
      // offsets remain view-only; reset restores the authored pose at the unchanged timeline time.
      const inspecting = surfaceEngaged && surfaceStage(t);
      const m = inspecting ? 1 : tiltAt(t), turn = inspecting ? PHI1 : turnAt(t);
      pivot.position.set(CX * (1 - m), CY + DROP3 * m, 0);
      pivot.rotation.set(-THETA * m, 0, turn * m);
      pivot.scale.set(1, 1, Math.max(1e-3, m));
      viewTurn.quaternion.identity();
      viewTurn.scale.setScalar(inspecting ? surfaceZoom : 1);
      if (inspecting && (surfacePitch !== 0 || surfaceYaw !== 0)) {
        desiredEuler.set(-THETA + surfacePitch, 0, turn + surfaceYaw);
        desiredView.setFromEuler(desiredEuler);
        viewTurn.quaternion.copy(pivot.quaternion).invert().multiply(desiredView);
      }
      pivot.updateMatrixWorld(true);
      return m;
    };
    const V = new THREE.Vector3();
    /** World position of a page point of the plot (height z) under the current pose. */
    const W = (p: ArrayLike<number>, dz = 0): RGB => { V.set(p[0], p[1], (p[2] ?? 0) + dz); inner.localToWorld(V); return [V.x, V.y, V.z]; };

    // the surface: shaded mesh, grid lines along a and along b, and contours of equal error
    const surfPos = new Float32Array((G + 1) * (G + 1) * 3), surfCol = new Float32Array((G + 1) * (G + 1) * 3);
    const surfIdx = new Uint32Array(G * G * 6);
    {
      // light from the upper left of the plot and above; a face turned away from it takes more ink
      const L = new THREE.Vector3(-0.55, 0.45, 0.7).normalize();
      const n = new THREE.Vector3();
      const da = WMAX / G;
      for (let i = 0; i <= G; i++)
        for (let j = 0; j <= G; j++) {
          const a = (i / G) * WMAX, b = (j / G) * WMAX, o = (i * (G + 1) + j) * 3;
          const p = onSurf(a, b);
          surfPos[o] = p[0]; surfPos[o + 1] = p[1]; surfPos[o + 2] = p[2];
          const dzx = (hgt(Math.min(WMAX, a + da), b) - hgt(Math.max(0, a - da), b)) / (px(Math.min(WMAX, a + da)) - px(Math.max(0, a - da)));
          const dzy = (hgt(a, Math.min(WMAX, b + da)) - hgt(a, Math.max(0, b - da))) / (py(Math.min(WMAX, b + da)) - py(Math.max(0, b - da)));
          n.set(-dzx, -dzy, 1).normalize();
          const k = 0.02 + 0.13 * (1 - Math.max(0, n.dot(L)));
          surfCol[o] = C.white[0] * k; surfCol[o + 1] = C.white[1] * k; surfCol[o + 2] = C.white[2] * k;
        }
      let q = 0;
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const v = i * (G + 1) + j;
          surfIdx.set([v, v + G + 1, v + 1, v + 1, v + G + 1, v + G + 2], q);
          q += 6;
        }
    }
    const surf = new SurfaceInk(surfPos, surfCol, surfIdx);
    inner.add(surf.object);
    const NGL = Math.floor(G / GL) - 1; // grid lines in each direction, inside the edges
    const grid = new LineField(2 * NGL * G + 4 * G);
    for (let i = 1; i <= NGL; i++) {
      const v = (i * GL * WMAX) / G;
      for (let j = 0; j < G; j++) {
        const u0 = (j / G) * WMAX, u1 = ((j + 1) / G) * WMAX;
        grid.push(onSurf(v, u0), onSurf(v, u1), C.white, 0.5);
        grid.push(onSurf(u0, v), onSurf(u1, v), C.white, 0.5);
      }
    }
    // the edges of the surface, a little darker
    for (let j = 0; j < G; j++) {
      const u0 = (j / G) * WMAX, u1 = ((j + 1) / G) * WMAX;
      grid.push(onSurf(0, u0), onSurf(0, u1), C.white, 0.9);
      grid.push(onSurf(WMAX, u0), onSurf(WMAX, u1), C.white, 0.9);
      grid.push(onSurf(u0, 0), onSurf(u1, 0), C.white, 0.9);
      grid.push(onSurf(u0, WMAX), onSurf(u1, WMAX), C.white, 0.9);
    }
    grid.commit();
    grid.uIntensity.value = 0;
    inner.add(grid.object);
    // contours: the curves a·b = p on which the error rises by a fixed step above its lowest value
    const CSTEP = 0.06, CN0 = 80;
    const levels: number[] = [];
    for (let d = CSTEP; RMIN + d < rmse(WMAX * WMAX); d += CSTEP) {
      // sxx p² − 2 sxy p + syy = N (RMIN + d)²: the two products p on either side of the fit
      const c = syy - NDATA * (RMIN + d) ** 2, disc = sxy * sxy - sxx * c;
      if (disc <= 0) continue;
      for (const p of [(sxy - Math.sqrt(disc)) / sxx, (sxy + Math.sqrt(disc)) / sxx]) if (p > 1e-4 && p < WMAX * WMAX) levels.push(p);
    }
    const contours = new LineField(levels.length * CN0);
    for (const p of levels) {
      // from b = WMAX down to a = WMAX along a·b = p, evenly in log a
      const a0 = p / WMAX, a1 = WMAX;
      for (let i = 0; i < CN0; i++) {
        const u = a0 * Math.pow(a1 / a0, i / CN0), v = a0 * Math.pow(a1 / a0, (i + 1) / CN0);
        contours.push(onSurf(u, p / u), onSurf(v, p / v), C.white, 1);
      }
    }
    contours.commit();
    contours.uIntensity.value = 0;
    inner.add(contours.object);

    // the axes of the plot, each with an arrowhead; the error axis stands at the far end of the weight-a axis
    const axes = new LineField(6, { width: 1.5 });
    const AH = 0.28, XE = px(WMAX), YE = py(WMAX);
    axes.push([PX0, PY0, 0], [XE, PY0, 0], C.white, 0.85);
    axes.push([XE, PY0, 0], [XE - AH, PY0 + AH * 0.5, 0], C.white, 0.85);
    axes.push([XE, PY0, 0], [XE - AH, PY0 - AH * 0.5, 0], C.white, 0.85);
    axes.push([PX0, PY0, 0], [PX0, YE, 0], C.white, 0.85);
    axes.push([PX0, YE, 0], [PX0 - AH * 0.5, YE - AH, 0], C.white, 0.85);
    axes.push([PX0, YE, 0], [PX0 + AH * 0.5, YE - AH, 0], C.white, 0.85);
    axes.commit();
    inner.add(axes.object);
    const ZTOP = hgt(WMAX, WMAX) + 0.9;
    // Orbit and magnify around the plot volume's own center, including its error-height axis.
    viewCenter.position.z = ZTOP / 2;
    uncenter.position.z = -ZTOP / 2;
    const zAxis = new LineField(3, { width: 1.5 });
    zAxis.push([XE, PY0, 0], [XE, PY0, ZTOP], C.white, 0.85);
    zAxis.push([XE, PY0, ZTOP], [XE - AH * 0.5, PY0, ZTOP - AH], C.white, 0.85);
    zAxis.push([XE, PY0, ZTOP], [XE + AH * 0.5, PY0, ZTOP - AH], C.white, 0.85);
    zAxis.commit();
    inner.add(zAxis.object);
    // dashed lines from the dots of slides 0.2 and 0.3 to the axes: a dot's place is the values of its two weights
    const DASH = 0.2;
    const drops = new LineField(360, { width: 1.1 });
    drops.count = 360;
    inner.add(drops.object);
    const dropsAt = (from: number, x: number, y: number, k: number) => {
      let n = from;
      for (let s = x - 0.3; s - DASH > PX0 && n < from + 60; s -= 2 * DASH) drops.seg(n++, [s, y, 0], [s - DASH, y, 0], C.white, k);
      for (let s = y - 0.3; s - DASH > PY0 && n < from + 120; s -= 2 * DASH) drops.seg(n++, [x, s, 0], [x, s - DASH, 0], C.white, k);
      while (n < from + 120) drops.seg(n++, [0, 0, 0], [0, 0, 0], C.white, 0);
    };
    // slide 0.2: the trace of the glide along the curve, dashed
    const TN = 34;
    const trace = new LineField(TN, { width: 1.8 });
    trace.count = TN;
    inner.add(trace.object);

    // ── the paths: one hairline per run, drawn as it flies; runs 1 and 2 get a heavier line ──
    const pz = new Float32Array(NR * (K + 1));
    for (let o = 0; o < NR * (K + 1); o++) pz[o] = hgt(pa[o], pb[o]);
    const trails = new LineField(NR * K);
    trails.count = NR * K;
    for (let r = 0; r < NR; r++)
      for (let k = 0; k < K; k++) {
        const o = r * (K + 1) + k;
        trails.seg(r * K + k, [px(pa[o]), py(pb[o]), pz[o]], [px(pa[o + 1]), py(pb[o + 1]), pz[o + 1]], C.white, 0);
      }
    trails.commit();
    inner.add(trails.object);
    const own = new LineField(2 * K, { width: 1.7 });
    own.count = 2 * K;
    inner.add(own.object);
    // the random starts of runs 1 and 2: small rings
    const rings = [R1, R2].map((r) => {
      const f = new LineField(32, { width: 1.3 });
      f.circle(startOf(r), 0.2, [1, 0, 0], [0, 1, 0], C.white, 0.9, 32);
      f.commit();
      inner.add(f.object);
      return f;
    });

    // ── the trained runs: one dot each ──
    const arrive = new Float32Array(NR);
    {
      let rank = 0;
      const rest = NR - 2;
      for (let r = 0; r < NR; r++) {
        if (r === R1 || r === R2) continue;
        // wide gaps first, then faster and faster
        arrive[r] = RAIN[0] + (RAIN[1] - FLIGHT - RAIN[0]) * Math.sqrt(rank / (rest - 1));
        rank++;
      }
    }
    const pts = new PointField({ count: NR, size: 0.11 });
    for (let r = 0; r < NR; r++) pts.setColor(r, C.white, r === R1 || r === R2 ? 1 : 0.5);
    pts.commit();
    inner.add(pts.object);
    // slide 0.2: the network's dot, and a lighter copy left at the first point when the dot moves on
    const pdots = new PointField({ count: 2, size: 0.11 });
    pdots.setColor(0, C.white, 1);
    pdots.setColor(1, C.white, 0.45);
    pdots.commit();
    inner.add(pdots.object);
    // ── 0.4: the valley floor and the spread along it ──
    const CN = 90;
    const aTop = FIT / 1.0, aEnd = Math.min(1.0, WMAX * 0.97); // from b = 1 down to a = 1
    const curvePt = (u: number): RGB => { const a = aTop + (aEnd - aTop) * u; return [px(a), py(FIT / a), 0]; };
    const curve = new LineField(CN, { width: 2.6 });
    curve.count = CN;
    inner.add(curve.object);
    // the double arrow along the curve, set off from it on the far side from the origin, over the middle SPAN of the end points
    const OFF = 0.85, AN = 60;
    const alongPt = (u: number): RGB => {
      const a = aLo + (aHi - aLo) * u, b = FIT / a;
      // normal of the curve on the page, pointing away from the origin
      const dxa = PW / WMAX, dyb = PH / WMAX; // page units per weight unit
      const tx = dxa, ty = (-FIT / (a * a)) * dyb, tl = Math.hypot(tx, ty);
      const nx = -ty / tl, ny = tx / tl;
      return [px(a) + nx * OFF, py(b) + ny * OFF, 0];
    };
    const along = new LineField(AN + 4, { width: 1.4 });
    along.count = AN + 4;
    inner.add(along.object);
    const alongMid = alongPt(0.5);
    const curveTop = curvePt(0);

    // ── 0.5 the triangle of the teams ──
    const prints = TEAM_ORDER.map((c) => new InkFill(teamTriangle(c).map(tri), C[PILLAR_HUE[c]], 0));
    prints.forEach((f) => group.add(f.object));
    const outline = new LineField(3, { width: 1.8 });
    CORNERS.forEach((c, i) => outline.push(tri(VERT[c]), tri(VERT[CORNERS[(i + 1) % 3]]), C.white, 0.85));
    outline.commit();
    group.add(outline.object);
    const spokes = new LineField(3, { width: 1.1 });
    CORNERS.forEach((c) => spokes.push(tri(VERT[c]), tri(CENTER), C.white, 0.5));
    spokes.commit();
    group.add(spokes.object);
    // the direction of the loop: a short arrow beside the middle of each side (shaft and head), drawn side by side
    const sides = LOOP.map(([a, b]) => {
      const p = tri(VERT[a]), q = tri(VERT[b]), dx = q[0] - p[0], dy = q[1] - p[1], l = Math.hypot(dx, dy);
      const m: RGB = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0], ml = Math.hypot(m[0], m[1]);
      return { m, u: [dx / l, dy / l] as [number, number], out: [m[0] / ml, m[1] / ml] as [number, number] };
    });
    const chevrons = new LineField(9, { width: 1.6 });
    for (let i = 0; i < 9; i++) chevrons.push([0, 0, 0], [0, 0, 0], C.white, 0);
    chevrons.commit();
    group.add(chevrons.object);
    /** A point beside the middle of side k, `d` world units outside the triangle. */
    const besideSide = (k: number, d: number): RGB => [sides[k].m[0] + sides[k].out[0] * d, sides[k].m[1] + sides[k].out[1] * d, 0];
    const CHEV_OUT = 0.36, SIDE_TEXT = 0.78; // how far outside the side its chevron and its sentence stand

    const num = (id: string) => String(ORDER.indexOf(id as (typeof ORDER)[number])).padStart(2, '0');
    const cornerHelp = (c: Corner) =>
      `<p>The team ${TEAM_ROLE[c]}.</p><p>` +
      TEAM_CHAPTERS[c].map((id) => `<b>${num(id)}</b> ${TITLES[id]}`).join('<br>') + '</p>';

    // label anchors on the network, at time t
    const midOf = (si: number, t: number): RGB => [netX(t) + (segs[si].ax + segs[si].bx) / 2, (segs[si].ay + segs[si].by) / 2, 0];
    const chainAt = (t: number, i: number): RGB => [netX(t) + CHX[i], CHY, 0];

    // per-run cache of what the paths last showed, so that a frame rewrites only the runs that changed
    const lastK = new Float32Array(NR).fill(-1), lastU = new Float32Array(NR).fill(-1);
    const fmt2 = (v: number) => v.toFixed(2);
    const NRS = NR.toLocaleString('en-US');

    const chapter: Chapter = {
      id: 'prologue',
      pillar: 'frame',
      kicker: 'Prologue',
      title: 'The program',
      source: 'Introducing PIRAMID · LessWrong, July 2026',
      links: [
        { text: 'Introducing PIRAMID', url: URL_INTRO },
        { text: 'Principles of Intelligence', url: 'https://princint.ai/' },
      ],
      duration: DUR,
      group,
      beats: [
        { at: T1, caption: 'Train two networks on the same examples and they can end up with different weights. Which differences matter for their predictions?' },
        { at: T2, caption: 'Isolate two weights in a chain. This small network computes b × a × x. Along the curve, an increase in a exactly offsets a decrease in b. Every input still gives the same output.' },
        { at: T3, caption: 'Both runs train on the same noisy examples. They start in different places and see the examples in different orders. Both approach the same curve.' },
        { at: T4, caption: 'From many random starts, training brings the weights close to the valley floor. Points on that curve make identical predictions. Small differences across it still change the predictions.' },
        { at: T5, caption: 'Here one number, a × b, tells us everything about the network’s predictions. PIRAMID asks whether statistical physics can help find quantities like this in much larger networks. Which properties of many weights, taken together, predict what a model will do?' },
        { at: T6, caption: 'Suppose a theory predicts that a network will learn a hierarchy. Researchers can generate examples from a known tree, train a network on them, then check whether an interpretability tool recovers the tree’s hidden variables.' },
        { at: T7, caption: 'In this presentation, a linear probe uses activations from an untrained network to predict hidden variables in percolation data. The next chapters ask four questions about what networks learn and use. PIRAMID sees that understanding as a foundation for scalable AI alignment.' },
      ],
      shots: [
        { t: 0, pos: [0, 0, 30], look: [0, 0, 0], fov: 38 },
        { t: DUR, pos: [0, 0, 30], look: [0, 0, 0], fov: 38 },
      ],
      interaction: {
        pick(t, p) {
          if (!surfaceStage(t)) return null;
          pose(t);
          // Project this plot's volume, not the whole chapter or the untransformed page plane.
          const points: [number, number][] = [];
          for (const x of [PX0, XE]) for (const y of [PY0, YE]) for (const z of [0, ZTOP]) {
            const w = W([x, y, z]), q = p.project(w[0], w[1], w[2]);
            if (q[2] >= -1 && q[2] <= 1) points.push([q[0], q[1]]);
          }
          if (points.length < 3) return null;
          points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
          const cross = (a: number[], b: number[], c: number[]) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
          const half = (ps: [number, number][]) => {
            const out: [number, number][] = [];
            for (const q of ps) {
              while (out.length > 1 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
              out.push(q);
            }
            return out.slice(0, -1);
          };
          const hull = [...half(points), ...half([...points].reverse())];
          const q = [p.x, p.y];
          return hull.length >= 3 && hull.every((a, i) => cross(a, hull[(i + 1) % hull.length], q) >= -0.5) ? 'surface' : null;
        },
        draggable: (region) => region === 'surface',
        drag(t, region, dx, dy) {
          if (!surfaceStage(t) || region !== 'surface' || (dx === 0 && dy === 0)) return;
          surfaceEngaged = true;
          surfaceYaw = ((surfaceYaw + dx * 0.006 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
          // Keep the view above the parameter plane, between 15 and 75 degrees of elevation.
          surfacePitch = clamp(surfacePitch + dy * 0.006, THETA - 75 * DEG, THETA - 15 * DEG);
        },
        wheel(t, region, delta) {
          if (!surfaceStage(t) || region !== 'surface' || delta === 0) return;
          surfaceEngaged = true;
          surfaceZoom = clamp(surfaceZoom * Math.exp(-delta * 0.0012), 0.65, 1.8);
        },
        reset: resetSurface,
        changed: () => surfaceEngaged,
      },
      update(t, api) {
        const leave = 1 - smooth(LEAVE[0], LEAVE[1], t);
        const m3 = pose(t);
        // Run labels can still be fading during the first surface gesture. Bring those anchors
        // through the same inspected transform while preserving their authored positions at rest.
        const plotLabel = (p: RGB): RGB => surfaceEngaged && surfaceStage(t) ? W(p) : p;
        const NX = netX(t);
        // ── the network, and the copy of it that folds into its dot ──
        const drawing = t < DRAW[1] + 0.1;
        const netOn = 1 - smooth(NET_OFF[0], NET_OFF[1], t);
        const ab = abAt(t, tmp);
        // the fold under way: into the first point of the curve (0.2), into run 1's start, into run 2's start (0.3)
        const F = t >= RESET[0] ? FLY2 : t >= REWIND[0] ? FLY1 : FOLD0;
        const dot: RGB = t >= RESET[0] ? startOf(R2) : t >= REWIND[0] ? startOf(R1) : onSurf(PA1, PB1);
        const fold = clamp((t - F[0]) / (F[1] - F[0]));
        for (let i = 0; i < segs.length; i++) {
          const q = segs[i], j = i - firstWeight;
          let k = q.ring ? 0.95 : dark(j === A ? ab[0] : j === B ? ab[1] : otherAt(j, t));
          // while it draws itself, the rings come in layer by layer and the lines follow
          if (drawing) k *= q.ring ? smooth(DRAW[0] + (1 - q.d) * 0.8, DRAW[0] + (1 - q.d) * 0.8 + 0.5, t) : smooth(DRAW[0] + 0.8, DRAW[1], t);
          else k *= netOn * (1 - 0.78 * smooth(CHAIN[0], CHAIN[1], t));
          net.seg(i, [NX + q.ax, q.ay, 0], [NX + q.bx, q.by, 0], C.white, k);
          // the copy leaves the network, shrinks on the way and ends as the dot; its right side goes a moment ahead
          const e = fold > 0 && fold < 1 ? easeInOut(clamp(fold * 1.12 - q.d * 0.12)) : 1;
          if (e >= 0.999) fly.seg(i, [0, 0, 0], [0, 0, 0], C.white, 0);
          else {
            const cx = NX + (dot[0] - NX) * e, cy = dot[1] * e, sc = 1 - e;
            fly.seg(i, [cx + q.ax * sc, cy + q.ay * sc, 0], [cx + q.bx * sc, cy + q.by * sc, 0], C.white, k * smooth(0, 0.12, fold));
          }
        }
        net.commit();
        fly.commit();
        const heavyOn = smooth(AXES[0], AXES[1], t) * netOn;
        [segA, segB].forEach((si, n) => {
          const q = segs[si];
          heavy.seg(n, [NX + q.ax, q.ay, 0], [NX + q.bx, q.by, 0], C.white, dark(ab[n]) * heavyOn);
        });
        heavy.commit();
        // the chain: the same two weights, as dark as in the network
        const chainOn = smooth(CHAIN[0], CHAIN[1], t) * netOn;
        for (let n = 0; n < 3; n++) {
          const c = chainAt(t, n);
          for (let i = 0; i < RSEG; i++) {
            const a0 = (i / RSEG) * Math.PI * 2, a1 = ((i + 1) / RSEG) * Math.PI * 2;
            chainRings.seg(n * RSEG + i, [c[0] + RING * Math.cos(a0), c[1] + RING * Math.sin(a0), 0], [c[0] + RING * Math.cos(a1), c[1] + RING * Math.sin(a1), 0], C.white, 0.95 * chainOn);
          }
        }
        for (let n = 0; n < 2; n++) {
          const p = chainAt(t, n), q = chainAt(t, n + 1), x0 = p[0] + RING + 0.08, x1 = q[0] - RING - 0.12, HL = 0.3;
          chainLines.seg(n, [x0, CHY, 0], [x1, CHY, 0], C.white, dark(ab[n]) * chainOn);
          chainHeads.seg(2 * n, [x1, CHY, 0], [x1 - HL, CHY + HL * 0.55, 0], C.white, 0.9 * chainOn);
          chainHeads.seg(2 * n + 1, [x1, CHY, 0], [x1 - HL, CHY - HL * 0.55, 0], C.white, 0.9 * chainOn);
        }
        chainRings.commit(); chainLines.commit(); chainHeads.commit();

        // ── the plot ──
        const axesOn = smooth(AXES[0], AXES[1], t) * leave;
        axes.uIntensity.value = axesOn;
        zAxis.uIntensity.value = m3 * leave;
        // the surface rises with the tilt. Seen from above in 0.4, only the faint contours remain.
        surf.uIntensity.value = m3 * leave;
        grid.uIntensity.value = 0.32 * m3 * leave;
        contours.uIntensity.value = (0.3 * m3 + 0.13 * smooth(UNTILT[0], UNTILT[1], t)) * leave;
        // slide 0.2: the network's dot, its slide along the curve, and the copy left behind
        const pOn = smooth(FOLD0[1] - 0.15, FOLD0[1] + 0.1, t) * (1 - smooth(REWIND[0] - 0.1, REWIND[0] + 0.4, t));
        const gl = ease(GLIDE, t);
        const pg = glideAt(gl, tmpP), xp = px(pg[0]), yp = py(pg[1]);
        pdots.setPos(0, xp, yp, 0);
        pdots.sizes[0] = pOn * BIG;
        pdots.setPos(1, px(PA1), py(PB1), 0);
        pdots.sizes[1] = pOn * BIG * smooth(GLIDE[0], GLIDE[0] + 0.3, t);
        pdots.commit();
        for (let i = 0; i < TN; i++) {
          // dashes from the first point to the dot
          const u0 = (i / TN) * gl, u1 = ((i + 0.6) / TN) * gl;
          const q0 = glideAt(u0, tmpG), x0 = px(q0[0]), y0 = py(q0[1]);
          const q1 = glideAt(u1, tmpG);
          trace.seg(i, [x0, y0, 0], [px(q1[0]), py(q1[1]), 0], C.cyan, gl > 0.001 ? 1.0 * pOn : 0);
        }
        trace.commit();
        // runs 1 and 2: how far along its path each one is
        const u1 = t < FLY1[1] ? 0 : ease(RUN1, t), u2 = t < FLY2[1] ? 0 : ease(RUN2, t);
        const on1 = smooth(FLY1[1] - 0.15, FLY1[1] + 0.1, t), on2 = smooth(FLY2[1] - 0.15, FLY2[1] + 0.1, t);
        const p1 = pathAt(R1, u1, tmp1), x1 = px(p1[0]), y1 = py(p1[1]), z1 = hgt(p1[0], p1[1]);
        const p2 = pathAt(R2, u2, tmpR2), x2 = px(p2[0]), y2 = py(p2[1]), z2 = hgt(p2[0], p2[1]);
        dropsAt(0, x1, y1, 0.55 * on1 * netOn);
        dropsAt(120, x2, y2, 0.55 * on2 * netOn);
        dropsAt(240, xp, yp, 0.55 * pOn);
        drops.commit();
        // the heavy paths of runs 1 and 2: drawn as far as the run has come; they step back while the others arrive
        const ownK = 0.95 * (1 - smooth(RAIN[0], RAIN[0] + 1.2, t));
        [[R1, u1, on1], [R2, u2, on2]].forEach(([r, u, on], n) => {
          const head = u * K;
          for (let k = 0; k < K; k++) {
            const o = r * (K + 1) + k, f = clamp(head - k);
            if (f <= 0 || on <= 0) { own.seg(n * K + k, [0, 0, 0], [0, 0, 0], C.white, 0); continue; }
            const ax = px(pa[o]), ay = py(pb[o]), az = pz[o];
            own.seg(n * K + k, [ax, ay, az], [ax + (px(pa[o + 1]) - ax) * f, ay + (py(pb[o + 1]) - ay) * f, az + (pz[o + 1] - az) * f], C.white, ownK * on);
          }
        });
        own.commit();
        rings[0].uIntensity.value = on1 * (1 - smooth(TILT[0], TILT[0] + 0.8, t));
        rings[1].uIntensity.value = on2 * (1 - smooth(TILT[0], TILT[0] + 0.8, t));

        // the other runs: each flies along its path in FLIGHT seconds and leaves its path behind, which then fades
        const FLOOR = 0.045 * (1 - 0.45 * smooth(DIM[0], DIM[1], t));
        let dirty = false;
        for (let r = 0; r < NR; r++) {
          if (r === R1 || r === R2) continue;
          const f = clamp((t - arrive[r]) / FLIGHT);
          const u = f <= 0 ? 0 : easeInOut(f);
          const kk = f <= 0 ? 0 : FLOOR + (0.5 - FLOOR) * (1 - smooth(arrive[r] + FLIGHT, arrive[r] + FLIGHT + FADE, t));
          if (Math.abs(kk - lastK[r]) < 2e-3 && Math.abs(u - lastU[r]) < 1e-4) continue;
          const head = u * K;
          for (let k = 0; k < K; k++) {
            const o = r * (K + 1) + k, fk = clamp(head - k), i = r * K + k;
            // the newest stretch of a path is the darkest
            const kseg = fk <= 0 ? 0 : kk * (0.45 + 0.55 * (k + 1) / K);
            const ax = px(pa[o]), ay = py(pb[o]), az = pz[o];
            if (fk < 1 && fk > 0) trails.seg(i, [ax, ay, az], [ax + (px(pa[o + 1]) - ax) * fk, ay + (py(pb[o + 1]) - ay) * fk, az + (pz[o + 1] - az) * fk], C.white, kseg);
            else if (lastU[r] !== u) trails.seg(i, [ax, ay, az], [px(pa[o + 1]), py(pb[o + 1]), pz[o + 1]], C.white, kseg);
            else { const c = trails.colors, jj = i * 6; for (let m = 0; m < 6; m++) c[jj + m] = C.white[m % 3] * kseg; }
          }
          lastK[r] = kk; lastU[r] = u;
          dirty = true;
        }
        trails.uIntensity.value = leave;
        if (dirty) trails.commit(); // NR × K segments: upload them only when a path changed

        // ── the dots ──
        const big = 1 + (BIG - 1) * (1 - smooth(RAIN[0], RAIN[0] + 1.5, t));
        for (let r = 0; r < NR; r++) {
          let size: number;
          if (r === R1) { pts.setPos(r, x1, y1, z1); size = on1 * big; }
          else if (r === R2) { pts.setPos(r, x2, y2, z2); size = on2 * big; }
          else {
            const f = clamp((t - arrive[r]) / FLIGHT);
            if (f <= 0) { pts.sizes[r] = 0; continue; }
            const e = easeInOut(f);
            pathAt(r, e, tmp2);
            pts.setPos(r, px(tmp2[0]), py(tmp2[1]), hgt(tmp2[0], tmp2[1]));
            // a dot in flight is a network still training: drawn smaller; it reaches full size where it stops
            size = 0.7 + 0.3 * smooth(0.85, 1, f);
          }
          pts.sizes[r] = size;
        }
        pts.uOpacity.value = leave * (1 - 0.5 * smooth(DIM[0], DIM[1], t)); // the dots step back so that the valley floor shows through them
        pts.commit();

        // ── 0.4 the valley floor and the spread along it ──
        const cu = ease(CURVE, t);
        for (let i = 0; i < CN; i++) {
          const u0 = i / CN, u1c = (i + 1) / CN;
          if (u0 >= cu) { curve.seg(i, [0, 0, 0], [0, 0, 0], C.white, 0); continue; }
          curve.seg(i, curvePt(u0), curvePt(Math.min(u1c, cu)), C.cyan, 1.3);
        }
        curve.uIntensity.value = leave;
        curve.commit();
        const al = ease(ALONG, t);
        for (let i = 0; i < AN; i++) {
          // the arrow grows from its middle to both ends
          const u0 = i / AN, u1a = (i + 1) / AN, lo = 0.5 - al / 2, hi = 0.5 + al / 2;
          const a0 = Math.max(u0, lo), a1 = Math.min(u1a, hi);
          if (a1 <= a0) { along.seg(i, [0, 0, 0], [0, 0, 0], C.white, 0); continue; }
          along.seg(i, alongPt(a0), alongPt(a1), C.white, 0.85);
        }
        // arrowheads, once the arrow has reached its ends
        const headK = 0.85 * smooth(0.92, 1, al);
        [[0, 0.03], [1, 0.97]].forEach(([u, v], n) => {
          const tip = alongPt(u), back = alongPt(v);
          const dx = tip[0] - back[0], dy = tip[1] - back[1], l = Math.hypot(dx, dy) || 1, ux = dx / l, uy = dy / l, HL = 0.32;
          along.seg(AN + 2 * n, tip, [tip[0] - ux * HL - uy * HL * 0.55, tip[1] - uy * HL + ux * HL * 0.55, 0], C.white, headK);
          along.seg(AN + 2 * n + 1, tip, [tip[0] - ux * HL + uy * HL * 0.55, tip[1] - uy * HL - ux * HL * 0.55, 0], C.white, headK);
        });
        along.uIntensity.value = leave;
        along.commit();

        // ── the triangle (it leaves when the chain of 0.6 comes in) ──
        const keep6 = 1 - smooth(LEAVE6[0], LEAVE6[1], t);
        outline.uIntensity.value = smooth(OUTLINE[0], OUTLINE[1], t) * keep6;
        spokes.uIntensity.value = smooth(OUTLINE[0] + 0.3, OUTLINE[1] + 0.3, t) * keep6;
        const on = (k: number) => smooth(CORNER0 + k * CORNER_GAP, CORNER0 + k * CORNER_GAP + 0.7, t) * keep6;
        prints.forEach((f, k) => { f.uInk.value = INK * on(k); });
        // the sides of the loop, one after another: a chevron in the direction the loop runs
        const sideOn = (k: number) => smooth(SIDE0 + k * SIDE_GAP, SIDE0 + k * SIDE_GAP + 0.6, t) * keep6;
        sides.forEach(({ u }, k) => {
          // on the paper just outside the side, where the team prints do not hide it
          const c = besideSide(k, CHEV_OUT), L = 0.55, H = 0.2, kk = 0.95 * sideOn(k);
          const tip: RGB = [c[0] + u[0] * L, c[1] + u[1] * L, 0], tail: RGB = [c[0] - u[0] * L, c[1] - u[1] * L, 0];
          chevrons.seg(3 * k, tail, tip, C.white, kk);
          chevrons.seg(3 * k + 1, tip, [tip[0] - u[0] * H - u[1] * H * 0.6, tip[1] - u[1] * H + u[0] * H * 0.6, 0], C.white, kk);
          chevrons.seg(3 * k + 2, tip, [tip[0] - u[0] * H + u[1] * H * 0.6, tip[1] - u[1] * H - u[0] * H * 0.6, 0], C.white, kk);
        });
        chevrons.commit();

        // ── labels ──
        const a1 = smooth(DRAW[1] - 0.6, DRAW[1], t) * (1 - smooth(T2 - 0.2, T2 + 0.2, t));
        const illustrative = Math.max(a1, heavyOn);
        if (illustrative > 0.01)
          api.label('network-origin', [netX(t), nodes[1][0][1] + RING, 0], 'Illustration', { alpha: illustrative, align: 'center', dy: -12 - 22 * smooth(AXES[0], AXES[1], t) });
        if (a1 > 0.01) {
          api.label('neuron', at(t, nodes[0][0], 0, RING), 'neuron', {
            alpha: a1, align: 'center', size: 'md', color: CSS.white,
          });
          api.label('weight', midOf(segW, t), 'weight', { alpha: a1, align: 'below', size: 'md', color: CSS.white, dy: 24, help: `<p>${THEME === 'dark' ? 'Brighter' : 'Darker'} lines denote larger weights in absolute value. The drawing illustrates how weights can change during training.</p>` });
        }
        if (heavyOn > 0.01) {
          api.label('wa', midOf(segA, t), 'weight a', { alpha: heavyOn, align: 'right', size: 'md', color: CSS.white, dx: -10, dy: -10 });
          api.label('wb', midOf(segB, t), 'weight b', { alpha: heavyOn, align: 'center', size: 'md', color: CSS.white, dy: -12 });
        }
        // the chain: what each neuron holds, and the two weights with their values
        if (chainOn > 0.01) {
          const c0 = chainAt(t, 0), c1 = chainAt(t, 1), c2 = chainAt(t, 2);
          const mid = (p: RGB, q: RGB): RGB => [(p[0] + q[0]) / 2, CHY, 0];
          // Keep the title within the chain's footprint: outside the input node it falls beyond the base frame,
          // so the measured framing box cannot include it. The run status has its own line below the title.
          api.label('ch-title', c1, 'linear model', { alpha: chainOn, align: 'below', size: 'sm', color: INK2, dy: 46, help: '<p>Each arrow multiplies its input by the weight on that arrow. The model computes only this chain, and the larger network above it shows where the two weights could sit.</p>' });
          api.label('ch-in', c0, 'input x', { alpha: chainOn, align: 'below', size: 'sm', color: CSS.white, dy: 18 });
          api.label('ch-mid', c1, 'a × x', { alpha: chainOn, align: 'below', size: 'sm', color: CSS.white, dy: 18 });
          api.label('ch-out', c2, `output b × a × x<small>≈ ${fmt2(ab[0] * ab[1])} × x</small>`, { alpha: chainOn, align: 'below', size: 'sm', color: CSS.white, dy: 18 });
          api.label('ch-a', mid(c0, c1), `× a <small>${fmt2(ab[0])}</small>`, { alpha: chainOn, align: 'center', size: 'md', color: CSS.white, dy: -10 });
          api.label('ch-b', mid(c1, c2), `× b <small>${fmt2(ab[1])}</small>`, { alpha: chainOn, align: 'center', size: 'md', color: CSS.white, dy: -10 });
        }
        // which run the network on the left is showing (0.3)
        const states: [string, number, number][] = [
          ['run 1: random start', REWIND[0], RUN1[0]],
          ['run 1: training', RUN1[0], RUN1[1]],
          ['run 1: trained', RUN1[1], RESET[0]],
          ['run 2: new random start', RESET[0], RUN2[0]],
          ['run 2: training', RUN2[0], RUN2[1]],
          ['run 2: trained', RUN2[1], NET_OFF[1]],
        ];
        states.forEach(([text, a, b], n) => {
          const al2 = smooth(a, a + 0.25, t) * (n === states.length - 1 ? netOn : 1 - smooth(b - 0.05, b + 0.2, t));
          if (al2 > 0.01) api.label(`state${n}`, chainAt(t, 1), text, { alpha: al2, align: 'below', size: 'sm', color: INK2, dy: 70 });
        });
        const flat = 1 - m3;
        if (axesOn > 0.01) {
          api.label('axis-a', W([XE, PY0, 0]), 'weight a', { alpha: axesOn, align: 'left', size: 'md', color: CSS.white, dx: 10, dy: flat > 0.5 ? 0 : 12, help: flat > 0.5 ? '<p>Each dot represents a linear model. Its position gives its pair of weights a and b.</p>' : undefined });
          // seen tilted, the label stands left of the arrow's tip, clear of the surface's edge
          api.label('axis-b', W([PX0, YE, 0]), 'weight b', { alpha: axesOn, align: flat > 0.5 ? 'center' : 'right', size: 'md', color: CSS.white, dx: flat > 0.5 ? 0 : -12, dy: flat > 0.5 ? -4 : 4 });
          api.label('axis-0', W([PX0, PY0, 0]), '0', { alpha: axesOn * flat, align: 'right', size: 'xs', color: INK2, dx: -6, dy: 8 });
        }
        if (m3 * leave > 0.01) {
          const controls = surfaceStage(t) ? '<p>Drag to turn the surface. Scroll over it to zoom.</p>' : '';
          api.label('axis-z', W([XE, PY0, ZTOP]), 'error above minimum', { alpha: m3 * leave, align: 'center', size: 'md', color: CSS.white, dy: -6, help: `<p>Height is root mean square prediction error on the training data minus its minimum. The paths show ${NRS} runs on the same ${NDATA} noisy examples. Each line smooths the early updates and connects them to the final weights.</p>${controls}` });
        }
        const computed = smooth(RUN1[0], RUN1[0] + 0.5, t) * leave;
        if (computed * flat > 0.01) api.label('runs-origin', W([XE, YE, 0]), 'Computed for this presentation', { alpha: computed * flat, align: 'center', dy: -10 });
        if (computed * m3 > 0.01) api.label('surface-origin', W([XE, PY0, ZTOP]), 'Computed for this presentation', { alpha: computed * m3, align: 'center', dy: -34 });
        // slide 0.2: the dot's two values on the axes, and the two points with their products
        if (pOn > 0.01) {
          api.label('va', [xp, PY0, 0], fmt2(pg[0]), { alpha: pOn, align: 'below', size: 'xs', color: INK2, dy: 6 });
          api.label('vb', [PX0, yp, 0], fmt2(pg[1]), { alpha: pOn, align: 'right', size: 'xs', color: INK2, dx: -8 });
          const k1 = pOn * smooth(GLIDE[0] + 0.2, GLIDE[0] + 0.6, t), k2 = pOn * smooth(GLIDE[1] - 0.1, GLIDE[1] + 0.3, t);
          if (k1 > 0.01) api.label('p1', [px(PA1), py(PB1), 0], `${fmt2(PA1)} × ${fmt2(PB1)} ≈ ${fmt2(FIT)}`, { alpha: k1, align: 'left', size: 'sm', color: CSS.cyan, dx: 14, plate: true });
          if (k2 > 0.01) api.label('p2', [px(PA2), py(PB2), 0], `${fmt2(PA2)} × ${fmt2(PB2)} ≈ ${fmt2(FIT)}`, { alpha: k2, align: 'center', size: 'sm', color: CSS.cyan, dy: -12, plate: true });
        }
        // runs 1 and 2: where each one started and where it ended (slide 0.3)
        const gone = 1 - smooth(T4 + 0.1, T4 + 0.6, t);
        const s1 = on1 * gone, s2 = on2 * gone;
        if (s1 > 0.01) api.label('start1', plotLabel(startOf(R1)), 'random start', { alpha: s1, align: 'below', size: 'xs', color: INK2, dx: 16, dy: 10 });
        if (s2 > 0.01) api.label('start2', plotLabel(startOf(R2)), 'random start', { alpha: s2, align: 'right', size: 'xs', color: INK2, dx: -12 });
        const r1 = smooth(RUN1[1] - 0.2, RUN1[1] + 0.3, t) * gone, r2 = smooth(RUN2[1] - 0.2, RUN2[1] + 0.3, t) * gone;
        if (r1 > 0.01) api.label('run1', plotLabel(endOf(R1)), 'run 1', { alpha: r1, align: 'left', size: 'lg', color: CSS.white, dx: 12, dy: -12 });
        if (r2 > 0.01) api.label('run2', plotLabel(endOf(R2)), 'run 2', { alpha: r2, align: 'left', size: 'lg', color: CSS.white, dx: 12, dy: -12 });
        const pth = smooth(RUN2[1], RUN2[1] + 0.5, t) * gone;
        if (pth > 0.01) { pathAt(R2, 0.5, tmp2); api.label('path', plotLabel([px(tmp2[0]), py(tmp2[1]), 0]), 'smoothed training path', { alpha: pth, align: 'left', size: 'xs', color: INK2, dx: 12 }); }
        const sh = smooth(CURVE[1] - 0.3, CURVE[1] + 0.4, t) * leave;
        if (sh > 0.01) api.label('shared', W(curveTop), 'Best-fit curve', { alpha: sh, align: 'left', size: 'md', color: CSS.cyan, dx: 12, plate: true, help: `<p>On the curve, a × b equals the best-fit slope, about ${fmt2(FIT)}. All points on it give the same output for a given input x.</p>` });
        const sp = smooth(ALONG[1] - 0.3, ALONG[1] + 0.4, t) * leave;
        if (sp > 0.01) api.label('spread', W(alongMid), 'Typical final weights', { alpha: sp, align: 'left', size: 'md', color: CSS.white, dx: 10, plate: true, help: `<p>The arrow spans the middle ${SPAN_PCT} of final values of weight a, from ${fmt2(aLo)} to ${fmt2(aHi)}, projected onto the best-fit curve.</p>` });
        if (t > CORNER0 - 0.2) {
          TEAM_ORDER.forEach((c, k) => {
            const a = on(k);
            if (a < 0.01) return;
            const p = tri(VERT[c]), col = CSS[PILLAR_HUE[c]], txt = `<span class="caps">${TEAM_LABEL[c]}</span>`;
            const opts = { alpha: a, size: 'md', color: col, help: cornerHelp(c) } as const;
            if (c === 'data') api.label('c-data', p, txt, { ...opts, align: 'center', dy: -4 });
            else if (c === 'theory') api.label('c-theory', p, txt, { ...opts, align: 'left', dx: 8 });
            else api.label('c-tools', p, txt, { ...opts, align: 'right', dx: -8 });
          });
          // the center belongs to all three teams: coarse-graining stands there
          const mix = on(2) * smooth(CORNER0 + 2 * CORNER_GAP + 0.4, CORNER0 + 2 * CORNER_GAP + 1.0, t);
          if (mix > 0.01) {
            api.label('c-mid', tri(CENTER), TITLES.scales, { alpha: mix, align: 'middle', size: 'md', color: CSS.white, plate: true, help: '<p>Coarse-graining describes a system at lower resolution. How much does a prediction change when fine details are omitted?</p>' });
          }
        }
        // what passes along each side, beside its chevron, and the way back under the bottom side
        if (t > SIDE0 - 0.2) {
          const s0 = sideOn(0), s1 = sideOn(1), s2 = sideOn(2), back = smooth(BACK, BACK + 0.6, t) * keep6;
          if (s0 > 0.01) api.label('e1', besideSide(0, SIDE_TEXT - 0.15), LOOP_SIDE.theoryTools, { alpha: s0, align: 'below', size: 'md', color: CSS.white });
          if (s1 > 0.01) api.label('e2', besideSide(1, SIDE_TEXT), LOOP_SIDE.toolsData, { alpha: s1, align: 'right', size: 'md', color: CSS.white });
          if (s2 > 0.01) api.label('e3', besideSide(2, SIDE_TEXT), LOOP_SIDE.dataTheory, { alpha: s2, align: 'left', size: 'md', color: CSS.white });
          if (back > 0.01) api.label('e4', besideSide(0, SIDE_TEXT - 0.15), LOOP_SIDE.back, { alpha: back, align: 'below', size: 'md', color: CSS.red, dy: 26 });
        }
        if (t > STEP0 - 0.2) {
          QUESTIONS.forEach(([test, question], k) => {
            const start = STEP0 + k * STEP_GAP;
            const alpha = smooth(start, start + 0.35, t);
            if (alpha <= 0.01) return;
            const at: RGB = [CH_X, chainY(k), 0];
            api.label(`test${k}`, at, `<span class="caps">${test}</span>`, { alpha, align: 'right', size: 'md', color: INK2, dx: -12 });
            api.label(`st${k}`, at, question, { alpha, align: 'left', size: 'md', color: CSS.white, dx: 12 });
          });
        }
      },

      // Notes add context that is not already stated in the caption or figure.
      inspect(t) {
        const hs: Hotspot[] = [];
        pose(t);
        if (t < T2)
          hs.push({
            id: 'network',
            pos: at(t, nodes[3][0], 0, RING),
            dy: -18,
            title: 'The computed experiment',
            body:
              '<p>The larger network drawing is illustrative. The experiment computes only the two weights a and b; the changing darkness of the other lines does not show measured training.</p>',
          });
        else if (t < T3)
          hs.push({
            id: 'axes',
            pos: W(onSurf(PA2, PB2)),
            dx: 18,
            dy: 16,
            title: 'An exact symmetry',
            body:
              '<p>Multiplying a by any nonzero number c and dividing b by c leaves their product unchanged.</p>' +
              '<p>The larger network is illustrative. Its highlighted pair shows where the two weights sit in sequence; the plotted experiment contains only those two weights.</p>',
          });
        else if (t < T4)
          // slide 0.3: the two runs
          hs.push({
            id: 'runs',
            pos: W(endOf(R1)),
            dx: 18,
            dy: 14,
            title: 'How the two runs were chosen',
            body:
              `<p>The two runs use the same ${NDATA} noisy examples. They were selected for long training paths and distant final weights. The lines smooth the early updates and connect them to the final weights.</p>`,
          });
        else if (t < T5)
          // slide 0.3b: the surface and the cloud
          hs.push({
            id: 'surface',
            pos: W(onSurf(0.95, 0.9)),
            dx: 14,
            title: 'The valley floor',
            body:
              `<p>Along the valley floor, a × b equals the slope that best fits the ${NDATA} noisy examples, about ${fmt2(FIT)}. Surface height represents root-mean-square prediction error above its minimum, so zero height does not mean perfect predictions.</p>` +
              `<p>All ${NRS} runs start with positive weights, so the plot shows the positive branch of the best-fit curve.</p>`,
          });
        else if (t < T6)
          hs.push({
            id: 'shared',
            pos: W(curvePt(0.12)),
            dx: -34,
            title: 'Why starting weights matter',
            body:
              `<p>${SPAN_PCT} of the runs end with weight a between ${fmt2(aLo)} and ${fmt2(aHi)}, the span of the arrow.</p>` +
              '<p>During training, a² − b² stays almost fixed in this network. The starting weights therefore largely determine where a run ends along the curve. Random example order adds a smaller spread across it.</p>',
          });
        else if (t < T7) {
          if (t > CORNER0 + 2 * CORNER_GAP)
            hs.push({
              id: 'teams',
              pos: tri(CENTER),
              dy: -40,
              title: 'What a failed test can reveal',
              body:
                '<p>Introducing PIRAMID imagines a tool finding only distributed or nonlinear patterns where researchers expected clean hierarchical features. They could then reexamine the theory, the tool, or the data model.</p>' +
                '<p>PIRAMID treats its hypotheses about scale as guides for research across all three teams.</p>',
            });
        } else if (t > STEP0 + 0.4) {
          hs.push({
            id: 'questions',
            pos: [CH_X, chainY(QUESTIONS.length - 1) - 0.7, 0],
            dx: 14,
            title: 'What the comparisons establish',
            body:
              '<p>Brill and Carlson fit separate probes to predict known hidden variables from raw inputs and from trained activations. In this presentation, we also fit probes to the untrained network’s activations. Comparing held-out errors on the same targets tests whether network training made the variables easier to predict.</p>' +
              '<p>Removing a feature can test whether an output depends on it. In the parity transformer, selected features determine the direction passed to each feedforward block. The decoder rescales its output to the norm of that bottleneck’s input. Testing whether an individual feature is necessary requires a separate intervention.</p>',
          });
        }
        return hs;
      },
    };
    return chapter;
  },
};

export default prologue;
