import * as THREE from 'three/webgpu';
import { attribute, uniform } from 'three/tsl';
import type { Beat, Chapter, ChapterFactory, Control, FrameApi, Hotspot, NoteDetails, Probe, ProbeInput, Readout, Result, Shot } from '../engine/types';
import { BEFORE_TRAINING_KIND } from '../engine/cards';
import { resolveParams } from '../engine/params';
import { C, CSS, THEME, printed, scaleRGB, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { mulberry32, smooth, clamp, lerp } from '../lib/math';
import { generate, clusterTree, familyTable, sizeCCDF, fitYuleSimon, yuleSimonCCDF, demoCut, slowMerge, markedPoint, heroCut } from './percolation.gen';

/**
 * CRITICAL PERCOLATION — Brill & Carlson, arXiv:2606.20347. Eleven slides, one idea each.
 *
 * Slides 1 to 7, the data, computed live by build() with the paper's generator (percolation.gen.ts):
 *   1 the problem: a small tree of real concepts, drawn flat;
 *   2 the dataset (every cluster) and its cluster sizes;
 *   3 a six-point cluster built merge by merge beside a larger example’s completed latent tree;
 *   4 the same tree hung by depth and cut into subclusters (subcluster A);
 *   4B the largest cluster of the dataset cut the same way, with the small tree kept as an inset;
 *   5 the ring's points at their inputs x, beside their latent tree;
 *   5B the target y of point P, and every point colored by its own y.
 *
 * Slides 8 to 11 (`S6` on), the paper's test rerun in advance at the paper's scale: one self-contained block (`rerun` in
 * build) that reads public/data/percolation-offline.json, written by the offline study (offline/percolation), and owns
 * its figures, beats, notes and the chapter's comparison with the network before training. Slide 9 draws the probe
 * on the network and P's cluster colored by the probe's target; slide 11 draws the probe errors at every place.
 */

const DIM = 32;
const TAU = Math.PI * 2;
/** Categorical colors for groups and subclusters. The depth-cut palette treats entries i and i + 3 as one color family and gives siblings different families. */
const HUES: RGB[] = [scaleRGB(C.violet, 1.1), scaleRGB(C.cyan, 1), scaleRGB(C.white, 0.9), scaleRGB(C.magenta, 1), scaleRGB(C.cyan, 0.45), scaleRGB(C.violet, 0.42)];
const INK = C.white;
/** Key ink thinned to the lightness of the team ink: the negative sign, so that neither sign looks heavier. */
const NEG: RGB = scaleRGB(C.white, 0.43);
/** Dense clouds need less light on the dark theme than ink on paper. */
const TK = THEME === 'paper' ? 1 : 0.55;

// ── timeline (seconds, local) ──────────────────────────────────────────────────────────────────
// Where the slides begin. A slide's own length covers its animation; in film mode the engine holds its last
// frame until the text has had its reading time. Each slide ends on the state its caption describes: S4B is the
// second slide on the latent tree (the largest cluster), S5B the second slide on the data (the target y).
const S1 = 0.6, S2 = 4.5, S3 = 12, S4 = 30, S4B = S4 + 7.5, S5 = S4B + 8.5, S5B = S5 + 5.4, S6 = S5B + 7.0;
// 1 the tree of concepts
const T_TREE0 = 0.3; // its words and lines appear from the root down
// 2 the dataset grows from the center of the ball outwards, then its cluster sizes are drawn
const T_GROW0 = S2 + 0.3, T_GROW1 = S2 + 3.0, T_CC0 = S2 + 2.8;
// 3 the ring: its points appear, the merges before the slow one run quickly, the slow merge is named step by step
// (four callouts, each up long enough to read), then the merge happens and the rest run
const T_RING0 = S3 + 0.6;
// The construction slide uses a small, separate coalescent run while retaining the full tree beside it.
const teachingStage = (t: number) => t >= S3 && t < S4;
const TEACH_TREE_SHIFT = 340;
const teachingTreeShift = (t: number) => TEACH_TREE_SHIFT * (1 - smooth(S4, S4 + 1, t));
const T_PRE0 = S3 + 2.4, T_PRE1 = S3 + 4.8;
const T_SLOWM = S3 + 10.6, SLOW_DUR = 1.9;
const T_POST0 = T_SLOWM + 3.0, T_MERGE1 = S4 - 1.2;
const T_HA = S4 + 0.1, T_PICK = S4 + 0.6; // identify A and P when the larger example becomes the focus
// 4 the tree hung by depth and cut; then (4B) the largest cluster
const T_HANG0 = S4 + 0.3, T_HANG1 = S4 + 1.9; // the latents move from their merge step to their depth
const T_KD0 = S4 + 2.3, T_KD1 = S4 + 4.5; // the cut moves down to its depth
const T_FA = S4 + 4.8; // subcluster A is named
const T_INS0 = S4B + 0.1, T_INS1 = S4B + 1.3; // the small tree shrinks into an inset at the top right
const T_K0 = S4B + 2.4, T_K1 = S4B + 6.4; // the cut moves down the largest cluster's tree
const T_CALL = S4B + 6.6; // the call-out on its largest subcluster
// 5 the ring's points move to their inputs x, beside their latent tree, which comes back from the inset
const T_BACK0 = S5 + 0.2, T_BACK1 = S5 + 1.2; // the small tree returns to full size, shifted right by DX5
const T_WALK0 = S5 + 1.8, T_WALK1 = S5 + 4.0;
const DX5 = 180; // frame pixels: the small tree's shift on slide 5, clear of the inputs x on the left
// 5B the target of point P is built; every point takes the color of its own y
const T_BLD0 = S5B + 1.0, T_BLD1 = S5B + 3.8; // the path values appear from the root to the point
const T_BOX = S5B + 4.6; // what the network gets and what the ground truth holds
const T_YC0 = S5B + 5.2, T_YC1 = S5B + 6.0; // the points turn from their subcluster's color to the color of their own y
// the inset of the small tree (frame pixels): its box, top right, clear of the cut's label on the bars
const INS = { x0: 1302, x1: 1552, y0: 128, y1: 262 };

// ── screen grid (fractions of the frame, 16:9 reference) ───────────────────────────────────────
// A caption of five lines begins near y = 648 of 900 and reaches x = 776 of 1600. The left panel and
// the bottom labels of the chart therefore stay above y = 620.
const ASPECT = 16 / 9;
const LX = 0.225, LY = 0.456; // center of the left panel
const BALL = 0.22; // half-height of the dataset cloud, as a fraction of the frame height
const PX0 = 0.455, PX1 = 0.7, PY0 = 0.285, PY1 = 0.653; // chart frame: left, right, top, bottom
const PW = 4, PH = (PW * ((PY1 - PY0) / (PX1 - PX0))) / ASPECT; // chart frame in panel units
const UPX = PW / ((PX1 - PX0) * 1600); // panel units per frame pixel
const DP = 14; // distance of the camera-anchored chart from the camera
const TITLE_X = 0.0255, TITLE_Y = 0.232, LEG_Y = 0.676; // left-panel title and legend line (clear of the HUD's title and caption plates)
const DATA_LEG_Y = 0.7;
const DATA_WINDOW = { x0: 0.018, x1: 0.414, y0: 0.263, y1: 0.682 };
const CLUSTER_WINDOW = { x0: 0.012, x1: 0.41, y0: 0.227, y1: 0.678 };
const T_DY = -47, N_DY = -14; // chart title and y-axis name, pixels above the frame

/** The slider moves in steps of 0.01, so its position 0.33 stands for the critical value 1/3. */
const isCritical = (p01: number) => Math.abs(p01 - 1 / 3) < 0.005;

// Explore is frozen (engine/explore.ts); these keys stay as `?p.<key>` overrides of the generator.
const controls: Control[] = [
  { key: 'seed', label: 'seed  cluster trees', kind: 'range', value: 26, min: 1, max: 200, step: 1, hint: 'Resamples the cluster sizes and every tree. Inputs and targets change with the trees.' },
  { key: 'seedE', label: 'seed  input positions', kind: 'range', value: 6, min: 1, max: 200, step: 1, hint: 'Resamples the inputs $x$ only. Every edge and every target $y$ stays the same.' },
  { key: 'seedV', label: 'seed  values', kind: 'range', value: 1, min: 1, max: 200, step: 1, hint: 'Resamples the latent values $z$ and the targets $y$ only. Every edge and every input $x$ stays the same.' },
  { key: 'n', label: '$n$  points', kind: 'range', value: 40000, min: 5000, max: 60000, step: 1000, hint: 'More points allow larger clusters and a more precise estimate of how their frequency falls with size.' },
  { key: 'p01', label: '$p_{01}$  $P(\\text{new cluster})$', kind: 'range', value: 1 / 3, min: 0.15, max: 0.6, step: 0.01, format: (v) => (isCritical(v) ? '1/3' : v.toFixed(2)), hint: '$p_{01} = 1/3$ is critical, and the slider position 0.33 stands for it. Other values move the size exponent to $\\tau = (2 - p_{01})/(1 - p_{01})$.' },
  { key: 's', label: '$s$  ring size of the demo cluster', kind: 'range', value: 56, min: 16, max: 96, step: 1, hint: 'Number of nodes that the cyclic coalescent merges one at a time.' },
  {
    key: 'view', label: '3-D view of the $\\mathbb{R}^{32}$ walk', kind: 'select', value: 'pca',
    options: [{ value: 'pca', label: 'per-cluster PCA' }, { value: 'random', label: 'one random projection' }],
    hint: 'Principal component analysis (PCA) shows each cluster along the three directions in which its points spread most. A random projection compresses its fractal dimension of 4 into 3-D.',
  },
  { key: 'k', label: '$k$  cut depth of the latent tree', kind: 'range', value: 0, min: 0, max: 60, step: 1, live: true, hint: 'Each latent at depth $k$ of the largest cluster has one subcluster: the points under it.' },
];

const int = (x: number) => Math.round(x).toLocaleString('en-US');
/** An integer inside maths: the thousands separator needs braces in TeX. */
const ti = (x: number) => int(x).replace(/,/g, '{,}');
const sm = smooth;
const f2 = (x: number) => (Math.abs(x) < 5e-3 ? 0 : x).toFixed(2); // two decimals, no "−0.00"
/** 0 before a, 1 after b, linear in between. */
const ramp = (a: number, b: number, x: number) => clamp((x - a) / (b - a));
/** Fades in over [a, a + f] and out over [b, b + f]. */
const win = (t: number, a: number, b: number, f = 0.6) => sm(a, a + f, t) * (1 - sm(b, b + f, t));
/** A minus sign, not a hyphen. */
const neg = (x: string) => x.replace(/^-/, '−');

/** Filled cyclic group marks, shared by the construction sequence and the larger depth-cut example. */
function cyclicRibbons(groups: { start: number; count: number; color: RGB }[], points: number,
  at: (q: number, r: number) => ArrayLike<number>, radius: number, width: number) {
  const positions: number[] = [], colors: number[] = [];
  const triangle = (a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>, ink: RGB) => {
    for (const p of [a, b, c]) { positions.push(p[0], p[1], 0); colors.push(...ink.map((v) => v * 0.85)); }
  };
  for (const g of groups) {
    const start = g.start - 0.36, end = start + g.count - 0.28;
    const arrow = Math.min(0.14 * points / 6, (end - start) * 0.3);
    const samples = Math.max(6, Math.ceil(72 * g.count / points));
    for (let i = 0; i < samples; i++) {
      const a = lerp(start, end - arrow, i / samples), b = lerp(start, end - arrow, (i + 1) / samples);
      const p0 = at(a, radius - width), p1 = at(a, radius + width);
      const p2 = at(b, radius - width), p3 = at(b, radius + width);
      triangle(p0, p1, p2, g.color); triangle(p2, p1, p3, g.color);
    }
    triangle(at(end - arrow, radius - 2 * width), at(end - arrow, radius + 2 * width), at(end, radius), g.color);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.MeshBasicNodeMaterial(), opacity = uniform(1);
  material.colorNode = attribute('color', 'vec3').mul(opacity);
  material.transparent = true; material.depthWrite = false; material.depthTest = false;
  material.side = THREE.DoubleSide; material.blending = THREE.AdditiveBlending;
  return { object: new THREE.Mesh(geometry, material), uIntensity: opacity };
}

/** Append a dashed line from a to b (panel units): n dashes of length `on` with gaps `off`, stretched so that it starts and ends with a dash. */
function dashTo(L: LineField, ax: number, ay: number, bx: number, by: number, on: number, off: number, c: ArrayLike<number>, k: number) {
  const len = Math.hypot(bx - ax, by - ay);
  if (len < 1e-9) return;
  const n = Math.max(1, Math.round((len + off) / (on + off))), u = 1 / (n * on + (n - 1) * off);
  for (let i = 0; i < n; i++) {
    const f0 = i * (on + off) * u, f1 = f0 + on * u;
    L.push([lerp(ax, bx, f0), lerp(ay, by, f0), 0], [lerp(ax, bx, f1), lerp(ay, by, f1), 0], c, k);
  }
}

/** Write one segment straight into a LineField (no temporaries). */
function seg(L: LineField, i: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  r0: number, g0: number, b0: number, r1 = r0, g1 = g0, b1 = b0) {
  const p = L.positions, c = L.colors, j = i * 6;
  p[j] = ax; p[j + 1] = ay; p[j + 2] = az; p[j + 3] = bx; p[j + 4] = by; p[j + 5] = bz;
  c[j] = r0; c[j + 1] = g0; c[j + 2] = b0; c[j + 3] = r1; c[j + 4] = g1; c[j + 5] = b1;
}
const showL = (L: LineField, on: boolean) => { L.object.visible = on && L.count > 0; };

/** A frontal camera that puts the world point S at the center of the left panel (or at the frame fraction lx), d away, with a small yaw / pitch about S. */
function mkShot(t: number, S: ArrayLike<number>, d: number, yawDeg = 0, pitchDeg = 0, fov = 38, lx = LX, ly = LY): Shot {
  const hf = 2 * d * Math.tan((fov * Math.PI) / 360), wf = hf * ASPECT;
  const dx = (0.5 - lx) * wf, dy = (ly - 0.455) * hf;
  const ya = (yawDeg * Math.PI) / 180, pa = (pitchDeg * Math.PI) / 180;
  const cy = Math.cos(ya), sy = Math.sin(ya), cp = Math.cos(pa), sp = Math.sin(pa);
  const r = [cy, 0, -sy], b = [sy * cp, sp, cy * cp], u = [-sp * sy, cp, -sp * cy];
  const look: [number, number, number] = [S[0] + r[0] * dx + u[0] * dy, S[1] + r[1] * dx + u[1] * dy, S[2] + r[2] * dx + u[2] * dy];
  return { t, pos: [look[0] + b[0] * d, look[1] + b[1] * d, look[2] + b[2] * d], look, fov };
}

/** The offline study slides, with their own figures, beats, notes and results. */
interface Block {
  beats: Beat[];
  end: number; // local time at which the block, and the chapter, ends
  draw(t: number, api: FrameApi): void;
  notes(t: number): Hotspot[];
  details(t: number): NoteDetails | undefined;
  results: Result[];
}

/** Where the offline study's results are served (public/data, next to the page). */
const OFFLINE_URL = `${import.meta.env.BASE_URL}data/percolation-offline.json`;

/**
 * Label heights for the values y: neighbors end up far enough apart for their heights h (default: `gap`
 * each, so that neighbors are `gap` apart), in the same order, inside [lo, hi].
 */
function dodge(y: number[], gap: number, lo: number, hi: number, h?: number[]): number[] {
  const H = h ?? y.map(() => gap), sep = (a: number, b: number) => (H[a] + H[b]) / 2;
  const idx = y.map((_, i) => i).sort((a, b) => y[a] - y[b]), out = y.slice();
  idx.forEach((i, q) => { out[i] = Math.max(clamp(y[i], lo, hi), q ? out[idx[q - 1]] + sep(idx[q - 1], i) : lo); });
  for (let q = idx.length - 1; q >= 0; q--) out[idx[q]] = Math.min(out[idx[q]], q === idx.length - 1 ? hi : out[idx[q + 1]] - sep(idx[q], idx[q + 1]));
  return out;
}

/**
 * A short sample of a chart's line, put before the line's name in a legend: solid or dashed, with a filled mark, an
 * open mark or none, as the line itself is drawn. It takes the color of the text around it.
 */
function lineKey(dashed: boolean, mark: 'filled' | 'open' | null, w = 2) {
  const segs = dashed ? [[0, 5], [8.5, 12], [22, 25.5], [29, 34]] : [[0, 34]];
  return '<svg width="2.8em" height="0.9em" viewBox="0 0 34 11" style="vertical-align:-0.08em;margin-right:0.45em;overflow:visible">' +
    segs.map(([a, b]) => `<line x1="${a}" y1="5.5" x2="${b}" y2="5.5" stroke="currentColor" stroke-width="${w}"/>`).join('') +
    (mark === 'filled' ? '<circle cx="17" cy="5.5" r="3.2" fill="currentColor"/>'
      : mark === 'open' ? '<circle cx="17" cy="5.5" r="4.4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/>' : '') +
    '</svg>';
}

/** A value z as a color: the team ink above 0, the thinned key ink below, stronger for larger |z| (as for y on slide 5). */
function zRGB(z: number, out: number[]) {
  const a = clamp(Math.abs(z) / 2.2), c = z < 0 ? NEG : C.violet, b = (0.16 + 0.95 * Math.pow(a, 1.1)) * 0.85;
  out[0] = c[0] * b; out[1] = c[1] * b; out[2] = c[2] * b;
}
/** The color zRGB prints for a value z, as CSS: for the keys and the color bar that name these marks. */
function zCss(z: number) {
  const a = clamp(Math.abs(z) / 2.2), b = (0.16 + 0.95 * Math.pow(a, 1.1)) * 0.85;
  return z < 0 ? printed([['white', 0.43 * b]]) : printed([['violet', b]]);
}
/** A key mark in the color of the value z. */
const dot = (z: number) => `<span style="color:${zCss(z)}">●</span>`;
/** A color bar for values z from −2 to +2, as an inline drawing for a label. */
const zBarCss = () => {
  const stops = [-2, -1.5, -1, -0.5, -0.01, 0.01, 0.5, 1, 1.5, 2].map((z) => `${zCss(z)} ${(((z + 2) / 4) * 100).toFixed(1)}%`).join(', ');
  return `−2 <span style="display:inline-block;width:7.5em;height:0.62em;vertical-align:-0.02em;margin:0 0.35em;background:linear-gradient(to right, ${stops})"></span> +2`;
};

const percolation: ChapterFactory = {
  id: 'percolation',
  label: 'percolation · cyclic coalescent',
  controls,
  async build(ctx, params) {
    // The engine owns Notes mode; DOM classes only reflect it.
    const notesOpen = () => !!ctx.film?.notesOpen;
    const P = resolveParams(controls, params);
    const seed = Math.round(P.seed), seedE = Math.round(P.seedE), seedV = Math.round(P.seedV);
    const nPts = Math.round(P.n), p01 = isCritical(+P.p01) ? 1 / 3 : +P.p01, sRing = Math.round(P.s);
    const view: 'pca' | 'random' = P.view === 'random' ? 'random' : 'pca';
    const critical = isCritical(p01);
    // Fetch the offline study's results while the generator runs.
    const offlineReq = fetch(OFFLINE_URL).then((r) => { if (!r.ok) throw new Error(`${OFFLINE_URL}: ${r.status}`); return r.json(); });
    offlineReq.catch(() => {}); // awaited below

    // ═══ 1. the generator ══════════════════════════════════════════════════════════════════════
    const data = generate({ n: nPts, d: DIM, seeds: { graph: seed, embedding: seedE, value: seedV }, p01, demoSize: sRing, view }, mulberry32);
    const { n, nClusters, nPA, size, off, latOff, edgeU, edgeV, demo } = data;
    await ctx.log(`${int(nPA)} clusters ✓  measuring`);

    // ═══ 2. measurements: the two checks of the generator ═════════════════════════════════════
    const paSize = size.subarray(0, nPA);
    const rhoTh = 1 / (1 - p01); // CCDF exponent predicted for preferential attachment (Yule–Simon)
    const fittedRho = fitYuleSimon(paSize);
    const cc = sizeCCDF(paSize);
    let nTiny = 0; // clusters of one or two points
    for (let c = 0; c < nPA; c++) if (paSize[c] <= 2) nTiny++;
    const hero = data.largest, hs = size[hero], ho = off[hero];
    const degMin = Math.min(50, hs);
    const dCnt = [0, 0, 0, 0, 0];
    let dTot = 0, degClusters = 0;
    for (let c = 0; c < nPA; c++) {
      if (size[c] < degMin) continue;
      degClusters++;
      for (let i = off[c]; i < off[c + 1]; i++) { dCnt[Math.min(4, data.degree[i])]++; dTot++; }
    }
    const degShare = [1, 2, 3].map((j) => dCnt[j] / Math.max(1, dTot));
    const slopeRun = neg(`-${fittedRho.toFixed(2)}`), slopeTh = neg(`-${rhoTh.toFixed(2)}`);
    const degOk = Math.abs(degShare[0] - Math.exp(-1)) <= 0.02;
    await ctx.log('statistics ✓  laying out');

    // ═══ 3. world layout of the dataset ════════════════════════════════════════════════════════
    // Each cluster is drawn in its own 3-frame around a display position. Positions are uniform in a
    // 3-ball: the directions of the sampled 32-D origins are kept and their radii are replaced by ranks,
    // because the model only says that origins are i.i.d. uniform in a ball. The whole constellation is
    // rotated rigidly so that the largest cluster lies on the camera side. The demo cluster sits at the
    // world origin, where the ring is drawn; the dataset's pictures leave it out.
    const K = view === 'pca' ? 0.72 : 1.7; // world units per walk step
    const R3 = 8.5; // radius of the ball of cluster positions
    const BLOB: RGB = [3.2, 0.4, -6.8];
    const W = new Float32Array(n * 3);
    const orgW = new Float32Array(nClusters * 3);
    const wLog = new Float32Array(nClusters);
    {
      const rad = new Float32Array(nPA);
      for (let c = 0; c < nPA; c++) rad[c] = Math.hypot(data.origin[c * 3], data.origin[c * 3 + 1], data.origin[c * 3 + 2]);
      const byRad = Array.from({ length: nPA }, (_, c) => c).sort((a, b) => rad[a] - rad[b]);
      const rNew = new Float32Array(nPA);
      byRad.forEach((c, rank) => { rNew[c] = R3 * Math.cbrt((rank + 0.5) / nPA); });
      const qRot = new THREE.Quaternion();
      const h0 = new THREE.Vector3(data.origin[hero * 3], data.origin[hero * 3 + 1], data.origin[hero * 3 + 2]);
      if (h0.length() > 1e-9) qRot.setFromUnitVectors(h0.clone().normalize(), new THREE.Vector3(0.2, 0.1, 0.97).normalize());
      const rr = mulberry32(seed * 31 + 7);
      const v = new THREE.Vector3();
      for (let c = 0; c < nClusters; c++) {
        let ox = 0, oy = 0, oz = 0;
        if (c !== demo) {
          v.set(data.origin[c * 3], data.origin[c * 3 + 1], data.origin[c * 3 + 2]);
          if (rad[c] > 1e-9) v.multiplyScalar(rNew[c] / rad[c]);
          v.applyQuaternion(qRot);
          ox = BLOB[0] + v.x; oy = BLOB[1] + v.y; oz = BLOB[2] + v.z;
        }
        orgW[c * 3] = ox; orgW[c * 3 + 1] = oy; orgW[c * 3 + 2] = oz;
        wLog[c] = clamp(Math.log10(size[c]) / 3.3);
        // a uniformly random rotation of the cluster's principal frame (Shoemake); hero and demo keep theirs
        const u1 = rr(), u2 = rr() * TAU, u3 = rr() * TAU;
        let m00 = 1, m01 = 0, m02 = 0, m10 = 0, m11 = 1, m12 = 0, m20 = 0, m21 = 0, m22 = 1;
        if (view === 'pca' && c !== hero && c !== demo) {
          const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
          const qx = a * Math.sin(u2), qy = a * Math.cos(u2), qz = b * Math.sin(u3), qw = b * Math.cos(u3);
          m00 = 1 - 2 * (qy * qy + qz * qz); m01 = 2 * (qx * qy - qz * qw); m02 = 2 * (qx * qz + qy * qw);
          m10 = 2 * (qx * qy + qz * qw); m11 = 1 - 2 * (qx * qx + qz * qz); m12 = 2 * (qy * qz - qx * qw);
          m20 = 2 * (qx * qz - qy * qw); m21 = 2 * (qy * qz + qx * qw); m22 = 1 - 2 * (qx * qx + qy * qy);
        }
        for (let i = off[c]; i < off[c + 1]; i++) {
          const x = data.rel[i * 3] * K, y = data.rel[i * 3 + 1] * K, z = data.rel[i * 3 + 2] * K;
          W[i * 3] = ox + m00 * x + m01 * y + m02 * z;
          W[i * 3 + 1] = oy + m10 * x + m11 * y + m12 * z;
          W[i * 3 + 2] = oz + m20 * x + m21 * y + m22 * z;
        }
      }
    }
    const Hc: RGB = [orgW[hero * 3], orgW[hero * 3 + 1], orgW[hero * 3 + 2]];
    // bounding box of the largest cluster in the x–y plane (the camera is frontal), for framing and labels
    const hbb = [Infinity, -Infinity, Infinity, -Infinity];
    for (let i = ho; i < ho + hs; i++) {
      hbb[0] = Math.min(hbb[0], W[i * 3]); hbb[1] = Math.max(hbb[1], W[i * 3]);
      hbb[2] = Math.min(hbb[2], W[i * 3 + 1]); hbb[3] = Math.max(hbb[3], W[i * 3 + 1]);
    }
    const Hb: RGB = [(hbb[0] + hbb[1]) / 2, (hbb[2] + hbb[3]) / 2, Hc[2]]; // center of the box
    const hext = [Math.max(0.6, (hbb[1] - hbb[0]) / 2), Math.max(0.6, (hbb[3] - hbb[2]) / 2)];

    const group = new THREE.Group();

    // ═══ 4. the demo cluster: ring, merges, the depth cut, the marked point, the inputs x ══════════
    const T = clusterTree(data, demo);
    const s = T.s, m = T.m, NN = 2 * s - 1, root = NN - 1;
    const doff = off[demo], dlo = latOff[demo];
    const { kd, A, maxDepth: maxDd } = demoCut(T);
    const jS = slowMerge(T); // the merge the film plays slowly
    const pk = markedPoint(T, A, kd, data.y, doff); // the point the film follows, local number
    // Positions are numbered from the first point of the root's arc, so every group is an interval
    // [q0, q1] of 0..s−1 and the ring and the tree of merges share one order.
    const rs = T.start[root];
    const qOf = (p: number) => (p - rs + s) % s;
    const RR = 3.4;
    // The ring turns so that the slow merge happens at its top, where its callouts have room: the boundary between
    // its two groups sits at 12 o'clock, the drawn point's group to its left, the next group to its right.
    const qEndA = (qOf(T.start[T.ca[jS]]) + T.cnt[T.ca[jS]] - 1) % s;
    const rot = (TAU * (qEndA + 1)) / s;
    const angQ = (q: number) => Math.PI / 2 - (TAU * (q + 0.5)) / s + rot; // clockwise in ring order
    const ringP = new Float32Array(s * 3); // ring layout, in the plane z = 0 facing the camera
    for (let p = 0; p < s; p++) { const a = angQ(qOf(p)); ringP[p * 3] = RR * Math.cos(a); ringP[p * 3 + 1] = RR * Math.sin(a); }
    // the same points at their inputs x: the cluster's walk in its first two principal axes, scaled to the panel
    const WALK_R = 4.2;
    const walkP = new Float32Array(s * 3);
    {
      let r = 1e-6;
      for (let p = 0; p < s; p++) r = Math.max(r, Math.hypot(data.rel[(doff + p) * 3], data.rel[(doff + p) * 3 + 1]));
      for (let p = 0; p < s; p++) { walkP[p * 3] = (data.rel[(doff + p) * 3] * WALK_R) / r; walkP[p * 3 + 1] = (data.rel[(doff + p) * 3 + 1] * WALK_R) / r; }
    }
    // merge schedule: the merges before the slow one evenly over [T_PRE0, T_PRE1], the slow one on its own, the rest
    // accelerating over [T_POST0, T_MERGE1]
    const tau = new Float32Array(m), dur = new Float32Array(m);
    {
      const step = (T_PRE1 - T_PRE0) / Math.max(1, jS);
      for (let j = 0; j < jS; j++) { tau[j] = T_PRE0 + j * step; dur[j] = clamp(1.6 * step, 0.34, 0.9); }
      tau[jS] = T_SLOWM; dur[jS] = SLOW_DUR;
      const mp = m - jS - 1;
      if (mp > 0) {
        const total = T_MERGE1 - T_POST0 - 0.45, r = 0.8, d0 = Math.min(0.5, total / mp);
        const G = (1 - Math.pow(r, mp)) / (1 - r);
        const dMin = Math.max(0.02, (total - d0 * G) / Math.max(1e-6, mp - G));
        let t = T_POST0;
        for (let q = 0; q < mp; q++) {
          const dj = dMin + (d0 - dMin) * Math.pow(r, q), j = jS + 1 + q;
          tau[j] = t; dur[j] = clamp(1.6 * dj, 0.34, 1.2);
          t += dj;
        }
      }
    }
    // group colors: neighboring nodes start with clearly different hues; a merged group keeps the hue of its larger part
    const colIdx = new Uint8Array(NN);
    {
      const rc = mulberry32(seed * 131 + 17);
      const byQ = new Uint8Array(s);
      for (let q = 0; q < s; q++) {
        let c = Math.floor(rc() * HUES.length), guard = 0;
        while (guard++ < 12 && ((q > 0 && c % 3 === byQ[q - 1] % 3) || (q === s - 1 && c % 3 === byQ[0] % 3))) c = (c + 1) % HUES.length;
        byQ[q] = c;
      }
      for (let p = 0; p < s; p++) colIdx[p] = byQ[qOf(p)];
    }
    const chg: Int32Array[] = []; // per merge: the points whose group color changes
    for (let j = 0; j < m; j++) {
      const a = T.ca[j], b = T.cb[j];
      const big = T.cnt[a] >= T.cnt[b] ? a : b, small = big === a ? b : a;
      colIdx[s + j] = colIdx[big];
      const list = new Int32Array(T.cnt[small]);
      for (let q = 0; q < list.length; q++) list[q] = (T.start[small] + q) % s;
      chg.push(list);
    }
    const palRing: RGB[] = HUES.map((_, i) => HUES[(i - colIdx[root] + 2 + HUES.length) % HUES.length]); // the final group is key ink
    const inA = new Uint8Array(NN);
    {
      const st = [A];
      while (st.length) { const x = st.pop()!; inA[x] = 1; if (x >= s) st.push(T.ca[x - s], T.cb[x - s]); }
    }
    /** The accent of subcluster A, the one the film follows. */
    const ACC: RGB = C.cyan, ACC_CSS = CSS.cyan;
    /** The other subclusters of the small tree, in turn along the ring: no teal, so that subcluster A stays the only one. */
    const FAM_OTHER: RGB[] = [scaleRGB(C.violet, 1.1), scaleRGB(C.white, 0.8), C.magenta, scaleRGB(C.violet, 0.72)];
    const ISO_D: RGB = scaleRGB(C.slate, 0.55); // a point above the cut, in no subcluster
    /**
     * The color of every node when the small tree is cut at depth k (1 … kd): −2 a latent above the cut (key ink),
     * −1 a point above the cut, 0 the subcluster that holds A, 1 … 4 the other subclusters in ring order.
     */
    const cutIdx: Int8Array[] = [];
    for (let k = 0; k <= kd; k++) {
      const top = new Int8Array(NN).fill(-3); // color of each node at depth k
      const atK: number[] = [];
      for (let node = 0; node < NN; node++) if (T.depth[node] === k) atK.push(node);
      atK.sort((a, b) => qOf(T.start[a]) - qOf(T.start[b]));
      let i = 0;
      for (const node of atK) top[node] = isAncestor(node, A) ? 0 : 1 + (i++ % FAM_OTHER.length);
      const idx = new Int8Array(NN);
      for (let node = 0; node < NN; node++) {
        if (T.depth[node] < k) { idx[node] = node < s ? -1 : -2; continue; }
        let x = node;
        for (let d = T.depth[node]; d > k; d--) x = T.par[x];
        idx[node] = top[x];
      }
      cutIdx.push(idx);
    }
    function isAncestor(a: number, b: number) { for (let x = b; x >= 0; x = T.par[x]) if (x === a) return true; return false; }
    const cutRGB = (i: number): RGB => (i === -2 ? INK : i === -1 ? ISO_D : i === 0 ? ACC : FAM_OTHER[i - 1]);
    // The larger example’s points and links; depth-cut groups use the same ribbons as construction.
    const nodeSize = clamp(Math.sqrt(56 / s), 0.7, 1.25);
    const NODE = 0.36, ARCH_PT = 0.16;
    const nodes = new PointField({ count: s, size: NODE, bokeh: 0, twinkle: 0 });
    const chordF = new LineField(m, { width: 2 });
    chordF.count = m;
    const atRing = (q: number, r: number): [number, number, number] => [r * Math.cos(angQ(q)), r * Math.sin(angQ(q)), 0];
    // Each depth cut uses the same cyclic group ribbons as the six-point construction.
    // Short branches that end above the cut keep their gray points and receive no ribbon.
    const cutRibbons = cutIdx.map((idx, k) => {
      const bands = cyclicRibbons(Array.from({ length: NN }, (_, node) => node)
        .filter((node) => T.depth[node] === k)
        .map((node) => ({ start: qOf(T.start[node]), count: T.cnt[node], color: cutRGB(idx[node]) })),
      s, atRing, RR + 0.37, 0.055);
      bands.object.name = `percolation-cut-groups-${k}`;
      group.add(bands.object);
      return bands;
    });
    const selectedRibbon = cyclicRibbons([{ start: qOf(T.start[A]), count: T.cnt[A], color: ACC }],
      s, atRing, RR + 0.37, 0.055);
    group.add(selectedRibbon.object);
    // point P: a ring around it, in the plane of the screen
    const RING_N = 28;
    const pickRing = new LineField(RING_N, { width: 2 });
    pickRing.count = RING_N;
    // each point's own target y as a color (slide 5), and the probe's target at the cut depth kd: the value z of the
    // point's latent at that depth, or none for a point whose path ends at or above it (probe-target slide)
    const yCol = new Float32Array(s * 3), tgtCol = new Float32Array(s * 3), tgtHas = new Uint8Array(s);
    {
      const tmp = [0, 0, 0];
      for (let p = 0; p < s; p++) {
        zRGB(data.y[doff + p], tmp);
        yCol.set(tmp, p * 3);
        let x = p;
        for (let d = T.depth[p]; d > kd; d--) x = T.par[x];
        if (T.depth[p] > kd && x >= s) { tgtHas[p] = 1; zRGB(data.zLat[dlo + x - s], tmp); tgtCol.set(tmp, p * 3); }
        else tgtCol.set(ISO_D, p * 3);
      }
    }
    group.add(chordF.object, nodes.object, pickRing.object);

    // ═══ 5. the dataset as a 3-D point cloud ═══════════════════════════════════════════════════
    const arch = new PointField({ count: n, size: ARCH_PT, bokeh: 0, twinkle: 0 });
    const baseCol = new Float32Array(n * 3);
    const rc = new Float32Array(nClusters), cl = data.clusterOf; // per-cluster visibility, cluster of each point
    const heroInk = clamp(Math.sqrt(2600 / Math.max(1, hs)), 0.4, 1); // a very large cluster overlaps itself: use less ink per mark
    for (let c = 0; c < nClusters; c++) {
      const w = wLog[c], kb = (0.22 + 0.3 * w) * TK;
      for (let i = off[c]; i < off[c + 1]; i++) {
        arch.setPos(i, W[i * 3], W[i * 3 + 1], W[i * 3 + 2]);
        arch.sizes[i] = 0.8 + 0.3 * w;
        baseCol[i * 3] = C.violet[0] * kb; baseCol[i * 3 + 1] = C.violet[1] * kb; baseCol[i * 3 + 2] = C.violet[2] * kb;
      }
    }
    arch.colors.fill(0);
    arch.commit();
    // cluster edges as hairlines: largest clusters first, within a segment budget
    const E_MAX = 48000;
    const order = Array.from({ length: nClusters }, (_, c) => c).filter((c) => size[c] > 1 && c !== hero && c !== demo).sort((a, b) => size[b] - size[a]);
    if (hs > 1) order.unshift(hero);
    const eL = new Int32Array(Math.min(data.nLat, E_MAX)), ef = new Float32Array(eL.length);
    let nE = 0;
    for (const c of order) {
      const mc = size[c] - 1;
      if (nE + mc > eL.length) continue;
      for (let j = 0; j < mc; j++) { ef[nE] = (0.07 + 0.33 * wLog[c] * wLog[c]) * TK; eL[nE++] = latOff[c] + j; }
    }
    const archL = new LineField(Math.max(1, nE));
    for (let e = 0; e < nE; e++) {
      const u = edgeU[eL[e]] * 3, v = edgeV[eL[e]] * 3;
      seg(archL, e, W[u], W[u + 1], W[u + 2], W[v], W[v + 1], W[v + 2], 0, 0, 0);
    }
    archL.count = nE;
    archL.commit();
    // Only the dataset marks move under pointer input. Their local origin is the center of the
    // constellation, so rotation and magnification preserve that center on the screen.
    const cloudPivot = new THREE.Group(), cloudContent = new THREE.Group();
    cloudPivot.position.set(...BLOB);
    cloudContent.position.set(-BLOB[0], -BLOB[1], -BLOB[2]);
    cloudContent.add(archL.object, arch.object);
    cloudPivot.add(cloudContent);
    group.add(cloudPivot);
    const cloudRotation = new THREE.Quaternion(), clusterRotation = new THREE.Quaternion(), turn = new THREE.Quaternion();
    let cloudZoom = 1, clusterZoom = 1;
    const datasetSlide = (t: number) => t >= S2 && t < S3;
    const clusterSlide = (t: number) => t >= S4B && t < S5;
    const rotationChanged = (q: THREE.Quaternion) => Math.abs(q.w) < 1 - 1e-9;
    const clusterViewChanged = () => Math.abs(clusterZoom - 1) > 1e-6 || rotationChanged(clusterRotation);
    const resetCloud = () => { cloudRotation.identity(); clusterRotation.identity(); cloudZoom = clusterZoom = 1; };
    await ctx.log(`${int(nE)} edges ✓  cutting the latent tree`);

    // ═══ 6. subclusters of the largest cluster at every cut depth ═════════════════════════════
    const TH = clusterTree(data, hero);
    const FT = familyTable(TH, 60, HUES.length, 3);
    const kCap = FT.kCap, kFilm = heroCut(TH, kCap); // the film's cut: subclusters from one point to hundreds
    const hm = Math.max(0, hs - 1);
    // The cut view can rotate the largest cluster independently of the faint background. At the
    // default view the original batches still render it, preserving the authored picture exactly.
    const clusterPts = new PointField({ count: hs, size: ARCH_PT, bokeh: 0, twinkle: 0 });
    clusterPts.object.name = 'percolation-largest-cluster-points';
    clusterPts.positions.set(W.subarray(ho * 3, (ho + hs) * 3));
    clusterPts.sizes.set(arch.sizes.subarray(ho, ho + hs));
    clusterPts.colors.fill(0);
    clusterPts.commit();
    const clusterLines = new LineField(Math.max(1, hm));
    clusterLines.object.name = 'percolation-largest-cluster-edges';
    clusterLines.positions.set(archL.positions.subarray(0, hm * 6));
    clusterLines.count = hm;
    clusterLines.commit();
    const clusterPivot = new THREE.Group(), clusterContent = new THREE.Group();
    clusterPivot.name = 'percolation-largest-cluster';
    // Per-cluster input offsets are centered on their centroid by the generator, so Hc is its
    // true three-dimensional center; Hb is only the camera's frontal bounding-box center.
    clusterPivot.position.set(...Hc);
    clusterContent.position.set(-Hc[0], -Hc[1], -Hc[2]);
    clusterContent.add(clusterLines.object, clusterPts.object);
    clusterPivot.add(clusterContent);
    group.add(clusterPivot);
    const FAM_K = 0.65 * heroInk, FAM_EDGE = 0.4 * heroInk;
    // slide 2: the largest cluster in the team's ink at the strength of a subcluster, the other clusters fainter
    const HERO_COL: RGB = scaleRGB(C.violet, FAM_K), HERO_EDGE: RGB = scaleRGB(C.violet, FAM_EDGE);
    const OTHERS_K = 0.42;
    const ISO: RGB = [C.slate[0] * 0.3, C.slate[1] * 0.3, C.slate[2] * 0.3];
    const CUT: RGB = [0, 0, 0]; // a cut edge carries no ink
    // null = follow the film. A value from a control or URL replaces the dataset cut only while Notes is open.
    let liveK: number | null = params && params.k !== undefined ? clamp(+P.k, 0, kCap) : null;
    const notesCutK = () => (notesOpen() ? liveK : null);
    // The first cuts of a random tree peel off a few points each, so the film starts the cut two levels
    // above the first depth at which a tenth of the cluster has left the largest subcluster.
    let kSplit = 0;
    while (kSplit < kFilm && FT.largest[kSplit] > 0.9 * hs) kSplit++;
    const kStart = kSplit >= kFilm ? 0 : Math.max(0, Math.min(kSplit - 2, kFilm - 12));
    const kAnim = (t: number) => kStart + (kFilm - kStart) * Math.pow(clamp((t - T_K0) / (T_K1 - T_K0)), 1.25);
    // The bars of the latent tree begin at the row where the cut begins: the rows above it all look alike.
    const kTop = kStart;
    /** Subclusters at the cut depth k: one per node at depth k, single points included; the points above the cut are in none. */
    const subAt = (k: number) => {
      let nSub = 0, above = 0, small = 0, min = Infinity;
      for (let node = 0; node < 2 * hs - 1; node++) if (TH.depth[node] === k) { nSub++; if (TH.cnt[node] <= 2) small++; min = Math.min(min, TH.cnt[node]); }
      for (let p = 0; p < hs; p++) if (TH.depth[p] < k) above++;
      return { nSub, above, small, min, largest: FT.largest[k] };
    };
    const subFilm = subAt(kFilm);
    // The call-out names the largest subcluster at the final cut, in its own color. Its leader line ends at the
    // subcluster's core: the point nearest to the median x and y of its points in the frontal view. Where the
    // call-out itself goes is decided once the camera is known (section 9).
    let callNode = -1;
    for (let node = hs; node < 2 * hs - 1; node++) if (TH.depth[node] === kFilm && (callNode < 0 || TH.cnt[node] > TH.cnt[callNode])) callNode = node;
    const callCore: RGB = [Hb[0], Hb[1], Hb[2]];
    if (callNode >= 0) {
      const xs: number[] = [], ys: number[] = [], ids: number[] = [];
      for (let p = 0; p < hs; p++) {
        if (TH.depth[p] < kFilm) continue;
        let node = p;
        for (let dd = TH.depth[p]; dd > kFilm; dd--) node = TH.par[node];
        if (node !== callNode) continue;
        const i = (ho + p) * 3;
        xs.push(W[i]); ys.push(W[i + 1]); ids.push(i);
      }
      if (xs.length) {
        const mx = xs.slice().sort((a, b) => a - b)[xs.length >> 1], my = ys.slice().sort((a, b) => a - b)[ys.length >> 1];
        let bi = ids[0], bd = Infinity;
        for (const i of ids) { const d = (W[i] - mx) ** 2 + (W[i + 1] - my) ** 2; if (d < bd) { bd = d; bi = i; } }
        callCore[0] = W[bi]; callCore[1] = W[bi + 1]; callCore[2] = W[bi + 2];
      }
    }
    /** The text color of each entry of HUES (the light tints read in the darker text value of their hue). */
    const HUE_CSS = [CSS.violet, CSS.cyan, CSS.white, CSS.magenta, CSS.cyan, CSS.violet];
    const callCss = callNode >= 0 ? HUE_CSS[FT.col[callNode]] : CSS.white;

    // the path of the marked point: z_0 (root latent) … z_{d−1}, then the point's own value z_d
    const pickD = T.depth[pk], gp = doff + pk;
    const pathZ = new Float32Array(pickD + 1);
    const pathSize = new Int32Array(pickD + 1); // points owned by each latent on the path (1 for the point itself)
    pathZ[pickD] = data.zLeaf[gp];
    pathSize[pickD] = 1;
    for (let node = T.par[pk], i = pickD - 1; node >= 0 && i >= 0; node = T.par[node], i--) { pathZ[i] = data.zLat[dlo + node - s]; pathSize[i] = T.cnt[node]; }
    let pathSum = 0;
    for (let i = 0; i <= pickD; i++) pathSum += pathZ[i];
    const pickY = pathSum / Math.sqrt(1 + pickD); // equals data.y[gp] up to rounding

    // ═══ 7. the chart panel (camera-anchored, flat, facing the viewer) ═════════════════════════
    const inset = new THREE.Group();
    group.add(inset);
    const AXK = 0.8, GRIDK = 0.16;
    const frameLines = (L: LineField) => { L.push([0, 0, 0], [PW, 0, 0], INK, AXK); L.push([0, 0, 0], [0, PH, 0], INK, AXK); };

    // 7a. slide 2: size CCDF on log–log axes with the exact Yule–Simon law
    const lxMax = Math.max(1, Math.ceil(Math.log10(Math.max(2, cc.s[cc.s.length - 1])) * 2) / 2);
    const lyMin = -Math.max(1, Math.ceil(Math.log10(nPA) * 2) / 2);
    const ccX = (lx: number) => (lx / lxMax) * PW, ccY = (ly: number) => ((ly - lyMin) / -lyMin) * PH;
    const xDec: number[] = [], yDec: number[] = [];
    for (let v = 0; v <= lxMax + 1e-9; v++) xDec.push(v);
    for (let v = 0; v >= lyMin - 1e-9; v--) yDec.push(v);
    const ccAx = new LineField(8 + 2 * (xDec.length + yDec.length));
    frameLines(ccAx);
    for (const v of xDec) { ccAx.push([ccX(v), 0, 0], [ccX(v), -0.07, 0], INK, AXK); if (v > 0) ccAx.push([ccX(v), 0, 0], [ccX(v), PH, 0], INK, GRIDK); }
    for (const v of yDec) { ccAx.push([0, ccY(v), 0], [-0.07, ccY(v), 0], INK, AXK); if (v < 0) ccAx.push([0, ccY(v), 0], [PW, ccY(v), 0], INK, GRIDK); }
    ccAx.commit();
    const CURVE_N = 56;
    const ccCurve = new LineField(CURVE_N, { width: 2 });
    {
      let prev: RGB | null = null;
      for (let i = 0; i <= CURVE_N; i++) {
        const lx = (lxMax * i) / CURVE_N, ly = Math.log10(Math.max(1e-300, yuleSimonCCDF(Math.pow(10, lx), rhoTh)));
        const p3: RGB = [ccX(lx), ccY(ly), 0];
        if (prev && ly >= lyMin) ccCurve.push(prev, p3, C.slate, 1.1);
        prev = p3;
      }
      ccCurve.commit();
    }
    const curveSegs = ccCurve.count;
    const ccStride = Math.max(1, Math.ceil(cc.s.length / 500));
    const ccIdx: number[] = [];
    for (let i = 0; i < cc.s.length; i += ccStride) ccIdx.push(i);
    if (ccIdx[ccIdx.length - 1] !== cc.s.length - 1) ccIdx.push(cc.s.length - 1);
    const ccPts = new PointField({ count: ccIdx.length, size: 0.115, bokeh: 0, twinkle: 0 });
    ccIdx.forEach((i, q) => {
      ccPts.setPos(q, ccX(Math.log10(cc.s[i])), ccY(Math.log10(cc.p[i])), 0.01);
      ccPts.setColor(q, C.violet, 1);
    });
    ccPts.commit();
    inset.add(ccAx.object, ccCurve.object, ccPts.object);

    // 7b. slide 4: the latent tree of the largest cluster as bars: row k = the latents at depth k, from kTop down to kBot.
    // The rows run a few depths past the film's last cut: they stay faint, so the cut is seen to lie inside the tree.
    const kBot = Math.min(kCap, kFilm + 6);
    const nRows = kBot - kTop + 1;
    const icY = (k: number) => PH - ((k - kTop + 0.5) / nRows) * PH;
    // The chart is wider than the other charts: on slide 4 it shares the frame with the ring's wider latent tree.
    const IW = 1.3 * PW;
    const icX = (p: number) => (p / Math.max(1, hs)) * IW;
    const rowEnd = new Int32Array(kBot + 2);
    const IC_MAX = 16000;
    const IC_W = clamp(PH / UPX / nRows - 2.4, 2, 9); // stroke width in pixels
    // Every row is on the chart from the start: faint below the cut (icGhost), at full strength down to it (icBars on top).
    const icBars = new LineField(IC_MAX, { width: IC_W });
    const icGhost = new LineField(IC_MAX, { width: IC_W });
    const icMin = Math.max(2, Math.ceil(hs / 450)); // bars below this size are not drawn
    const icRs = hs > 1 ? TH.start[2 * hs - 2] : 0;
    {
      const byDepth: number[][] = Array.from({ length: kBot + 1 }, () => []);
      for (let node = hs; node < 2 * hs - 1; node++) if (TH.depth[node] >= kTop && TH.depth[node] <= kBot && TH.cnt[node] >= icMin) byDepth[TH.depth[node]].push(node);
      const cap = (IC_W / 2 + 0.8) * UPX; // half a stroke plus a hairline gap, in panel units
      for (let k = 0; k <= kBot; k++) {
        for (const node of byDepth[k]) {
          const x0 = (TH.start[node] - icRs + hs) % hs;
          const xa = icX(x0) + cap, xb = Math.max(xa + 0.002, icX(x0 + TH.cnt[node]) - cap);
          icBars.push([xa, icY(k), 0], [xb, icY(k), 0], HUES[FT.col[node]], 0.68);
          icGhost.push([xa, icY(k), 0], [xb, icY(k), 0], HUES[FT.col[node]], 0.24);
        }
        rowEnd[k + 1] = icBars.count;
      }
      icBars.commit();
      icGhost.commit();
    }
    const icStep = nRows <= 14 ? 4 : nRows <= 50 ? 8 : 16;
    const icTicks: number[] = [];
    for (let v = Math.ceil(kTop / icStep) * icStep; v <= kBot; v += icStep) icTicks.push(v);
    const icAx = new LineField(8 + icTicks.length);
    icAx.push([0, 0, 0], [IW, 0, 0], INK, AXK);
    icAx.push([0, 0, 0], [0, PH, 0], INK, AXK);
    for (const v of icTicks) icAx.push([0, icY(v), 0], [-0.07, icY(v), 0], INK, AXK);
    icAx.push([0, 0, 0], [0, -0.07, 0], INK, AXK);
    icAx.push([IW, 0, 0], [IW, -0.07, 0], INK, AXK);
    icAx.commit();
    const icCur = new LineField(1, { width: 2 }); // the cut: a rule across the chart under the deepest row at full strength
    icCur.count = 1;
    inset.add(icAx.object, icGhost.object, icBars.object, icCur.object);

    // 7c. slide 5: how the target of the marked point is built: the values on its path as bars, their running sum as a line
    const nbz = pickD + 1;
    const BX0 = 0.1, BX1 = 0.72 * PW, BYX = 0.9 * PW; // the path's bars lie between BX0 and BX1, the target's own bar at BYX
    const bPitch = (BX1 - BX0) / nbz;
    const BAR_W = clamp(Math.round((bPitch / UPX) * 0.62), 2, 11), Y_W = 15; // stroke widths in pixels
    const runZ = new Float32Array(nbz); // running sum down the path, divided by sqrt(1 + d_x); the last entry is y
    let zLim = 2.4;
    {
      let run = 0;
      for (let i = 0; i < nbz; i++) {
        run += pathZ[i];
        runZ[i] = run / Math.sqrt(nbz);
        zLim = Math.max(zLim, Math.abs(pathZ[i]) + 0.25, Math.abs(runZ[i]) + 0.25);
      }
      zLim = Math.ceil(zLim * 2) / 2;
    }
    const bX = (i: number) => BX0 + (i + 0.5) * bPitch, bY = (z: number) => (0.5 + z / (2 * zLim)) * PH;
    /** A bar from the zero line to z, shortened by the stroke's round caps so that it ends where the value is. */
    const zBar = (L: LineField, x: number, z: number, wPx: number, col?: RGB, k = 0.9) => {
      const cap = Math.min((wPx / 2) * UPX, Math.abs(bY(z) - bY(0)) / 2), sg = z < 0 ? -1 : 1;
      L.push([x, bY(0) + sg * cap, 0], [x, bY(z) - sg * cap, 0], col ?? (z < 0 ? NEG : C.violet), k);
    };
    // The bar of latent A (depth kd on the path) and the point's own value (the last bar) are named.
    const aBar = kd < pickD ? kd : -1, last = nbz - 1;
    const bldBars = new LineField(nbz, { width: BAR_W });
    for (let i = 0; i < nbz; i++) zBar(bldBars, bX(i), pathZ[i], BAR_W, INK, i === aBar ? 0 : i === last ? 0.8 : 0.45);
    bldBars.commit();
    const A_W = Math.max(BAR_W + 3, 6);
    const bldA = new LineField(1, { width: A_W });
    if (aBar >= 0) zBar(bldA, bX(aBar), pathZ[aBar], A_W, ACC, 1.2);
    bldA.commit();
    // dashed leads: from A's bar down to its label under the bars, and from the last bar up to its label above them
    const A_LAB = -0.86 * zLim, OWN_LAB = 0.86 * zLim;
    const bldA_L = new LineField(40), bldOwnL = new LineField(40);
    if (aBar >= 0) dashTo(bldA_L, bX(aBar), bY(Math.min(0, pathZ[aBar])) - 6 * UPX, bX(aBar), bY(A_LAB) + 4 * UPX, 4 * UPX, 3 * UPX, ACC, 0.9);
    dashTo(bldOwnL, bX(last), bY(Math.max(0, pathZ[last])) + 6 * UPX, bX(last), bY(OWN_LAB) - 9 * UPX, 4 * UPX, 3 * UPX, INK, 0.7);
    bldA_L.commit();
    bldOwnL.commit();
    const bldRun = new LineField(2 * nbz, { width: 2 }); // staircase: one tread per term, one riser between two terms
    for (let i = 0; i < nbz; i++) {
      if (i > 0) bldRun.push([bX(i) - bPitch / 2, bY(runZ[i - 1]), 0], [bX(i) - bPitch / 2, bY(runZ[i]), 0], INK, 0.85);
      bldRun.push([bX(i) - bPitch / 2, bY(runZ[i]), 0], [bX(i) + bPitch / 2, bY(runZ[i]), 0], INK, 0.85);
    }
    bldRun.commit();
    const bldY = new LineField(1, { width: Y_W });
    zBar(bldY, BYX, pickY, Y_W);
    bldY.commit();
    const bldAx = new LineField(48);
    bldAx.push([0, 0, 0], [PW, 0, 0], INK, AXK);
    bldAx.push([0, 0, 0], [0, PH, 0], INK, AXK);
    bldAx.push([0, bY(0), 0], [PW, bY(0), 0], INK, 0.5);
    const zTicks = [-2, 0, 2];
    for (const v of zTicks) { bldAx.push([0, bY(v), 0], [-0.07, bY(v), 0], INK, AXK); if (v !== 0) bldAx.push([0, bY(v), 0], [PW, bY(v), 0], INK, GRIDK); }
    for (const x of [bX(0), bX(last), BYX]) bldAx.push([x, 0, 0], [x, -0.07, 0], INK, AXK);
    if (aBar >= 0) bldAx.push([bX(aBar), 0, 0], [bX(aBar), -0.07, 0], INK, AXK);
    const bldAxN = bldAx.count; // the dashes that carry the path's last running sum over to the target's bar come after the axes
    dashTo(bldAx, BX1, bY(pickY), BYX - (Y_W / 2 + 3) * UPX, bY(pickY), 5 * UPX, 4 * UPX, INK, 0.8);
    const bldAxAll = bldAx.count;
    bldAx.commit();
    inset.add(bldAx.object, bldBars.object, bldA.object, bldA_L.object, bldOwnL.object, bldRun.object, bldY.object);
    await ctx.log('charts ✓');

    // ═══ 8. the flat figures, in frame pixels (y down), anchored to the camera like the chart ═══════
    const NX0 = 40, NY1 = 600; // frame pixel of the group's origin
    const netG = new THREE.Group();
    group.add(netG);
    const NP = (x: number, y: number): RGB => [x - NX0, NY1 - y, 0];

    // 8a. slide 1: a small tree of real concepts. Its words are labels; each line runs from under a parent's word to
    // over a child's. Under each word of the last row sits a small cloud of examples in that class's color: the two
    // kinds of bird share one hue, the two kinds of dog another, so the nesting shows in the colors as it will in
    // the subclusters of the data.
    const CT = [
      { w: 'animal', x: 800, y: 205, up: -1 },
      { w: 'bird', x: 600, y: 300, up: 0 },
      { w: 'dog', x: 1000, y: 300, up: 0 },
      { w: 'sparrow', x: 505, y: 395, up: 1 },
      { w: 'owl', x: 695, y: 395, up: 1 },
      { w: 'terrier', x: 905, y: 395, up: 2 },
      { w: 'poodle', x: 1095, y: 395, up: 2 },
    ];
    const CT_NOTE = { x: 800, y: 530 };
    const ctAt = (i: number) => T_TREE0 + 0.55 * (CT[i].up < 0 ? 0 : CT[CT[i].up].up < 0 ? 1 : 2); // when word i appears
    const ctL = new LineField(CT.length, { width: 1.5 });
    ctL.count = CT.length - 1;
    const CT_EX = 26; // examples under each word of the last row
    const ctLeaves = CT.map((_, i) => i).filter((i) => i >= 3);
    const ctCol: RGB[] = [scaleRGB(C.violet, 1.1), scaleRGB(C.violet, 0.5), C.cyan, scaleRGB(C.cyan, 0.5)];
    const ctDots = new PointField({ count: ctLeaves.length * CT_EX, size: 1, bokeh: 0, twinkle: 0 });
    {
      const rr = mulberry32(977);
      ctLeaves.forEach((li, q) => {
        for (let e = 0; e < CT_EX; e++) {
          const a = rr() * TAU, r = Math.sqrt(rr());
          ctDots.setPos(q * CT_EX + e, ...NP(CT[li].x + 46 * r * Math.cos(a), 455 + 24 * r * Math.sin(a)));
          ctDots.setColor(q * CT_EX + e, ctCol[q], 0.9);
          ctDots.sizes[q * CT_EX + e] = 7;
        }
      });
      ctDots.commit();
    }
    netG.add(ctL.object, ctDots.object);

    // A native adaptation of Brill–Carlson Fig. 3: six stable states from one actual six-point
    // cyclic-coalescent run. Its private RNG streams never enter the dataset or the full demo tree.
    const exampleData = generate({ n: 2, d: 3, seeds: { graph: 14, embedding: 1, value: 1 }, p01: 1 / 3, demoSize: 6, view: 'pca' }, mulberry32);
    const exampleTree = clusterTree(exampleData, exampleData.demo), exN = exampleTree.s;
    const exampleGroups: number[][] = [Array.from({ length: exN }, (_, p) => p)];
    const exampleColor = new Uint8Array(2 * exN - 1);
    for (let p = 0; p < exN; p++) exampleColor[p] = p;
    for (let j = 0; j < exampleTree.m; j++) {
      const a = exampleTree.ca[j], b = exampleTree.cb[j], node = exN + j;
      exampleGroups.push(exampleGroups[j].filter((x) => x !== a && x !== b).concat(node));
      exampleColor[node] = exampleColor[exampleTree.cnt[a] >= exampleTree.cnt[b] ? a : b];
    }
    const EX_R = 53, EX_ARC = 70;
    const exampleAt = (p: number, r = EX_R): [number, number] => {
      const a = -Math.PI / 2 + (TAU * p) / exN;
      return [r * Math.cos(a), r * Math.sin(a)];
    };
    const examplePanels = exampleGroups.map((groups, step) => {
      const x = 150 + (step % 3) * 220, y = 310 + Math.floor(step / 3) * 200;
      const panel = new THREE.Group();
      panel.name = `percolation-coalescent-step-${step}`;
      panel.position.set(...NP(x, y));
      const dots = new PointField({ count: exN, size: 9, bokeh: 0, twinkle: 0 });
      const oldEdges = new LineField(Math.max(1, exampleTree.m), { width: 1.6 });
      const newEdge = new LineField(42, { width: 2.5 });
      for (let p = 0; p < exN; p++) { const [px, py] = exampleAt(p); dots.setPos(p, px, -py, 0); dots.setColor(p, INK, 0.95); }
      dots.commit();
      const arcs = cyclicRibbons(groups.map((node) => ({
        start: exampleTree.start[node], count: exampleTree.cnt[node], color: HUES[exampleColor[node]],
      })), exN, (q, r) => { const [x, y] = exampleAt(q, r); return [x, -y]; }, EX_ARC, 3);
      for (let j = 0; j < step; j++) {
        const a = exampleAt(exampleTree.eu[j]), b = exampleAt(exampleTree.ev[j]);
        const field = j === step - 1 ? newEdge : oldEdges;
        field.push([a[0], -a[1], 0], [b[0], -b[1], 0], j === step - 1 ? C.red : INK, j === step - 1 ? 1 : 0.7);
        if (j === step - 1) for (const p of [a, b]) newEdge.circle([p[0], -p[1], 0], 9, [1, 0, 0], [0, 1, 0], C.red, 1, 20);
      }
      for (const f of [oldEdges, newEdge]) f.commit();
      panel.add(arcs.object, oldEdges.object, newEdge.object, dots.object);
      netG.add(panel);
      return { panel, dots, arcs, oldEdges, newEdge, x, y, step };
    });

    // 8b. slides 3 and 4: the latent tree of the ring's cluster. On slide 3 a latent sits at the height of its merge
    // step; on slide 4 the same tree hangs from its root, every node at the height of its depth, and the cut runs
    // across it. The leaves are the ring's points in ring order.
    const TX0 = 470, TX1 = 1240, TYT = 196, TYB = 566; // the leaves' span; the top and the bottom of the tree
    const tX = new Float32Array(NN), tYm = new Float32Array(NN), tYd = new Float32Array(NN);
    const uM = (TYB - TYT) / (m + 0.5), uD = (TYB - TYT) / maxDd;
    for (let p = 0; p < s; p++) { tX[p] = TX0 + ((TX1 - TX0) * (qOf(p) + 0.5)) / s; tYm[p] = TYB; }
    for (let j = 0; j < m; j++) { tX[s + j] = (tX[T.ca[j]] + tX[T.cb[j]]) / 2; tYm[s + j] = TYB - (j + 1) * uM; }
    for (let node = 0; node < NN; node++) tYd[node] = TYT + T.depth[node] * uD;
    /** Height of a node: its merge step (h = 0, slide 3) or its depth (h = 1, slide 4). */
    const tY = (node: number, h: number) => lerp(tYm[node], tYd[node], h);
    const mStep = m <= 24 ? 5 : m <= 60 ? 20 : 30, dStep = maxDd <= 12 ? 2 : maxDd <= 30 ? 5 : 10, tickStep = s <= 24 ? 5 : s <= 64 ? 20 : 30;
    const mTicks: number[] = [], dTicks: number[] = [], qTicks: number[] = [];
    for (let v = 0; v <= m; v += mStep) mTicks.push(v);
    for (let v = 0; v <= maxDd; v += dStep) dTicks.push(v);
    for (let q = 0; q < s - tickStep * 0.3; q += tickStep) qTicks.push(q);
    const TAX = TX0 - 18, TQY = TYB + 12; // the vertical axis; the line under the leaves
    const tAxM = new LineField(1 + mTicks.length), tAxD = new LineField(1 + dTicks.length), tAxQ = new LineField(1 + qTicks.length);
    tAxM.push(NP(TAX, TYB), NP(TAX, TYB - m * uM), INK, AXK);
    for (const v of mTicks) tAxM.push(NP(TAX, TYB - v * uM), NP(TAX - 6, TYB - v * uM), INK, AXK);
    tAxD.push(NP(TAX, TYT), NP(TAX, TYB), INK, AXK);
    for (const v of dTicks) tAxD.push(NP(TAX, TYT + v * uD), NP(TAX - 6, TYT + v * uD), INK, AXK);
    tAxQ.push(NP(TX0 - 6, TQY), NP(TX1 + 6, TQY), INK, AXK);
    for (const q of qTicks) tAxQ.push(NP(TX0 + ((TX1 - TX0) * (q + 0.5)) / s, TQY), NP(TX0 + ((TX1 - TX0) * (q + 0.5)) / s, TQY + 6), INK, AXK);
    for (const L of [tAxM, tAxD, tAxQ]) L.commit();
    const tF = new LineField(3 * m, { width: 2 }); // per merge: two risers and the bracket
    tF.count = 3 * m;
    const tPts = new PointField({ count: NN, size: 1, bokeh: 0, twinkle: 0 });
    const T_LEAF = 8 * clamp(Math.sqrt(56 / s), 0.6, 1.2), T_LAT = 4.5; // dot sizes in pixels
    const tCut = new LineField(1, { width: 1.5 });
    tCut.count = 1;
    const tPick = new LineField(24, { width: 1.5 }); // the ring around the marked point's leaf
    netG.add(tAxM.object, tAxD.object, tAxQ.object, tF.object, tPts.object, tCut.object, tPick.object);

    // 8c. slide 5: what the network gets and what the ground truth holds, as a two-column box between the two panels' titles
    const BOX = { x0: 396, x1: 700, y0: 176, y1: 240 }, BOX_MID = (BOX.x0 + BOX.x1) / 2;
    const boxL = new LineField(6, { width: 1.5 });
    boxL.push(NP(BOX.x0, BOX.y0), NP(BOX.x1, BOX.y0), INK, 0.7);
    boxL.push(NP(BOX.x1, BOX.y0), NP(BOX.x1, BOX.y1), INK, 0.7);
    boxL.push(NP(BOX.x1, BOX.y1), NP(BOX.x0, BOX.y1), INK, 0.7);
    boxL.push(NP(BOX.x0, BOX.y1), NP(BOX.x0, BOX.y0), INK, 0.7);
    boxL.push(NP(BOX_MID, BOX.y0 + 10), NP(BOX_MID, BOX.y1 - 10), INK, 0.4);
    boxL.commit();
    netG.add(boxL.object);

    // ═══ 9. per-frame state ════════════════════════════════════════════════════════════════════
    const cur = new Float32Array(s * 3); // current group color of each point of the ring
    const flash = new Float32Array(NN);
    const posNow = new Float32Array(s * 3); // current position of each point of the ring (ring or inputs x)
    const _v = new THREE.Vector3(), _r = new THREE.Vector3(), _u = new THREE.Vector3(), _b = new THREE.Vector3();
    let halfH = 1, halfW = 1, sc = 1, scN = 1;
    let camNow: THREE.PerspectiveCamera | null = null; // the camera of the last frame, for the notes
    const frame = (cam: THREE.PerspectiveCamera) => {
      camNow = cam;
      cam.matrixWorld.extractBasis(_r, _u, _b);
      const frustumHalfH = Math.tan((cam.fov * Math.PI) / 360) * DP;
      halfW = frustumHalfH * Math.min(cam.aspect, ASPECT);
      halfH = halfW / ASPECT;
      sc = ((PX1 - PX0) * 2 * halfW) / PW;
      inset.position.copy(cam.position).addScaledVector(_b, -DP).addScaledVector(_r, (PX0 - 0.5) * 2 * halfW).addScaledVector(_u, -(PY0 - 0.455) * 2 * halfH - PH * sc);
      inset.quaternion.copy(cam.quaternion);
      inset.scale.setScalar(sc);
      scN = (2 * halfW) / 1600; // world units per frame pixel at the distance of the flat panels
      netG.position.copy(cam.position).addScaledVector(_b, -DP).addScaledVector(_r, (NX0 / 1600 - 0.5) * 2 * halfW).addScaledVector(_u, -(NY1 / 900 - 0.455) * 2 * halfH);
      netG.quaternion.copy(cam.quaternion);
      netG.scale.setScalar(scN);
    };
    /** World point of a position on the chart (panel units, origin at the lower-left corner of the frame). */
    const pt = (x: number, y: number): [number, number, number] => {
      _v.copy(inset.position).addScaledVector(_r, x * sc).addScaledVector(_u, y * sc);
      return [_v.x, _v.y, _v.z];
    };
    /** World point of a position on the flat figures (frame pixels, y down). */
    const npt = (x: number, y: number): [number, number, number] => {
      _v.copy(netG.position).addScaledVector(_r, (x - NX0) * scN).addScaledVector(_u, (NY1 - y) * scN);
      return [_v.x, _v.y, _v.z];
    };
    /** World point that projects to the screen fraction (fx from the left, fy from the top). */
    const scr = (cam: THREE.PerspectiveCamera, fx: number, fy: number): [number, number, number] => {
      _v.copy(cam.position).addScaledVector(_b, -DP).addScaledVector(_r, (fx - 0.5) * 2 * halfW).addScaledVector(_u, -(fy - 0.455) * 2 * halfH);
      return [_v.x, _v.y, _v.z];
    };
    // The same two corners drive the GPU mask and the hit region. The camera applies re-framing
    // to both, and the pointer's projection also includes the shared figure zoom and pan.
    const cloudTopLeft = new THREE.Vector3(), cloudBottomRight = new THREE.Vector3();
    const cloudScreen = new THREE.Vector4();
    const frameCloud = (t: number, cam: THREE.PerspectiveCamera) => {
      const { x0, y0, x1, y1 } = t < S4B ? DATA_WINDOW : CLUSTER_WINDOW;
      cloudTopLeft.set(...scr(cam, x0, y0));
      cloudBottomRight.set(...scr(cam, x1, y1));
      _v.copy(cloudTopLeft).project(cam);
      const left = 0.5 + _v.x / 2, top = 0.5 - _v.y / 2;
      _v.copy(cloudBottomRight).project(cam);
      cloudScreen.set(left, top, 0.5 + _v.x / 2, 0.5 - _v.y / 2);
      const feather = 0.013 * (cloudScreen.z - left) / (x1 - x0);
      for (const field of [arch, archL, clusterPts, clusterLines]) {
        field.uWindow.value.copy(cloudScreen);
        field.uFeather.value = feather;
      }
    };
    const cloudHit = (p: ProbeInput) => {
      const [x0, y0] = p.project(cloudTopLeft.x, cloudTopLeft.y, cloudTopLeft.z);
      const [x1, y1] = p.project(cloudBottomRight.x, cloudBottomRight.y, cloudBottomRight.z);
      return p.x > x0 && p.x < x1 && p.y > y0 && p.y < y1 ? { x0, y0, x1, y1 } : null;
    };
    let archSig = '';

    const mergesDone = (t: number) => { let k = 0; while (k < m && tau[k] <= t) k++; return k; };
    const hlA = (t: number) => sm(T_HA, T_HA + 0.7, t);
    const mixTo = (c: number[], to: RGB, f: number) => { c[0] += (to[0] - c[0]) * f; c[1] += (to[1] - c[1]) * f; c[2] += (to[2] - c[2]) * f; };
    const c3 = [0, 0, 0];
    const tag = (c: string, txt: string) => `<span style="color:${c}">${txt}</span>`;
    const TITLE = { color: CSS.white, align: 'left', size: 'title', dy: T_DY, dx: -12 } as const; // a chart's title, above its frame
    const SUB = { color: CSS.white, align: 'left', dy: N_DY, dx: -12 } as const; // the line under it

    // ── what is on screen when: one function per object, used by update() and by the notes ──
    const treeOf = (t: number) => (t > T_TREE0 - 0.05 ? 1 - sm(S2, S2 + 0.5, t) : 0); // 1 the tree of concepts
    const dataOf = (t: number) => win(t, S2 + 0.2, S3, 0.5); // 2 the dataset, with its title and legend
    const cloudOf = (t: number) => dataOf(t) + win(t, S4B + 0.6, S5, 0.6); // 2 and the second half of 4
    const ccdfOf = (t: number) => sm(T_CC0, T_CC0 + 0.8, t) * (1 - sm(S3, S3 + 0.5, t)); // 2 the cluster sizes
    const heroOf = (t: number) => sm(T_GROW1 - 0.6, T_GROW1 + 0.2, t); // 2: the largest cluster stands out
    const ringOf = (t: number) => sm(T_RING0 - 0.4, T_RING0 + 0.2, t) * (1 - sm(S4B, S4B + 0.6, t)) + sm(S5 + 1.0, S5 + 1.6, t) * (1 - sm(S6, S6 + 0.5, t)); // 3, 4 and 5 the ring's points
    const mergeTreeOf = (t: number) => sm(T_RING0 + 0.6, T_RING0 + 1.8, t) * (1 - sm(S5B, S5B + 0.5, t)); // 3, 4 and 5 the ring's latent tree (an inset on 4B)
    const insOf = (t: number) => sm(T_INS0, T_INS1, t) * (1 - sm(T_BACK0, T_BACK1, t)); // 0 the small tree at full size, 1 the inset (4B only)
    const backOf = (t: number) => sm(T_BACK0, T_BACK1, t); // 5: the small tree is back, shifted right by DX5
    const famOf = (t: number) => sm(S4B + 1.0, S4B + 1.8, t) * (1 - sm(S5, S5 + 0.5, t)); // 4 the largest cluster, cut
    const barsOf = (t: number) => (hs > 1 ? sm(S4B + 1.2, S4B + 2.1, t) * (1 - sm(S5, S5 + 0.5, t)) : 0); // 4 its latent tree as bars
    const walkOf = (t: number) => sm(S5 + 1.0, S5 + 1.6, t) * (1 - sm(S6, S6 + 0.5, t)); // 5 the ring's points at their inputs x
    const buildOf = (t: number) => sm(S5B + 0.4, S5B + 1.1, t) * (1 - sm(S6, S6 + 0.5, t)); // 5B how y is built
    const boxOf = (t: number) => sm(T_BOX, T_BOX + 0.6, t) * (1 - sm(S6, S6 + 0.5, t)); // 5B what the network gets
    const pickOf = (t: number) => sm(T_PICK, T_PICK + 0.5, t) * (1 - sm(S5, S5 + 0.5, t)) + sm(S5 + 1.2, S5 + 1.8, t) * (1 - sm(S6, S6 + 0.5, t)); // point P's ring

    // ── 3, 4 and 5: the ring's cluster, its latent tree, the cut, the inputs x ──
    /** The depth of the cut on slide 4 as it moves down, from 1 to kd. */
    const cutK = (t: number) => (kd <= 1 ? kd : 1 + (kd - 1) * ramp(T_KD0, T_KD1, t));
    const tc3 = [0, 0, 0];
    const drawDemo = (t: number, api: FrameApi) => {
      const cam = api.camera;
      const ringA = ringOf(t), treeIn = mergeTreeOf(t);
      const teaching = teachingStage(t), teachingShift = teachingTreeShift(t);
      const ringDisplay = teaching ? 0 : ringA;
      const mergeTime = teaching ? T_MERGE1 + 2 : t;
      const vis = ringDisplay > 0.001, treeOn = treeIn > 0.001;
      const hang = sm(T_HANG0, T_HANG1, t), cutOn = sm(T_KD0 - 0.4, T_KD0 + 0.1, t);
      const wm = sm(T_WALK0, T_WALK1, t); // 0 on the ring, 1 at the inputs x
      const onRing = wm < 0.001;
      // the small tree moves into its inset: frame pixels of the full-size tree → the inset's box
      const ins = insOf(t), axA = 1 - ins;
      const isx = (INS.x1 - INS.x0) / (TX1 - TX0), isy = (INS.y1 - INS.y0) / (TYB - TYT);
      const dx5 = DX5 * backOf(t); // slide 5: the tree stands right of the inputs x
      const tmx = (x: number) => lerp(x, INS.x0 + (x - TX0) * isx, ins) + dx5 + teachingShift, tmy = (y: number) => lerp(y, INS.y0 + (y - TYT) * isy, ins);
      const TN = (x: number, y: number) => NP(tmx(x), tmy(y)), tnpt = (x: number, y: number) => npt(tmx(x), tmy(y));
      const tp = (x: number, y: number) => npt(x + dx5 + teachingShift, y); // a label of the full-size tree
      const yc = sm(T_YC0, T_YC1, t); // 0 subcluster colors, 1 each point's own y
      nodes.object.visible = vis;
      showL(chordF, vis);
      showL(tAxM, treeOn && hang < 0.999 && axA > 0.001);
      showL(tAxD, treeOn && hang > 0.001 && axA > 0.001);
      showL(tAxQ, treeOn && axA > 0.001);
      showL(tF, treeOn);
      tPts.object.visible = treeOn;
      showL(tCut, treeOn && cutOn > 0.001);
      const aPick = teaching ? 0 : pickOf(t);
      showL(pickRing, vis && aPick > 0.01);
      showL(tPick, treeOn && aPick > 0.01);
      // latent A's arc: from the moment A is named until the ring leaves (its points stay subcluster A after the cut)
      const aAArc = vis && onRing ? hlA(t) * ringA : 0;
      selectedRibbon.object.visible = aAArc > 0.001 && cutOn < 0.999;
      selectedRibbon.uIntensity.value = aAArc * (1 - cutOn);
      const kf = cutK(t), k0 = Math.floor(kf), k1 = Math.min(kd, k0 + 1), kq = sm(0.2, 0.8, kf - k0);
      cutRibbons.forEach((bands, k) => {
        const alpha = cutOn * (k === k0 ? 1 - kq : k === k1 ? kq : 0);
        bands.object.visible = vis && onRing && t >= S4 && alpha > 0.001;
        bands.uIntensity.value = ringDisplay * alpha;
      });
      if (!vis && !treeOn) return;
      nodes.uOpacity.value = ringA;
      chordF.uIntensity.value = ringA;
      for (const f of [tAxM, tAxD, tAxQ]) f.object.position.x = dx5 + teachingShift; // the axes move with the tree on slide 5
      tAxM.uIntensity.value = treeIn * (1 - hang) * axA;
      tAxD.uIntensity.value = treeIn * hang * axA;
      tAxQ.uIntensity.value = treeIn * axA;
      tF.uIntensity.value = treeIn;
      tPts.uOpacity.value = treeIn;
      tCut.uIntensity.value = treeIn * cutOn;
      const hA = teaching ? 0 : hlA(t) * (1 - cutOn);
      /** The color of a node at the cut: blended between two depths while the cut moves. */
      const cutC = (node: number, out: number[]) => {
        const a = cutRGB(cutIdx[k0][node]), b = cutRGB(cutIdx[k1][node]);
        out[0] = lerp(a[0], b[0], kq); out[1] = lerp(a[1], b[1], kq); out[2] = lerp(a[2], b[2], kq);
      };

      // group colors and emphasis from the merge history
      for (let p = 0; p < s; p++) { const c = palRing[colIdx[p]]; cur[p * 3] = c[0]; cur[p * 3 + 1] = c[1]; cur[p * 3 + 2] = c[2]; }
      flash.fill(0);
      for (let j = 0; j < m; j++) {
        const q = (mergeTime - tau[j]) / dur[j];
        if (q <= 0) continue;
        const rcj = sm(0.45, 0.85, q);
        if (rcj > 0) {
          const c = palRing[colIdx[s + j]], list = chg[j];
          for (let i = 0; i < list.length; i++) {
            const o = list[i] * 3;
            cur[o] += (c[0] - cur[o]) * rcj; cur[o + 1] += (c[1] - cur[o + 1]) * rcj; cur[o + 2] += (c[2] - cur[o + 2]) * rcj;
          }
        }
        if (q < 1.4 && j !== jS) {
          flash[T.eu[j]] += sm(0, 0.1, q) * (1 - sm(0.45, 0.9, q));
          flash[T.ev[j]] += sm(0.4, 0.55, q) * (1 - sm(0.7, 1.2, q));
          flash[s + j] += sm(0.3, 0.5, q) * (1 - sm(0.9, 1.4, q));
        }
      }
      // the points (on the ring, or moving to their inputs x) and the leaves of the tree
      const pc = nodes.colors, np = nodes.positions, nsz = nodes.sizes;
      const dc = tPts.colors, dsz = tPts.sizes;
      for (let p = 0; p < s; p++) {
        const o = p * 3;
        c3[0] = cur[o]; c3[1] = cur[o + 1]; c3[2] = cur[o + 2];
        let k = 1;
        if (hA > 0) { if (inA[p]) mixTo(c3, ACC, hA); else k = lerp(k, 0.4, hA); }
        if (cutOn > 0) { cutC(p, tc3); mixTo(c3, tc3 as unknown as RGB, cutOn); }
        const a0 = T_RING0 + (2.1 * qOf(p)) / s, ap = sm(a0, a0 + 0.6, mergeTime), fl = Math.min(1, flash[p]);
        posNow[o] = lerp(ringP[o], walkP[o], wm); posNow[o + 1] = lerp(ringP[o + 1], walkP[o + 1], wm); posNow[o + 2] = 0;
        np[o] = posNow[o]; np[o + 1] = posNow[o + 1]; np[o + 2] = 0;
        tPts.setPos(p, ...TN(tX[p], tY(p, hang)));
        dc[o] = c3[0] * k * ap; dc[o + 1] = c3[1] * k * ap; dc[o + 2] = c3[2] * k * ap;
        dsz[p] = T_LEAF * (1 + 0.5 * fl) * lerp(1, 0.5, ins);
        // slide 5 ends with each point in the color of its own target y (the tree is gone by then)
        if (yc > 0) { c3[0] = lerp(c3[0], yCol[o], yc); c3[1] = lerp(c3[1], yCol[o + 1], yc); c3[2] = lerp(c3[2], yCol[o + 2], yc); k = lerp(k, 1, yc); }
        pc[o] = c3[0] * k * ap; pc[o + 1] = c3[1] * k * ap; pc[o + 2] = c3[2] * k * ap;
        nsz[p] = nodeSize * (1 + 0.5 * fl);
      }

      // the brackets of the tree, and the line each merge adds between two points
      const kLine = lerp(0.6 * (1 - 0.5 * cutOn), 0.55, wm);
      for (let j = 0; j < m; j++) {
        const node = s + j, q = (mergeTime - tau[j]) / dur[j];
        const a = T.ca[j], b = T.cb[j];
        c3[0] = INK[0]; c3[1] = INK[1]; c3[2] = INK[2];
        let k = 0.75;
        if (hA > 0) { if (inA[node]) { mixTo(c3, ACC, hA); k = lerp(k, 1.1, hA); } else k = lerp(k, 0.3, hA); }
        if (cutOn > 0) { cutC(node, tc3); mixTo(c3, tc3 as unknown as RGB, cutOn); k = lerp(k, cutIdx[kd][node] === -2 ? 0.75 : 1, cutOn); }
        const kNew = k + 0.45 * Math.min(1, flash[node]);
        const g1 = q <= 0 ? 0 : sm(0.12, 0.55, q), uo = T.eu[j] * 3, vo = T.ev[j] * 3, ke = g1 > 0 ? kLine : 0;
        seg(chordF, j, posNow[uo], posNow[uo + 1], 0, lerp(posNow[uo], posNow[vo], g1), lerp(posNow[uo + 1], posNow[vo + 1], g1), 0, INK[0] * ke, INK[1] * ke, INK[2] * ke);
        // two risers from the groups joined, then the bracket from group a to the next group b
        const g2 = q <= 0 ? 0 : sm(0.3, 0.62, q), g3 = q <= 0 ? 0 : sm(0.58, 0.9, q);
        const yj = tY(node, hang), ya = tY(a, hang), yb = tY(b, hang), k2 = g2 > 0 ? kNew : 0, k3 = g3 > 0 ? kNew : 0;
        tF.seg(3 * j, TN(tX[a], ya), TN(tX[a], lerp(ya, yj, g2)), c3 as unknown as RGB, k2);
        tF.seg(3 * j + 1, TN(tX[b], yb), TN(tX[b], lerp(yb, yj, g2)), c3 as unknown as RGB, k2);
        tF.seg(3 * j + 2, TN(tX[a], yj), TN(lerp(tX[a], tX[b], g3), yj), c3 as unknown as RGB, k3);
        const kl = (q <= 0 ? 0 : sm(0.8, 1.0, q)) * (k + 0.2);
        tPts.setPos(node, ...TN(tX[node], yj));
        dc[node * 3] = c3[0] * kl; dc[node * 3 + 1] = c3[1] * kl; dc[node * 3 + 2] = c3[2] * kl;
        dsz[node] = T_LAT * (1 + 0.5 * Math.min(1, flash[node]) + 0.9 * (node === A ? Math.max(hA, cutOn) : 0)) * lerp(1, 0.6, ins);
      }
      // the cut: a rule across the tree between the rows kf − 1 and kf
      const yCut = TYT + (kf - 0.5) * uD;
      tCut.seg(0, TN(TX0 - 10, yCut), TN(TX1 + 10, yCut), INK, 0.9);
      tCut.commit();

      chordF.commit();
      nodes.commit();
      tF.commit();
      tPts.commit();
      // the marked point: a ring in the plane of the screen on the cluster, a ring on its leaf
      if (aPick > 0.01) {
        const o = pk * 3, dz = (posNow[o] - cam.position.x) * -_b.x + (posNow[o + 1] - cam.position.y) * -_b.y + (0 - cam.position.z) * -_b.z;
        const r = 9 * scN * (dz / DP); // nine frame pixels at that depth
        for (let i = 0; i < RING_N; i++) {
          const a0 = (i / RING_N) * TAU, a1 = ((i + 1) / RING_N) * TAU;
          seg(pickRing, i,
            posNow[o] + r * (Math.cos(a0) * _r.x + Math.sin(a0) * _u.x), posNow[o + 1] + r * (Math.cos(a0) * _r.y + Math.sin(a0) * _u.y), r * (Math.cos(a0) * _r.z + Math.sin(a0) * _u.z),
            posNow[o] + r * (Math.cos(a1) * _r.x + Math.sin(a1) * _u.x), posNow[o + 1] + r * (Math.cos(a1) * _r.y + Math.sin(a1) * _u.y), r * (Math.cos(a1) * _r.z + Math.sin(a1) * _u.z),
            INK[0], INK[1], INK[2]);
        }
        pickRing.uIntensity.value = aPick * ringA;
        pickRing.commit();
        if (treeOn) {
          tPick.clear();
          tPick.circle(TN(tX[pk], tY(pk, hang)), lerp(9, 5, ins), [1, 0, 0], [0, 1, 0], INK, 1, 24);
          tPick.uIntensity.value = aPick * treeIn;
          tPick.commit();
        }
      }

      // ── labels ──
      const o = { color: CSS.white } as const;
      const preWalk = 1 - sm(0.1, 0.45, wm), postWalk = sm(0.55, 0.9, wm);
      const aRingT = ringDisplay * preWalk, aWalkT = walkOf(t) * postWalk; // titles of the ring, then of the inputs x
      api.label('ta', scr(cam, TITLE_X, TITLE_Y), 'Example cluster', { ...o, alpha: aRingT, align: 'left', size: 'title', dy: 8, help: `<p>The ring arranges the ${s} points in the order used during construction. Their input coordinates come from the branching random walk.</p>` });
      api.label('tw', scr(cam, TITLE_X, TITLE_Y), 'Input coordinates $x$', { ...o, alpha: aWalkT, align: 'left', size: 'title', dy: 8, help: `<p>The branching random walk places the ${s} points in input space. The links are the same edges that connected the points on the ring.</p>` });
      const pk1 = sm(T_PICK, T_PICK + 0.5, t), preCut = 1 - sm(0.1, 0.45, cutOn), postCut = sm(0.55, 0.9, cutOn);
      const ringKey = ' · ring = P';
      api.label('la', scr(cam, TITLE_X, LEG_Y), `${tag(ACC_CSS, 'teal')} = points under A · line = link${pk1 > 0.5 ? ringKey : ''}`, { ...o, alpha: ringDisplay * 0.9 * preCut, align: 'left', dy: -7 });
      api.label('la2', scr(cam, TITLE_X, LEG_Y), `color and arc = subcluster<small>gray = above the cut${ringKey}</small>`, { ...o, alpha: ringA * 0.9 * postCut * preWalk, align: 'left', dy: -7 });
      api.label('lw', scr(cam, TITLE_X, LEG_Y), `colored by subcluster<small>line = link${ringKey}</small>`, { ...o, alpha: ringA * 0.9 * postWalk * (1 - sm(0.1, 0.45, yc)), align: 'left', dy: -7 });
      api.label('lwy', scr(cam, TITLE_X, LEG_Y), `target $y$ · ${tag(CSS.violet, `${dot(1.2)} above 0`)} · ${tag(CSS.white, `${dot(-1.2)} below 0`)}<small>line = link${ringKey}</small>`, { ...o, alpha: ringA * 0.9 * postWalk * sm(0.55, 0.9, yc), align: 'left', dy: -7 });
      // point P, named wherever it is ringed
      if (aPick > 0.01 && vis) {
        const px = posNow[pk * 3], py = posNow[pk * 3 + 1];
        api.label('pkR', [px, py, 0], 'P', { ...o, alpha: aPick * ringA, align: onRing && px < 0 ? 'right' : 'left', size: 'md', dx: onRing && px < 0 ? -4 : 4, dy: -14 });
      }
      // P's name under its leaf, below the axis line unless a tick label stands there
      const pDy = qTicks.some((q) => Math.abs(q - qOf(pk)) < 3) ? 15 : 30;
      if (aPick > 0.01 && treeOn) api.label('pkT', tnpt(tX[pk], tY(pk, hang)), 'P', { ...o, alpha: aPick * treeIn, align: ins > 0.5 ? 'left' : 'below', size: ins > 0.5 ? 'sm' : 'md', dx: 0, dy: ins > 0.5 ? 9 : pDy - 12, plate: true });
      if (treeOn && axA > 0.001) {
        const TTX = TAX - 4, mergeLabels = 1 - sm(0.1, 0.45, hang), depthLabels = sm(0.55, 0.9, hang);
        api.label('tb', tp(TTX, TYT - 58), 'Example latent tree', { ...o, alpha: treeIn * mergeLabels * axA, align: 'left', size: 'title', help: `<p>This ${s}-point cluster is built separately from the construction example. Each bracket marks a merge and the latent it creates over the points it joins. Height records the merge step.</p>` });
        api.label('tb2', tp(TTX, TYT - 58), 'Latent tree by depth', { ...o, alpha: treeIn * depthLabels * axA, align: 'left', size: 'title', help: '<p>A node’s depth is the number of latents above it. Each latent at the cut defines one subcluster below it. Gray points belong to branches that end above the cut and have no subcluster there.</p>' });
        api.label('byn', tp(TTX, TYT - 30), 'merge step', { ...o, alpha: treeIn * 0.9 * mergeLabels * axA, align: 'left' });
        api.label('byn2', tp(TTX, TYT - 30), 'depth', { ...o, alpha: treeIn * 0.9 * depthLabels * axA, align: 'left' });
        mTicks.forEach((v) => api.label('bm' + v, tp(TAX - 9, TYB - v * uM), String(v), { ...o, alpha: treeIn * 0.9 * mergeLabels * axA, align: 'right' }));
        dTicks.forEach((v) => api.label('bd' + v, tp(TAX - 9, TYT + v * uD), String(v), { ...o, alpha: treeIn * 0.9 * depthLabels * axA, align: 'right' }));
        qTicks.forEach((q) => api.label('bq' + q, tp(TX0 + ((TX1 - TX0) * (q + 0.5)) / s, TQY + 8), String(q), { ...o, alpha: treeIn * 0.9 * axA, align: 'below' }));
        api.label('bxn', tp((TX0 + TX1) / 2, TQY + 8), 'points in ring order', { ...o, alpha: treeIn * 0.9 * axA, align: 'below' });
      }
      // the inset: the small tree, cut, kept while the largest cluster is cut the same way
      if (treeOn && ins > 0.001)
        api.label('ins-t', npt((INS.x0 + INS.x1) / 2, INS.y1 + 14), `Example tree<small>${tag(ACC_CSS, 'teal')} = subcluster A · ring = P</small>`, { ...o, alpha: treeIn * ins, align: 'below' });
      // latent A, then its subcluster
      const aFam = sm(T_FA, T_FA + 0.5, t);
      if (!teaching && hlA(t) > 0.01 && treeOn && axA > 0.001) {
        // the name stands on the side away from the center, over the empty corner above A's own branches
        const away = tX[A] > (TX0 + TX1) / 2 ? 'left' : 'right', room = away === 'right' ? tX[A] - TAX > 230 : TX1 + 60 - tX[A] > 230;
        const at = tp(tX[A], tY(A, hang)), al = room ? away : away === 'right' ? 'left' : 'right';
        api.label('tagA', at, `latent A: ${T.cnt[A]} points under it`, { color: ACC_CSS, alpha: hlA(t) * treeIn * (1 - aFam) * axA, align: al, size: 'md', dy: -12, plate: true });
        // subcluster A's name stands in the empty corner above its side of the tree, under the cut
        const aLeft = (qOf(T.start[A]) + T.cnt[A] / 2) / s < 0.5;
        api.label('tagA3', tp(aLeft ? TX0 - 6 : TX1 + 6, TYT + (kd + 1.1) * uD), `subcluster A:<br>the ${T.cnt[A]} points under latent A`, { color: ACC_CSS, alpha: treeIn * aFam * axA, align: aLeft ? 'left' : 'right', size: 'md', plate: true });
      }
      if (cutOn > 0.01 && treeOn && axA > 0.001) api.label('cutk', tp(TX1 + 14, yCut), `the cut: depth $k = ${Math.round(kf)}$`, { ...o, alpha: treeIn * cutOn * axA * (1 - backOf(t)), align: 'left', size: 'md', dy: -7 });
    };

    const drawCoalescentExample = (t: number, api: FrameApi) => {
      const on = teachingStage(t), intro = sm(S3 + 0.15, S3 + 0.7, t);
      for (const p of examplePanels) {
        p.panel.visible = on;
        if (!on) continue;
        const reveal = S3 + 0.4 + p.step * 2.1;
        const alpha = intro * sm(reveal, reveal + 0.55, t);
        p.dots.uOpacity.value = alpha;
        for (const f of [p.arcs, p.oldEdges, p.newEdge]) f.uIntensity.value = alpha;
        api.label(`example-step-${p.step}`, npt(p.x, p.y + 82), p.step === 0 ? 'Start' : `Merge ${p.step}`, { color: CSS.white, alpha, align: 'middle', size: 'md' });
      }
      if (!on) return;
      api.label('example-title', npt(50, 210), 'Cluster construction', {
        color: CSS.white, alpha: intro * sm(S3 + 0.5, S3 + 0.9, t), align: 'left', size: 'title',
        help: '<p>Successive states of a six-point cluster. The arrows follow the order of groups around the ring. Red marks the new link and its endpoints. The larger example beside it is a separate cluster built by the same merge rule.</p>',
      });
      api.label('example-key', npt(50, 635), `dot = one point · arc = one group · ${tag(CSS.red, 'red')} = new link`, { color: CSS.white, alpha: intro, align: 'left' });
    };

    // ── 2 and 4: the dataset, then its largest cluster cut into subclusters ──
    const drawArch = (t: number) => {
      const cloud = cloudOf(t);
      const on = cloud > 0.001;
      const separateCluster = clusterSlide(t) && clusterViewChanged();
      arch.object.visible = on;
      showL(archL, on);
      clusterPts.object.visible = on && separateCluster;
      showL(clusterLines, on && separateCluster);
      if (!on) { archSig = ''; return; }
      arch.uOpacity.value = clusterPts.uOpacity.value = cloud;
      archL.uIntensity.value = clusterLines.uIntensity.value = cloud;
      // on slide 2 the clusters appear inside a ball that grows from the center of the dataset
      const Rv = t < S3 + 1 ? lerp(-2.5, R3 + 4, sm(T_GROW0, T_GROW1, t)) : 1e6;
      const cut = t >= S4B ? 1 : 0; // slide 4: the other clusters faint, the largest one in its subclusters
      const kf = notesCutK() ?? kAnim(t);
      const hl = cut ? 0 : heroOf(t); // slide 2: the largest cluster in the team's ink, the others fainter
      const sig = `${Rv.toFixed(3)}|${cut}|${kf.toFixed(3)}|${hl.toFixed(3)}|${separateCluster}`;
      if (sig === archSig) return;
      archSig = sig;
      for (let c = 0; c < nClusters; c++) {
        const a = (Rv - Math.hypot(orgW[c * 3] - BLOB[0], orgW[c * 3 + 1] - BLOB[1], orgW[c * 3 + 2] - BLOB[2])) / 2.5;
        rc[c] = a <= 0 ? 0 : a >= 1 ? 1 : a * a * (3 - 2 * a);
      }
      rc[demo] = 0; // the ring's cluster is the film's illustration and not part of the dataset
      const pc = arch.colors;
      const dimF = (1 - 0.88 * cut) * lerp(1, OTHERS_K, hl);
      for (let i = 0; i < n; i++) {
        const o = i * 3, f = rc[cl[i]] * dimF;
        pc[o] = baseCol[o] * f; pc[o + 1] = baseCol[o + 1] * f; pc[o + 2] = baseCol[o + 2] * f;
      }
      // the largest cluster: in the team's ink on slide 2, in subcluster colors at cut depth k on slide 4
      const k0 = Math.min(kCap, Math.floor(kf)), k1 = Math.min(kCap, k0 + 1), kq = sm(0.15, 0.85, kf - k0);
      const aH = rc[hero];
      {
        const ld = TH.depth, fl = FT.leaf;
        for (let p = 0; p < hs; p++) {
          const o = (ho + p) * 3;
          if (!cut) { for (let e = 0; e < 3; e++) pc[o + e] = lerp(baseCol[o + e], HERO_COL[e], hl) * aH; continue; }
          const c0 = ld[p] < k0 ? ISO : HUES[fl[k0 * hs + p]], c1 = ld[p] < k1 ? ISO : HUES[fl[k1 * hs + p]];
          const g0 = ld[p] < k0 ? 1 : FAM_K, g1 = ld[p] < k1 ? 1 : FAM_K;
          for (let e = 0; e < 3; e++) pc[o + e] = lerp(c0[e] * g0, c1[e] * g1, kq) * aH;
        }
      }
      // edges: neutral ink scaled by the visibility of their cluster
      const lc = archL.colors;
      for (let e = 0; e < nE; e++) {
        const L = eL[e], u = edgeU[L], i = e * 6, f = ef[e] * rc[cl[u]] * dimF;
        lc[i] = lc[i + 3] = INK[0] * f; lc[i + 1] = lc[i + 4] = INK[1] * f; lc[i + 2] = lc[i + 5] = INK[2] * f;
      }
      if (hs > 1) {
        // the largest cluster's own edges come first in the list. Cut, an edge whose latent lies above the cut carries
        // no ink; any other edge takes its subcluster's color.
        const fe = FT.edge;
        for (let j = 0; j < hm; j++) {
          const i = j * 6;
          const i0 = fe[k0 * hm + j], i1 = fe[k1 * hm + j];
          const c0 = i0 === 255 ? CUT : HUES[i0], c1 = i1 === 255 ? CUT : HUES[i1];
          for (let e = 0; e < 3; e++) {
            const v = cut ? lerp(c0[e], c1[e], kq) * FAM_EDGE * aH : lerp(INK[e] * ef[0], HERO_EDGE[e], hl) * aH;
            lc[i + e] = v; lc[i + 3 + e] = v;
          }
        }
      }
      if (separateCluster) {
        clusterPts.colors.set(pc.subarray(ho * 3, (ho + hs) * 3));
        pc.fill(0, ho * 3, (ho + hs) * 3);
        clusterLines.colors.set(lc.subarray(0, hm * 6));
        lc.fill(0, 0, hm * 6);
        clusterPts.commit(false, true, false);
        clusterLines.commit();
      }
      arch.commit(false, true, false);
      archL.commit();
    };

    // ── 2, 4 and 5: the charts ──
    const sgn = (v: number) => (v > 0 ? '+' + v : String(v).replace('-', '−'));
    /** A share as a tick label: 10^v as a percentage. */
    const pctTick = (v: number) => { const p = 100 * Math.pow(10, v); return `${p >= 1 ? Math.round(p) : p.toFixed(Math.ceil(-Math.log10(p) - 1e-9))}%`; };
    const kAt = (t: number) => Math.min(kCap, Math.round(notesCutK() ?? kAnim(t)));
    const drawCharts = (t: number, api: FrameApi) => {
      const o = { color: CSS.white } as const;
      // 2: the cluster sizes
      const aC = ccdfOf(t);
      const onC = aC > 0.001;
      if (onC) {
        ccAx.uIntensity.value = aC;
        ccPts.uOpacity.value = aC;
        ccPts.drawCount = Math.max(1, Math.ceil(ccIdx.length * sm(T_CC0 + 0.4, T_CC0 + 1.8, t)));
        ccCurve.uIntensity.value = aC;
        ccCurve.count = Math.ceil(curveSegs * sm(T_CC0 + 1.4, T_CC0 + 2.3, t));
        api.label('td', pt(0, PH), 'Cluster sizes', { ...TITLE, alpha: aC, help: '<p>Each dot gives the share of generated clusters at least that large. The line is the long-run prediction for the generator’s cluster-growth rule.</p>' });
        api.label('cc-yn', pt(0, PH), 'share of clusters at least this large', { ...SUB, alpha: aC * 0.9 });
        xDec.forEach((v) => api.label('ccx' + v, pt(ccX(v), -0.07), int(Math.pow(10, v)), { ...o, alpha: aC * 0.9, align: 'center', dy: 32 }));
        yDec.forEach((v) => api.label('ccy' + v, pt(-0.07, ccY(v)), pctTick(v), { ...o, alpha: aC * 0.9, align: 'right' }));
        api.label('cc-xn', pt(PW / 2, -0.07), 'size of the cluster (points)', { ...o, alpha: aC * 0.9, align: 'center', dy: 52 });
        // the key stands in the empty upper right, above the theory's line, between the two top gridlines
        api.label('cc-th', pt(0.3 * PW, ccY(-0.5)),
          `${tag(CSS.violet, '●')} generated data<br>${tag(CSS.slate, '—')} generator prediction`,
          { ...o, alpha: aC * sm(T_CC0 + 1.6, T_CC0 + 2.4, t), align: 'left', plate: true });
      }
      showL(ccAx, onC);
      showL(ccCurve, onC);
      ccPts.object.visible = onC;

      // 4: the latent tree of the largest cluster as bars
      const aI = barsOf(t);
      const onI = aI > 0.001;
      if (onI) {
        const kf = notesCutK() ?? kAnim(t);
        const kr = clamp(kf, kTop, kBot);
        icBars.uIntensity.value = aI;
        icGhost.uIntensity.value = aI;
        icAx.uIntensity.value = aI;
        icCur.uIntensity.value = aI;
        icBars.count = rowEnd[Math.min(kBot, Math.floor(kr)) + 1];
        const rowH = PH / nRows, yc = icY(kr) + 0.5 * rowH; // the cut runs above the subclusters' row, as on the small tree
        seg(icCur, 0, -0.07, yc, 0, IW + 0.08, yc, 0, INK[0], INK[1], INK[2]); // across the chart, with a tick on its axis
        icCur.commit();
        api.label('tf', pt(0, PH), 'Latent tree of the largest cluster', { ...TITLE, alpha: aI, dy: T_DY - 12, help: '<p>Each bar is one latent, with width proportional to the points under it. Its children sit below it. A row is one depth, starting at zero at the root. The cut selects one bar per subcluster, colored to match the cloud.</p>' });
        icTicks.forEach((v, q) => api.label('ick' + v, pt(-0.07, icY(v)), q === 0 ? `depth ${v}` : String(v), { ...o, alpha: aI * 0.9 * sm(2.2, 3.8, Math.abs(icY(v) - yc - 0.3 * rowH) / rowH), align: 'right' }));
        api.label('ic-x0', pt(0, -0.07), '0', { ...o, alpha: aI * 0.9, align: 'center', dy: 32 });
        api.label('ic-x1', pt(IW, -0.07), int(hs), { ...o, alpha: aI * 0.9, align: 'center', dy: 32 });
        api.label('ic-xn', pt(IW / 2, -0.07), 'points of the cluster in ring order', { ...o, alpha: aI * 0.9, align: 'center', dy: 52 });
        api.label('ic-cur', pt(IW + 0.08, yc), `the cut: depth $k = ${kAt(t)}$`, { ...o, alpha: aI, align: 'left', size: 'md', dx: 6 });
        api.label('ic-kc', pt(-0.07, yc), String(kAt(t)), { ...o, alpha: aI, align: 'right', size: 'md' }); // the cut's depth on the axis
      }
      showL(icAx, onI);
      showL(icGhost, onI);
      showL(icBars, onI);
      showL(icCur, onI);

      // 5: the path sum of the marked point
      const aH = buildOf(t);
      const onH = aH > 0.001;
      if (onH) {
        const nShow = Math.min(nbz, Math.floor(nbz * ramp(T_BLD0, T_BLD1, t) + 1e-6));
        const aY = sm(T_BLD1 + 0.2, T_BLD1 + 0.8, t);
        bldAx.uIntensity.value = aH;
        bldBars.uIntensity.value = aH;
        bldRun.uIntensity.value = aH;
        bldY.uIntensity.value = aH * aY;
        bldBars.count = nShow;
        bldRun.count = Math.max(0, 2 * nShow - 1);
        bldAx.count = aY > 0.01 ? bldAxAll : bldAxN;
        api.label('th', pt(0, PH), 'Target $y$ for point P', { ...TITLE, alpha: aH, dy: T_DY - 12, help: '<p>The bars are the values $z$ along P’s path from the root to the point itself. The line is their running sum, divided by the square root of the number of values on the full path.</p>' });
        api.label('h-yn', pt(0, PH), 'bars = $z$ · line = scaled sum', { ...SUB, alpha: aH * 0.9, dy: N_DY - 7 });
        for (const v of zTicks) api.label('hz' + v, pt(-0.07, bY(v)), sgn(v), { ...o, alpha: aH * 0.9, align: 'right' });
        api.label('h-x0', pt(bX(0), -0.07), '0', { ...o, alpha: aH * 0.9, align: 'center', dy: 32 });
        api.label('h-x1', pt(bX(last), -0.07), String(pickD), { ...o, alpha: aH * 0.9, align: 'center', dy: 32 });
        api.label('h-xy', pt(BYX, -0.07), '$y$', { ...o, alpha: aH * 0.9, align: 'center', dy: 32 });
        api.label('h-xn', pt(PW / 2, -0.07), 'depth on the path', { ...o, alpha: aH * 0.9, align: 'center', dy: 52 });
        const aA = aBar >= 0 && aBar < nShow ? aH * sm(T_BLD1 + 0.6, T_BLD1 + 1.2, t) : 0;
        const aOwn = nShow >= nbz ? aH * sm(T_BLD1 + 0.3, T_BLD1 + 0.9, t) : 0;
        showL(bldA, onH && aBar >= 0 && aBar < nShow);
        showL(bldA_L, aA > 0.001);
        showL(bldOwnL, aOwn > 0.001);
        bldA.uIntensity.value = aH;
        bldA_L.uIntensity.value = aA;
        bldOwnL.uIntensity.value = aOwn;
        if (aBar >= 0 && aBar < nShow) {
          api.label('h-a', pt(bX(aBar), bY(A_LAB)), `$z$ of latent A<small>shared by every point of subcluster A</small>`, { color: ACC_CSS, alpha: aA, align: 'left', dx: -8, dy: 2, size: 'md', plate: true });
          if (aBar > 0 && Math.abs(bX(aBar) - bX(0)) > 22 * UPX) api.label('h-xa', pt(bX(aBar), -0.07), String(aBar), { color: ACC_CSS, alpha: aH * 0.9, align: 'center', dy: 32 });
        }
        api.label('h-own', pt(bX(last), bY(OWN_LAB)), "P's own value", { ...o, alpha: aOwn, align: 'right', dx: 10, size: 'md', plate: true });
        api.label('h-y', pt(BYX, bY(pickY)), `$y = ${neg(f2(pickY))}$<small>their sum, scaled</small>`, { ...o, alpha: aH * aY, align: pickY >= 0 ? 'center' : 'below', size: 'md', dy: pickY >= 0 ? 0 : 6 });
      }
      showL(bldAx, onH);
      showL(bldBars, onH);
      showL(bldRun, onH);
      showL(bldY, onH);
      if (!onH) { showL(bldA, false); showL(bldA_L, false); showL(bldOwnL, false); }
    };

    // ── 1 and 5: the flat figures of the data slides ──
    const drawFlat = (t: number, api: FrameApi) => {
      const o = { color: CSS.white } as const;
      // 1: the tree of concepts
      const aT = treeOf(t);
      const onT = aT > 0.001;
      showL(ctL, onT);
      ctDots.object.visible = onT;
      if (onT) {
        ctL.uIntensity.value = aT;
        // the examples come in with their class's word, a little after it
        ctDots.uOpacity.value = aT;
        ctDots.drawCount = ctLeaves.length * CT_EX;
        const aEx = sm(ctAt(3) + 0.3, ctAt(3) + 1.1, t);
        for (let i = 0; i < ctLeaves.length * CT_EX; i++) ctDots.sizes[i] = 7 * aEx;
        ctDots.commit(false, false, true);
        CT.forEach((c, i) => {
          const a0 = ctAt(i);
          if (c.up >= 0) {
            const p = CT[c.up], g = sm(a0 - 0.45, a0 + 0.05, t); // the line grows down from the parent's word
            const x0 = p.x, y0 = p.y + 17, x1 = c.x, y1 = c.y - 17;
            ctL.seg(i - 1, NP(x0, y0), NP(lerp(x0, x1, g), lerp(y0, y1, g)), INK, g > 0.001 ? 0.75 : 0);
          }
          api.label('ct' + i, npt(c.x, c.y), c.w, { ...o, alpha: aT * sm(a0, a0 + 0.4, t), align: 'middle', size: 'lg' });
        });
        ctL.commit();
        const aNote = aT * sm(T_TREE0 + 2.2, T_TREE0 + 2.8, t);
        api.label('ct-t', npt(CT[0].x, CT[0].y - 62), 'Classes nested inside classes', { ...o, alpha: aT, align: 'middle', size: 'title', help: '<p>The complete tree of classes is unknown for natural data.</p>' });
        api.label('ct-note', npt(CT_NOTE.x, CT_NOTE.y), 'dots = points, colored by class', { ...o, alpha: aNote, align: 'below' });
      }
      // 5: what the network gets and what the ground truth holds
      const aB = boxOf(t);
      showL(boxL, aB > 0.001);
      if (aB > 0.001) {
        boxL.uIntensity.value = aB;
        const c1 = (BOX.x0 + BOX_MID) / 2, c2 = (BOX_MID + BOX.x1) / 2, yH = BOX.y0 + 19, yT = BOX.y1 - 20;
        api.label('bx1h', npt(c1, yH), 'the network gets', { color: CSS.slate, alpha: aB, align: 'middle', size: 'sm' });
        api.label('bx1', npt(c1, yT), '$x$ and $y$', { ...o, alpha: aB, align: 'middle', size: 'md' });
        api.label('bx2h', npt(c2, yH), 'the ground truth', { color: CSS.slate, alpha: aB, align: 'middle', size: 'sm' });
        api.label('bx2', npt(c2, yT), 'the tree and every $z$', { ...o, alpha: aB, align: 'middle', size: 'md' });
      }
    };

    const offline = await offlineReq;
    if (offline?.schema !== 'percolation-offline/1') throw new Error(`${OFFLINE_URL}: unknown schema ${offline?.schema}`);
    await ctx.log('offline results ✓');

    // ═══ 10. the offline rerun at the paper's scale ═══════════════════════════════════════════════
    /**
     * Use the exported measurements from public/data/percolation-offline.json (schema percolation-offline/1).
     * The offline study trains the networks and fits the probes. This block draws in frame pixels (netG) and returns
     * the beats, notes and comparison with the network before training.
     */
    const rerun = (t0: number): Block => {
      const t1 = t0 + 7, t2 = t1 + 6.5, t3 = t2 + 8, end = t3 + 6;
      const D = offline, RUN = 'tree0-init0', TREE = 'tree0';
      const KC = D.constants, NET = D.networks[RUN];
      const P_N = KC.n, P_D = KC.d, P_WIDTH = KC.d_model, P_HIDDEN = KC.d_model * KC.expansion, P_BLOCKS = KC.blocks, P_MIN = KC.min_latent_points;
      const R_RIDGE = D.baselines[TREE].ridge.r2, R_NN = D.baselines[TREE].nn1.r2, R_NET = NET.test.r2, PAPER = D.baselines.paper;
      const across = D.decision_across_runs, nRuns: number = across.runs.length, call: { hidden: string; residual: string } = across.call;
      const r3 = (x: number) => x.toFixed(3);
      const netOf = (t: number) => sm(t0 + 0.3, t0 + 1.1, t) * (1 - sm(t2, t2 + 0.5, t)) + sm(t3 + 0.2, t3 + 0.9, t);
      const sitesOf = (t: number) => (t < t2 ? sm(t1 + 0.5, t1 + 1.2, t) : 1);
      const fitOf = (t: number) => sm(t0 + 0.6, t0 + 1.3, t) * (1 - sm(t1, t1 + 0.5, t));
      const blockOf = (t: number) => sm(t0 + 1.4, t0 + 2.2, t) * (1 - sm(t1, t1 + 0.5, t)); // the labeled steps inside a block
      const probeOf = (t: number) => sm(t1 + 1.0, t1 + 1.8, t) * (1 - sm(t2, t2 + 0.5, t)); // the probe on the fixed network
      const tgtOf = (t: number) => sm(t1 + 2.2, t1 + 2.9, t) * (1 - sm(t2, t2 + 0.5, t)); // P's cluster, colored by the probe's target
      const chartsOf = (t: number) => sm(t2 + 0.4, t2 + 1.0, t) * (1 - sm(t3, t3 + 0.5, t));
      const SWEEP = 1.0, T_RAW = t2 + 1.0, T_TR = t2 + 2.3, T_UN = t2 + 3.7;
      const glyphOf = (t: number) => sm(t3 + 0.4, t3 + 1.2, t); // the probe errors at every site in the network
      const ansOf = (t: number) => sm(t3 + 0.9, t3 + 1.5, t);
      const numWord = (k: number) => ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][k] ?? String(k);

      // ── the paper's network, at its size: coordinates in frame pixels, as for the other flat figures ──
      const YS = 300, YH = 440, BAR = 9; // residual stream; center of the hidden layers; stroke width of a bar
      const X_IN = 60, BOX_IN = [84, 140], X_R0 = 164, X_RL = 566, BOX_OUT = [588, 644], X_OUT = 664, BOX_H = 18, ADD_R = 8;
      const FPX = 128 / P_HIDDEN; // pixels per number: the hidden layer's bar is 128 px tall
      const nL = P_BLOCKS, pitch = (X_RL - X_R0) / nL, ks = pitch / 134;
      const xBr = (l: number) => X_R0 + 22 * ks + l * pitch, xHid = (l: number) => xBr(l) + 44 * ks, xAdd = (l: number) => xBr(l) + 88 * ks, xRes = (l: number) => X_R0 + (l + 1) * pitch;
      const netW = new LineField(40 + 30 * nL); // wires, boxes, arrowheads, adders
      const wire = (x0: number, y0: number, x1: number, y1: number, head = false) => {
        netW.push(NP(x0, y0), NP(x1, y1), INK, AXK);
        if (!head) return;
        const len = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / len, uy = (y1 - y0) / len, a = 5.5, w = 2.6;
        netW.push(NP(x1, y1), NP(x1 - a * ux - w * uy, y1 - a * uy + w * ux), INK, AXK);
        netW.push(NP(x1, y1), NP(x1 - a * ux + w * uy, y1 - a * uy - w * ux), INK, AXK);
      };
      const box = (x0: number, x1: number) => { wire(x0, YS - BOX_H, x1, YS - BOX_H); wire(x1, YS - BOX_H, x1, YS + BOX_H); wire(x1, YS + BOX_H, x0, YS + BOX_H); wire(x0, YS + BOX_H, x0, YS - BOX_H); };
      wire(X_IN + 9, YS, BOX_IN[0], YS, true);
      box(BOX_IN[0], BOX_IN[1]);
      box(BOX_OUT[0], BOX_OUT[1]);
      wire(BOX_OUT[1], YS, X_OUT, YS, true);
      for (let l = 0; l < nL; l++) {
        wire(l ? xAdd(l - 1) + ADD_R : BOX_IN[1], YS, xAdd(l) - ADD_R, YS); // the stream up to this block's adder
        wire(xBr(l), YS, xBr(l), YH);
        wire(xBr(l), YH, xHid(l) - 9, YH, true); // into the hidden layer
        wire(xHid(l) + 9, YH, xAdd(l), YH);
        wire(xAdd(l), YH, xAdd(l), YS + ADD_R, true); // back to the stream
        netW.circle(NP(xAdd(l), YS), ADD_R, [1, 0, 0], [0, 1, 0], INK, AXK, 20);
        wire(xAdd(l) - 4, YS, xAdd(l) + 4, YS);
        wire(xAdd(l), YS - 4, xAdd(l), YS + 4);
      }
      wire(xAdd(nL - 1) + ADD_R, YS, BOX_OUT[0], YS, true);
      netW.commit();
      const netDots = new PointField({ count: nL, size: 1, bokeh: 0, twinkle: 0 }); // where a block reads the stream
      for (let l = 0; l < nL; l++) { netDots.setPos(l, ...NP(xBr(l), YS)); netDots.setColor(l, INK, 0.9); netDots.sizes[l] = 6; }
      netDots.commit();
      // the places a probe reads: raw input, the stream after the input layer, then the hidden layer and the stream of each block
      const sites: { x: number; y: number; h: number; kind: 0 | 1 | 2 }[] = [
        { x: X_IN, y: YS, h: Math.max(BAR + 3, P_D * FPX), kind: 0 },
        { x: X_R0, y: YS, h: Math.max(BAR + 3, P_WIDTH * FPX), kind: 1 },
      ];
      for (let l = 0; l < nL; l++) sites.push({ x: xHid(l), y: YH, h: 128, kind: 2 }, { x: xRes(l), y: YS, h: Math.max(BAR + 3, P_WIDTH * FPX), kind: 1 });
      const SITE_COL: RGB[] = [INK, C.cyan, scaleRGB(C.violet, 1.2)], SITE_CSS = [CSS.white, CSS.cyan, CSS.violet];
      const NEUTRAL = 0.36; // ink of a bar before its place is marked as a probe's
      const netBars = new LineField(sites.length, { width: BAR });
      netBars.count = sites.length;
      netG.add(netW.object, netBars.object, netDots.object);
      let barSig = -1;
      const sq = (i: number, txt: string) => `${tag(SITE_CSS[i], '■')} ${txt}`;

      // ── Fit on held-out points during training, with the two baselines and the paper's values ──
      const FX0 = 800, FX1 = 1100, FY0 = 262, FY1 = 560; // plot area: passes over the data across, R² up
      const fitTicks = [0.4, 0.6, 0.8, 1], fitMin = fitTicks[0], fitMax = fitTicks[fitTicks.length - 1];
      const eTicks = [0, KC.epochs / 2, KC.epochs];
      const fxE = (e: number) => FX0 + (e / KC.epochs) * (FX1 - FX0);
      const fyR = (r: number) => FY1 - ((r - fitMin) / (fitMax - fitMin)) * (FY1 - FY0);
      const fitAx = new LineField(16);
      fitAx.push(NP(FX0, FY1), NP(FX1, FY1), INK, AXK);
      fitAx.push(NP(FX0, FY1), NP(FX0, FY0), INK, AXK);
      for (const v of fitTicks) {
        fitAx.push(NP(FX0, fyR(v)), NP(FX0 - 6, fyR(v)), INK, AXK);
        if (v > fitMin) fitAx.push(NP(FX0, fyR(v)), NP(FX1, fyR(v)), INK, GRIDK);
      }
      for (const e of eTicks) fitAx.push(NP(fxE(e), FY1), NP(fxE(e), FY1 + 6), INK, AXK);
      fitAx.commit();
      const cE: number[] = NET.curve.epoch, cR: number[] = NET.curve.val_r2, nC = cE.length;
      const fitCurve = new LineField(Math.max(1, nC - 1), { width: 2 });
      for (let i = 1; i < nC; i++) fitCurve.push(NP(fxE(cE[i - 1]), fyR(cR[i - 1])), NP(fxE(cE[i]), fyR(cR[i])), C.violet, 1);
      fitCurve.commit();
      // To the right of the epoch axis, separate the near-equal network and nearest-point scores horizontally.
      const fitResults = [
        { x: FX1 + 33, test: R_RIDGE, published: PAPER.ridge.r2 },
        { x: FX1 + 33, test: R_NN, published: PAPER.nn1.r2 },
        { x: FX1 + 14, test: R_NET, published: PAPER.mlp.r2 },
      ];
      const fitMarks = new PointField({ count: fitResults.length, size: 1, bokeh: 0, twinkle: 0 });
      fitResults.forEach(({ x, test }, q) => {
        fitMarks.setPos(q, ...NP(x, fyR(test)));
        fitMarks.setColor(q, q === 2 ? C.violet : INK, q === 2 ? 1 : 0.85);
        fitMarks.sizes[q] = q === 2 ? 9 : 7;
      });
      fitMarks.commit();
      const fitPaper = new LineField(4 * fitResults.length); // the paper's values: a diamond around each mark
      for (const { x, published } of fitResults) {
        const y = fyR(published), r = 7;
        fitPaper.push(NP(x, y - r), NP(x + r, y), INK, 0.9); fitPaper.push(NP(x + r, y), NP(x, y + r), INK, 0.9);
        fitPaper.push(NP(x, y + r), NP(x - r, y), INK, 0.9); fitPaper.push(NP(x - r, y), NP(x, y - r), INK, 0.9);
      }
      fitPaper.commit();
      const fitLabX = FX1 + 50, fitLabY = dodge(fitResults.map(({ test }) => fyR(test)), 22, FY0 - 6, FY1);
      const fitLeads = new LineField(fitResults.length);
      fitResults.forEach(({ x, test }, q) => fitLeads.push(NP(x + 9, fyR(test)), NP(fitLabX - 3, fitLabY[q]), INK, 0.45));
      fitLeads.commit();
      netG.add(fitAx.object, fitCurve.object, fitLeads.object, fitPaper.object, fitMarks.object);

      // ── The probe, a separate readout on the last block's hidden layer ──
      const PRB = { x0: 606, x1: 712, y: 487, h: 32 }, PR_SITE = 2 + 2 * (nL - 1), PR_OUT = PRB.x1 + 34;
      const prbL = new LineField(16, { width: 2 });
      {
        const xs = sites[PR_SITE].x + BAR / 2 + 3, y = PRB.y, y0 = y - PRB.h / 2, y1 = y + PRB.h / 2, a = 6, w = 3;
        prbL.push(NP(xs, y), NP(PRB.x0, y), INK, 1); // the tap on the hidden layer
        prbL.push(NP(PRB.x0, y), NP(PRB.x0 - a, y - w), INK, 1); prbL.push(NP(PRB.x0, y), NP(PRB.x0 - a, y + w), INK, 1);
        prbL.push(NP(PRB.x0, y0), NP(PRB.x1, y0), INK, 1); prbL.push(NP(PRB.x1, y0), NP(PRB.x1, y1), INK, 1); // the box
        prbL.push(NP(PRB.x1, y1), NP(PRB.x0, y1), INK, 1); prbL.push(NP(PRB.x0, y1), NP(PRB.x0, y0), INK, 1);
        prbL.push(NP(PRB.x1, y), NP(PR_OUT, y), INK, 1); // its output
        prbL.push(NP(PR_OUT, y), NP(PR_OUT - a, y - w), INK, 1); prbL.push(NP(PR_OUT, y), NP(PR_OUT - a, y + w), INK, 1);
      }
      prbL.commit();
      netG.add(prbL.object);

      // ── The probe's target on P's cluster. At each input x, the point changes from its target y to the z
      // of its latent at the cut depth kd, or to gray if it has no latent at that depth. ──
      const TG = { x: 1110, y: 470, r: 180 }, tgK = TG.r / WALK_R; // center and radius of the picture (frame pixels)
      const TGX0 = 905; // left edge of its title and legend
      const tgX = (p: number) => TG.x + walkP[p * 3] * tgK, tgY = (p: number) => TG.y - walkP[p * 3 + 1] * tgK;
      const tgtPts = new PointField({ count: s, size: 1, bokeh: 0, twinkle: 0 });
      for (let p = 0; p < s; p++) { tgtPts.setPos(p, ...NP(tgX(p), tgY(p))); tgtPts.sizes[p] = 16; }
      tgtPts.commit();
      const tgtE = new LineField(Math.max(1, m), { width: 1.5 });
      for (let j = 0; j < m; j++) tgtE.push(NP(tgX(T.eu[j]), tgY(T.eu[j])), NP(tgX(T.ev[j]), tgY(T.ev[j])), INK, 0.4);
      tgtE.commit();
      const tgtRing = new LineField(24, { width: 2 });
      tgtRing.circle(NP(tgX(pk), tgY(pk)), 10, [1, 0, 0], [0, 1, 0], INK, 1, 24);
      tgtRing.commit();
      netG.add(tgtE.object, tgtPts.object, tgtRing.object);
      // each point in the color of the probe's target: the z of its latent at depth k
      for (let i = 0; i < s * 3; i++) tgtPts.colors[i] = tgtCol[i];
      tgtPts.commit(false, true, false);
      let nTgt = 0;
      for (let p = 0; p < s; p++) nTgt += tgtHas[p];

      // ── Probe error by latent size for the hidden layers and residual stream ──
      // The entries of the paper's tree, their size bins and their per-latent errors, pooled over the three sites of a kind.
      const ENT = D.entries[TREE], PM = D.probe_mse[RUN], B5 = D.binned[RUN].bins5;
      const nB: number = B5.count.length;
      const qtl = (v: number[], q: number) => {
        if (!v.length) return NaN;
        const a = Float64Array.from(v).sort(), h = (a.length - 1) * q, lo = Math.floor(h), hi = Math.min(a.length - 1, lo + 1);
        return a[lo] + (a[hi] - a[lo]) * (h - lo);
      };
      const pooled = (arrs: number[][]) => {
        const bins: number[][] = Array.from({ length: nB }, () => []);
        for (const arr of arrs) for (let i = 0; i < arr.length; i++) if (ENT.kept[i] === 1 && ENT.bin5[i] >= 0) bins[ENT.bin5[i]].push(arr[i]);
        return { q50: bins.map((v) => qtl(v, 0.5)), q25: bins.map((v) => qtl(v, 0.25)), q75: bins.map((v) => qtl(v, 0.75)) };
      };
      const PAN = [
        { x0: 120, x1: 560, sites: ['hidden0', 'hidden1', 'hidden2'], col: scaleRGB(C.violet, 1.2), css: CSS.violet, title: 'Hidden layers', which: 'the three hidden layers' },
        { x0: 760, x1: 1200, sites: ['block0', 'block1', 'block2'], col: C.cyan, css: CSS.cyan, title: 'Residual stream', which: 'the stream after each of the three blocks' },
      ].map((p) => ({ ...p, raw: pooled([PM.raw]), tr: pooled(p.sites.map((s) => PM.trained[s])), un: pooled(p.sites.map((s) => PM.untrained[s])) }));
      const CY0 = 290, CY1 = 540, LO = Math.log10(0.0015), HI = Math.log10(1.5); // includes every displayed quartile
      const cyE = (v: number) => CY1 - ((Math.log10(Math.max(v, 1e-6)) - LO) / (HI - LO)) * (CY1 - CY0);
      const bxP = (p: { x0: number; x1: number }, b: number) => p.x0 + ((b + 0.5) * (p.x1 - p.x0)) / nB;
      const yTicks = [0.003, 0.01, 0.03, 0.1, 0.3, 1];
      const binEdgePts: number[] = B5.edges.map((f: number) => f * Math.sqrt(P_N));
      const pts2 = (v: number) => { const e = Math.pow(10, Math.floor(Math.log10(v)) - 1); return int(Math.round(v / e) * e); }; // two significant figures
      const chAx = new LineField(2 * (4 + yTicks.length * 2 + nB + 1));
      for (const p of PAN) {
        chAx.push(NP(p.x0, CY1), NP(p.x1, CY1), INK, AXK);
        chAx.push(NP(p.x0, CY1), NP(p.x0, CY0), INK, AXK);
        for (const v of yTicks) { chAx.push(NP(p.x0, cyE(v)), NP(p.x0 - 6, cyE(v)), INK, AXK); chAx.push(NP(p.x0, cyE(v)), NP(p.x1, cyE(v)), INK, GRIDK); }
        for (let b = 0; b <= nB; b++) { const x = p.x0 + (b * (p.x1 - p.x0)) / nB; chAx.push(NP(x, CY1), NP(x, CY1 + 6), INK, AXK); }
      }
      chAx.commit();
      netG.add(chAx.object);
      const MR = 5.5; // radius of an open mark
      /** One line of a panel: its bin medians joined, filled or open marks, dashed when open; quartile bars for the notes. */
      const series = PAN.flatMap((p) => [
        { p, q: p.raw, col: INK, k: 0.85, open: false, t0: T_RAW, size: 7 },
        { p, q: p.tr, col: p.col, k: 1, open: false, t0: T_TR, size: 9 },
        { p, q: p.un, col: p.col, k: 1, open: true, t0: T_UN, size: 0 },
      ]).map((sr) => {
        const ln = new LineField(nB * 14, { width: 2 }), rg = new LineField(nB * 14), bars = new LineField(nB * 3);
        const mkF = sr.open ? null : new PointField({ count: nB, size: 1, bokeh: 0, twinkle: 0 });
        const lnEnd = new Int32Array(nB + 1), rgEnd = new Int32Array(nB + 1), barEnd = new Int32Array(nB + 1);
        const dxo = sr.open ? 4 : sr.col === INK ? 0 : -4; // the before and after marks stand a little apart
        let prev = -1;
        for (let b = 0; b < nB; b++) {
          const x = bxP(sr.p, b) + dxo, m = sr.q.q50[b], y = cyE(m);
          if (isFinite(m)) {
            if (prev >= 0) {
              const xa = bxP(sr.p, prev) + dxo, ya = cyE(sr.q.q50[prev]);
              if (sr.open) {
                const len = Math.hypot(x - xa, y - ya), u = Math.min(0.45, (MR + 1.5) / Math.max(1e-6, len));
                dashTo(ln, lerp(xa, x, u) - NX0, NY1 - lerp(ya, y, u), lerp(x, xa, u) - NX0, NY1 - lerp(y, ya, u), 5, 4, sr.col, 0.9);
              } else ln.push(NP(xa, ya), NP(x, y), sr.col, sr.k * 0.9);
            }
            if (sr.open) rg.circle(NP(x, y), MR, [1, 0, 0], [0, 1, 0], sr.col, 1.1, 14);
            if (mkF) { mkF.setPos(b, ...NP(x, y)); mkF.setColor(b, sr.col, sr.k); mkF.sizes[b] = sr.size; }
            bars.push(NP(x, cyE(sr.q.q25[b])), NP(x, cyE(sr.q.q75[b])), sr.col, 0.5);
            prev = b;
          } else if (mkF) mkF.sizes[b] = 0;
          lnEnd[b + 1] = ln.count; rgEnd[b + 1] = rg.count; barEnd[b + 1] = bars.count;
        }
        ln.commit(); rg.commit(); bars.commit(); mkF?.commit();
        netG.add(bars.object, ln.object, rg.object);
        if (mkF) netG.add(mkF.object);
        return { ...sr, ln, rg, bars, mkF, lnEnd, rgEnd, barEnd };
      });
      const binsAt = (ts: number, t: number) => Math.min(nB, Math.floor(nB * ramp(ts, ts + SWEEP, t) + 1e-6));

      // ── The probe error at every site in the network, after and before training, as a share of the raw input's.
      // One pair of bars under each place: the median over the latents of the first tree (first starting weights) of the
      // probe's error there divided by the raw input's. A filled bar is the trained network, an open one the same network
      // before training; the dashed line is the raw input's own level. ──
      const siteKey = (i: number) => (i === 1 ? 'input_layer' : `${i % 2 === 0 ? 'hidden' : 'block'}${(i - 2) >> 1}`);
      const medRatio = (arr: number[]) => {
        const v: number[] = [];
        for (let e = 0; e < arr.length; e++) if (ENT.kept[e] === 1) v.push(arr[e] / PM.raw[e]);
        return qtl(v, 0.5);
      };
      const glyph = sites.map((_, i) => (i === 0 ? { tr: 1, un: 1 } : { tr: medRatio(PM.trained[siteKey(i)]), un: medRatio(PM.untrained[siteKey(i)]) }));
      // The chart stands in the right column: one slot per place, in the network's order from left to right; the
      // height is the error as a share of the raw input's (0 to 1), the dashed line the raw input's own level.
      const GX0 = 800, GX1 = 1230, GB = 492, GH = 190, GW = 11; // plot ends, baseline, height of the level 1, bar width
      const gSlot = (i: number) => GX0 + ((i + 0.5) * (GX1 - GX0)) / sites.length;
      const glyF = new LineField(sites.length * 2, { width: GW }); // filled bars: the raw input, and each place after training
      const glyL = new LineField(sites.length * 4 + 200, { width: 1.5 }); // open bars (before training), the axes, the raw input's level
      const gBar = (L: LineField, x: number, h: number, col: RGB) => {
        const cap = Math.min(GW / 2, h / 2); // the round caps of a wide stroke reach past its ends
        L.push(NP(x, GB - cap), NP(x, GB - h + cap), col, 1);
      };
      glyL.push(NP(GX0, GB), NP(GX1, GB), INK, AXK);
      glyL.push(NP(GX0, GB), NP(GX0, GB - GH - 14), INK, AXK);
      for (const v of [0, 0.5, 1]) glyL.push(NP(GX0, GB - v * GH), NP(GX0 - 6, GB - v * GH), INK, AXK);
      glyL.push(NP(GX0, GB - 0.5 * GH), NP(GX1, GB - 0.5 * GH), INK, GRIDK);
      dashTo(glyL, GX0 - NX0, NY1 - (GB - GH), GX1 - NX0, NY1 - (GB - GH), 5, 4, INK, 0.6);
      sites.forEach((st, i) => {
        const col = SITE_COL[st.kind], g = glyph[i], x = gSlot(i);
        if (i === 0) { gBar(glyF, x, GH, INK); return; }
        const xa = x - GW / 2 - 1.5, xb = x + GW / 2 + 1.5, h1 = GB - g.un * GH, x0 = xb - GW / 2 + 0.75, x1 = xb + GW / 2 - 0.75;
        gBar(glyF, xa, g.tr * GH, col);
        glyL.push(NP(x0, GB), NP(x0, h1), col, 1); glyL.push(NP(x0, h1), NP(x1, h1), col, 1); glyL.push(NP(x1, h1), NP(x1, GB), col, 1);
      });
      glyF.commit();
      glyL.commit();
      netG.add(glyL.object, glyF.object);
      /** Short names of the places, under their slots: block by block, as the network runs. */
      const slotName = (i: number) => (i === 0 ? 'raw<br>input' : i === 1 ? 'input<br>layer' : `${sites[i].kind === 2 ? 'hidden' : 'stream'}<br>${(i - 2 >> 1) + 1}`);

      // ── The comparison with the same network before training, also used on the closing board ──
      const DEC = D.decision[RUN];
      const cellRange = (ss: string[]) => {
        const v = ss.flatMap((s) => DEC.cells[s] as number[]);
        return `${Math.min(...v).toPrecision(3)}–${Math.max(...v).toPrecision(3)}`;
      };
      const noGain = call.hidden === 'no gain' && call.residual === 'no gain', helps = call.hidden === 'helps' && call.residual === 'helps';
      const runs = numWord(nRuns);
      const trainingSummary = noGain
        ? `At the authors’ probe penalty, training gave no consistent gain in estimating $z$ across ${runs} runs.`
        : helps ? `At the authors’ configured ridge penalty, training makes the latents easier for probes to read in all ${runs} runs.`
        : `At the authors’ configured ridge penalty, the ${runs} runs give no single answer on whether training makes the latents easier for probes to read.`;
      const control: Result = {
        label: "trained / untrained probe error at the authors’ configured ridge penalty, first tree and initialization",
        measured: `hidden layers ${cellRange(PAN[0].sites)}, block outputs ${cellRange(PAN[1].sites)}`,
        reference: `“Helps” requires every eligible bin at every site below ${KC.thresholds.helps}. “No gain” requires a strict majority at or above ${KC.thresholds.no_gain} at each site. All ${nRuns} runs: hidden layers ${call.hidden}, residual stream ${call.residual}.`,
        kind: BEFORE_TRAINING_KIND,
        group: 'finding',
        headline: true,
        board: true,
        say: trainingSummary,
        // the finale prints this verbatim as a row of its board: subject first, counts in words
        claim: noGain ? `Probes estimated latent values before training. In each of ${runs} runs at ridge α = ${KC.ridge_alpha}, most eligible bins at each site improved by 5% or less, or got worse.`
          : helps ? `Probes read the latents more easily after training than before in all ${runs} runs, at the authors’ configured ridge penalty.`
          : `The ${runs} runs give no single answer on whether training makes the latents easier to read, at the authors’ configured ridge penalty.`,
      };

      // The known target response supplies a quantitative benchmark for a future activation edit.
      const EDIT = { x: 40, y: 572 };
      const BLK = { e: [180, 440], p: [xAdd(0), YS + 42], a: [xAdd(0), YS - 12] }; // the inside of the first block: where its three steps are named (projecting back on its own line)
      const draw = (t: number, api: FrameApi) => {
        const o = { color: CSS.white } as const;
        const cam = api.camera;
        // Show the trained network, fade it behind the probe, then bring it back above the per-site errors.
        const aN = netOf(t), onN = aN > 0.001, aP = probeOf(t), aG = t >= t3 ? glyphOf(t) : 0;
        const fade = lerp(1, 0.4, aP);
        showL(netW, onN);
        showL(netBars, onN);
        netDots.object.visible = onN;
        showL(prbL, onN && aP > 0.001);
        showL(glyF, onN && aG > 0.001);
        showL(glyL, onN && aG > 0.001);
        if (onN) {
          netW.uIntensity.value = aN * fade;
          netBars.uIntensity.value = aN;
          netDots.uOpacity.value = aN * fade;
          prbL.uIntensity.value = aN * aP;
          glyF.uIntensity.value = aN * aG;
          glyL.uIntensity.value = aN * aG;
          const mix = sitesOf(t);
          if (mix !== barSig) {
            barSig = mix;
            sites.forEach((st, i) => {
              const c = SITE_COL[st.kind], k = lerp(NEUTRAL, st.kind ? 1 : 0.95, mix), half = st.h / 2 - BAR / 2;
              netBars.seg(i, NP(st.x, st.y - half), NP(st.x, st.y + half), [lerp(INK[0], c[0], mix), lerp(INK[1], c[1], mix), lerp(INK[2], c[2], mix)], k);
            });
            netBars.commit();
          }
          const mm = { ...o, alpha: aN * 0.9 * lerp(1, 0.7, aP) } as const;
          const at6 = t < t1 + 1 ? 1 : 0, at9 = t >= t3 ? 1 : 0;
          api.label('ti', scr(cam, TITLE_X, TITLE_Y), 'Residual network', {
            ...o, alpha: aN * (1 - at9) * (1 - sm(0.1, 0.45, aP)), align: 'left', size: 'title',
            help: `<p>Each bar’s height shows the number of coordinates at that location. Every residual block works like the first, with ${int(P_HIDDEN)} ReLU neurons in its hidden layer.</p>`,
          });
          api.label('ti7', scr(cam, TITLE_X, TITLE_Y), 'Probing a frozen network', { ...o, alpha: aN * sm(0.55, 0.9, aP), align: 'left', size: 'title', help: '<p>The probe reads activations at one location. Its weights are fitted while the network’s weights stay fixed.</p>' });
          api.label('ti9', scr(cam, TITLE_X, TITLE_Y), 'Probe locations', { ...o, alpha: aN * at9, align: 'left', size: 'title' });
          api.label('n-x', npt(X_IN, YS + BOX_H + 4), `input $x$<small>${P_D} numbers</small>`, { ...mm, align: 'below' });
          api.label('n-in', npt((BOX_IN[0] + BOX_IN[1]) / 2, YS), 'input<br>layer', { ...mm, align: 'middle', size: 'xs' });
          api.label('n-out', npt((BOX_OUT[0] + BOX_OUT[1]) / 2, YS), 'output<br>layer', { ...mm, align: 'middle', size: 'xs' });
          api.label('n-y', npt(X_OUT + 21, YS - 2), 'predicted $y$', { ...mm, align: 'center' });
          api.label('n-rs', npt((xRes(0) + X_RL) / 2, YS - 30), `residual stream: ${P_WIDTH} numbers`, { ...mm, align: 'center' });
          for (let l = 0; l < nL; l++) api.label('n-h' + l, npt(xHid(l), YH + 64 + 5), `hidden ${l + 1}`, { ...mm, align: 'below' });
          // Label the steps inside the first block. Every block works the same way.
          const aB = aN * blockOf(t) * at6;
          if (aB > 0.01) {
            api.label('bk-e', npt(BLK.e[0], BLK.e[1]), `expand + ReLU<small>${P_WIDTH} → ${int(P_HIDDEN)} numbers</small>`, { ...o, alpha: aB, align: 'right', size: 'xs' });
            api.label('bk-p', npt(BLK.p[0], BLK.p[1]), 'project back', { ...o, alpha: aB * sm(t0 + 2.0, t0 + 2.6, t), align: 'middle', size: 'xs', plate: true });
            api.label('bk-a', npt(BLK.a[0], BLK.a[1]), 'add to the stream', { ...o, alpha: aB * sm(t0 + 2.6, t0 + 3.2, t), align: 'center', size: 'xs' });
          }
          api.label('li1', scr(cam, TITLE_X, LEG_Y), `${sq(0, 'raw input')} &nbsp; ${sq(1, 'residual stream')} &nbsp; ${sq(2, 'hidden layers')}`, { ...o, alpha: aN * mix * 0.9 * (1 - aG), align: 'left' });
          // The probe, its site ℓ, and its output.
          if (aP > 0.01) {
            const a = aN * aP;
            api.label('pr-box', npt((PRB.x0 + PRB.x1) / 2, PRB.y), 'linear probe', { ...o, alpha: a, align: 'middle', size: 'md' });
            api.label('pr-l', npt((sites[PR_SITE].x + PRB.x0) / 2 + 4, PRB.y - 4), 'reads place $\\ell$', { ...o, alpha: a, align: 'center' });
            api.label('pr-out', npt(PR_OUT + 2, PRB.y), '$z$ at depth $k$', { ...o, alpha: a, align: 'left', size: 'md' });
          }
          // Bars compare the probe error at each site with the raw input's.
          if (aG > 0.01) {
            const a = aN * aG;
            api.label('gl-t', npt(GX0 - 12, 214), 'Probe error by location', {
              ...o, alpha: a, align: 'left', size: 'title',
              help: '<p>The comparison uses the first tree and initialization. For each latent, divide the probe’s mean squared error at that location by the raw-input probe’s error. Each bar shows the median of those ratios. The dashed line is the raw-input baseline at 1.</p>',
            });
            api.label('gl-k', npt(GX0 - 12, 262), 'median error ratio to raw input<small>filled = trained · open = untrained · dashed = raw input</small>', { ...o, alpha: a * 0.9, align: 'left' });
            for (const v of [0, 0.5, 1]) api.label('gl-y' + v, npt(GX0 - 9, GB - v * GH), String(v), { ...o, alpha: a * 0.9, align: 'right', size: 'xs' });
            sites.forEach((st, i) => api.label('gl-x' + i, npt(gSlot(i), GB + 6), slotName(i), { color: SITE_CSS[st.kind], alpha: a, align: 'below', size: 'xs' }));
          }
        }

        // Fit during training.
        const aF = fitOf(t), onF = aF > 0.001;
        showL(fitAx, onF);
        showL(fitPaper, onF);
        showL(fitLeads, onF);
        fitMarks.object.visible = onF;
        if (!onF) showL(fitCurve, false);
        if (onF) {
          const nShow = Math.max(0, Math.min(nC - 1, Math.floor((nC - 1) * ramp(t0 + 1.5, t0 + 4.3, t))));
          const aM = sm(t0 + 4.4, t0 + 4.9, t), aPp = sm(t0 + 5.1, t0 + 5.6, t);
          fitCurve.count = nShow;
          showL(fitCurve, nShow > 0);
          fitAx.uIntensity.value = aF;
          fitCurve.uIntensity.value = aF;
          fitMarks.uOpacity.value = aF * aM;
          fitLeads.uIntensity.value = aF * aM;
          fitPaper.uIntensity.value = aF * aPp;
          api.label('fit-t', npt(FX0 - 12, FY0 - 70), 'Predicting $y$', {
            ...o, alpha: aF, align: 'left', size: 'title',
            help: '<p>The curve is validation $R^2$ during training in the rerun. Dots are rerun test scores for the network and both baselines. Diamonds give the corresponding scores in Brill and Carlson’s Table 1.</p>',
          });
          api.label('fit-s', npt(FX0 - 12, FY0 - 24), '$R^2$ &nbsp; curve = validation &nbsp; ● rerun test &nbsp; ◇ published', { ...o, alpha: aF * 0.9, align: 'left' });
          for (const v of fitTicks) api.label('fy' + v, npt(FX0 - 9, fyR(v)), v.toFixed(1), { ...o, alpha: aF * 0.9, align: 'right' });
          for (const epoch of eTicks) api.label('fe-' + epoch, npt(fxE(epoch), FY1 + 6), int(epoch), { ...o, alpha: aF * 0.9, align: 'below', size: 'xs' });
          api.label('fe-n', npt((FX0 + FX1) / 2, FY1 + 28), 'training epochs', { ...o, alpha: aF * 0.9, align: 'below' });
          const labs = [`linear fit ${r3(R_RIDGE)}`, `full-data nearest neighbor ${r3(R_NN)}`, `network ${r3(R_NET)}`];
          labs.forEach((txt, q) => api.label('fl' + q, npt(fitLabX, fitLabY[q]), `${txt}${aPp > 0.5 ? ` ${tag(CSS.slate, `◇ ${fitResults[q].published}`)}` : ''}`, { color: q === 2 ? CSS.violet : CSS.white, alpha: aF * aM, align: 'left', size: q === 2 ? 'md' : 'sm' }));
        }

        // The probe's target on P's cluster.
        const aT = tgtOf(t), onT = aT > 0.001;
        tgtPts.object.visible = onT;
        showL(tgtE, onT);
        showL(tgtRing, onT);
        if (onT) {
          tgtPts.uOpacity.value = aT;
          tgtE.uIntensity.value = aT;
          tgtRing.uIntensity.value = aT;
          api.label('tgt-t', npt(TGX0, 214), 'Illustrated probe targets', { ...o, alpha: aT, align: 'left', size: 'title', help: `<p>In the ${s}-point example, each subcluster shares a target, the $z$ of its latent at depth $k$. Gray marks points with no latent at that depth. The probe experiment uses the larger tree.</p>` });
          api.label('tgt-l2', npt(TGX0, 240), `color = latent $z$ at depth $k$<small>${zBarCss()}</small>${nTgt < s ? '<small>gray = no target at depth $k$</small>' : ''}`, { ...o, alpha: aT * 0.9, align: 'left', dy: 8 });
          api.label('tgt-p', npt(tgX(pk), tgY(pk)), 'P<small>its target, the $z$ of latent A</small>', { ...o, alpha: aT, align: tgX(pk) > TG.x ? 'left' : 'right', size: 'md', dx: tgX(pk) > TG.x ? 6 : -6, plate: true });
        }

        // Probe errors by latent size.
        const aC = chartsOf(t), onC = aC > 0.001;
        showL(chAx, onC);
        for (const sr of series) {
          const nb = onC ? binsAt(sr.t0, t) : 0, on = onC && nb > 0;
          sr.ln.count = sr.lnEnd[nb]; sr.rg.count = sr.rgEnd[nb]; sr.bars.count = sr.barEnd[nb];
          showL(sr.ln, on); showL(sr.rg, on); showL(sr.bars, on && notesOpen());
          sr.ln.uIntensity.value = aC; sr.rg.uIntensity.value = aC; sr.bars.uIntensity.value = aC;
          if (sr.mkF) { sr.mkF.drawCount = nb; sr.mkF.object.visible = on; sr.mkF.uOpacity.value = aC; }
        }
        if (onC) {
          chAx.uIntensity.value = aC;
          api.label('ch-tag', npt(PAN[0].x0 - 12, 196), 'Separate probes estimate latent values $z$', { alpha: aC, align: 'left' });
          PAN.forEach((p, i) => {
            const key = `${tag(CSS.white, `${lineKey(false, 'filled')}raw input`)} &nbsp; ${tag(p.css, `${lineKey(false, 'filled')}trained network`)} &nbsp; ${tag(p.css, `${lineKey(true, 'open')}untrained network`)}`;
            api.label('ch-t' + i, npt(p.x0 - 12, CY0 - 44), `${p.title}<small>${key}</small>`, {
              color: p.css, alpha: aC, align: 'left', size: 'title', dy: -14,
              help: `<p>Each curve shows how well fitted probes estimate latent values $z$ on the first run. We pool the errors over ${p.which} and score them on points held out from probe fitting. Each mark is the median across latent–site pairs in its size bin. With Notes open, bars span the middle half of the errors.</p>`,
            });
            api.label('ch-yn' + i, npt(p.x0, CY0 - 13), 'Median probe MSE', { ...o, alpha: aC, align: 'left' });
            yTicks.forEach((v) => api.label(`ch-y${i}-${v}`, npt(p.x0 - 9, cyE(v)), String(v), { ...o, alpha: aC * 0.9, align: 'right', size: 'xs' }));
            binEdgePts.forEach((v, b) => api.label(`ch-x${i}-${b}`, npt(p.x0 + (b * (p.x1 - p.x0)) / nB, CY1 + 8), pts2(v), { ...o, alpha: aC * 0.9, align: 'below', size: 'xs' }));
            api.label('ch-xn' + i, npt((p.x0 + p.x1) / 2, CY1 + 26), 'subcluster size (points)', { ...o, alpha: aC * 0.9, align: 'below' });
          });
        }

        const aA = ansOf(t);
        if (aA > 0.01) {
          api.label('edit-title', npt(EDIT.x, EDIT.y), 'Changing latent A in the generator', { ...o, alpha: aA, align: 'left', size: 'md', help: '<p>Increasing A’s value by one adds one to the path sum of every point beneath it. This gives the expected target change for a selective intervention.</p>' });
          api.label('edit-in', npt(EDIT.x, EDIT.y + 32), `Point P below A<small>adding 1 to A raises $y$ by $1/\\sqrt{1+d_P}$ = ${(1 / Math.sqrt(1 + pickD)).toFixed(3)}</small>`, { color: ACC_CSS, alpha: aA, align: 'left' });
          api.label('edit-out', npt(EDIT.x + 385, EDIT.y + 32), 'Outside A’s subcluster<small>target stays the same</small>', { ...o, alpha: aA, align: 'left' });
        }
      };

      const notes = (t: number): Hotspot[] => {
        const out: Hotspot[] = [];
        const cam = camNow;
        if (!cam) return out;
        if (netOf(t) > 0.5 && t < t1)
          out.push({
            id: 'net',
            pos: scr(cam, 0.031, TITLE_Y - 24 / 900),
            title: 'A rerun at the paper’s scale',
            body:
              `<p>The independent rerun uses a cluster of ${int(P_N)} points with ${P_D} input coordinates, made with Brill and Carlson’s generator. The browser drawings use a separate, smaller dataset.</p>` +
              '<p>Brill and Carlson chose this residual network to resemble a transformer’s feed-forward block.</p>',
          });
        if (fitOf(t) > 0.5)
          out.push({
            id: 'fit',
            pos: npt(FX0 - 6, FY0 - 100),
            title: 'What the nearest-point comparison measures',
            body:
              '<p>The nearest-point prediction copies $y$ from the nearest other point in the whole dataset. Brill and Carlson use it to estimate the best practically achievable fit. Its reference pool includes labeled points from the validation and test splits.</p>' +
              '<p>All tree edges have the same length, so many points have equally near neighbors. Choosing among ties changes the score in the third decimal.</p>',
          });
        if (probeOf(t) > 0.5)
          out.push({
            id: 'probe',
            pos: npt(PRB.x0 - 6, PRB.y - 64),
            title: 'Which points enter a probe',
            body:
              `<p>A point must lie deeper than the probed depth $k$. Latents with fewer than ${P_MIN} points beneath them are excluded together with their points, following the paper.</p>` +
              '<p>At a given depth, points in the same subcluster all share one probe target. The probe estimates that value on points held out from probe fitting within those subclusters. The probe split is separate from the network’s split, so its validation points can include points used to train the network.</p>',
          });
        if (chartsOf(t) > 0.5) {
          out.push({
            id: 'readings',
            pos: npt(PAN[0].x0 - 30, CY0 - 44),
            title: 'The network predicts y. The probes estimate z.',
            body:
              '<p>The network trains to predict the combined target $y$ from $x$. Each probe is fitted separately to estimate individual latent values $z$. Brill and Carlson compared trained activations with raw inputs. We also probe the same network at initialization, using the same latent targets and point splits in all three conditions.</p>' +
              `<p>The curves pool the three sites of each kind on the first tree and initialization. From smallest to largest, the bins contain ${B5.count.join(', ')} latents. Each point marks a median. The bars span the middle half of the errors.</p>`,
          });
          out.push({
            id: 'repeats',
            pos: npt(PAN[1].x0 - 30, CY0 - 44),
            title: 'L2 regularization changes the comparison',
            body:
              '<p>The probes use ridge regression with L2 regularization, a penalty proportional to the sum of squared probe weights.</p>' +
              '<p>Increasing ridge $\\alpha$ from 1 to 10,000 on the first run puts the trained network ahead in 30 of 35 site–bin comparisons, including the input layer. At 100, it is ahead in 10 of 35. Most probe errors rise at 10,000, especially for smaller latents.</p>' +
              '<p>The five runs use three starting weights on one tree and one on each of two further trees. Latents nested in one tree are not independent samples.</p>',
          });
        }
        if (glyphOf(t) > 0.5 && t >= t3)
          out.push({
            id: 'sites',
            pos: npt(GX0 - 30, 214),
            title: 'The input layer preserves the input',
            body: '<p>The input layer is an affine transformation of $x$. At the paper’s ridge penalty, its probe errors nearly match those on $x$. Changing the penalty can change that comparison. The nonlinear transformations in the later blocks can change what a linear probe predicts.</p>',
          });
        if (ansOf(t) > 0.5)
          out.push({
            id: 'answer',
            pos: npt(EDIT.x - 8, EDIT.y - 22),
            title: 'A prediction for a selective intervention',
            body:
              '<p>In the small tree, increasing A’s value changes one term in every descendant’s target. The change is smaller for deeper points, because their sums have a larger denominator. All points outside the subcluster retain their targets.</p>' +
              '<p>This prediction comes from the generator. An intervention inside the trained network would have to show that its own output responds selectively in the same way.</p>',
          });
        return out;
      };
      const details = (t: number): NoteDetails | undefined => {
        if (t < t0) return undefined;
        if (t < t1) return {
          title: 'How the rerun compares with the paper',
          body:
            '<p>The rerun uses the one-cluster generator settings in Table 4 and the training settings in Table 5, including 500 epochs. The authors’ public example config uses 250 epochs. The regenerated data could not be compared with the paper’s original files.</p>' +
            '<p>The rerun’s $R^2$ scores fall within 0.01 of Table 1, and the eight-bin probe errors have a similar scale and ordering to the medians in Fig. 4a. This comparison uses values read from the figure. Fig. 5’s scale could not be matched under the aggregations tried. No result here relies on that figure.</p>',
        };
        if (t < t2) return {
          title: 'Fitting and scoring the probes',
          body:
            '<p>At each network location and depth, a probe fits a weighted sum of activations plus an offset. Activations are standardized before fitting ridge regression with the penalty specified in the authors’ probe code. The penalty is proportional to the sum of squared probe weights. Each latent’s score averages the squared prediction error on its probe validation points.</p>' +
            '<p>Depths run from 10 to 200 in steps of 5. The paper also lists 0 and 5, but all eligible points of the first tree share one latent at those depths. Their probe target would be constant, so they are excluded.</p>' +
            '<p>The paper reports probes on trained networks. This presentation fits additional probes on the same networks at initialization.</p>',
        };
        if (t < t3) return {
          title: 'How the five runs are compared',
          body:
            `<p>For the network’s own target $y$, the first run’s validation $R^2$ rises from ${r3(cR[0])} after epoch ${cE[0]} to ${r3(cR[nC - 1])} after epoch ${cE[nC - 1]}. The validation curve starts after the first epoch. The initialization log records a training-set loss.</p>` +
            `<p>For the probe comparison at ridge $\\alpha=${KC.ridge_alpha}$, divide each latent’s trained probe MSE by its untrained probe MSE. Drop latents with mean-predictor MSE below 0.0001 and take the median ratio per site in bins with at least ten latents. At every site in each of the five runs, a strict majority of eligible bins had ratios of at least 0.95, a reduction of at most 5% or an increase. The residual group includes the input layer and all three block outputs.</p>` +
            '<p>Raising the cutoff to 1 makes the hidden result mixed on the two additional trees and the residual result mixed on all five runs. At the original cutoff, the result holds when omitting the input layer and when fitting and scoring the first run’s probes only on points held out from network training.</p>' +
            '<p>Similar probe errors leave open whether the networks represent $z$ in the same way or use it to predict $y$.</p>',
        };
        const vs = D.diagnostics.value_swap;
        const ratios = (Object.values(vs.z.other_net_over_own_net) as number[][]).flat().filter((v) => v !== null && v !== undefined && isFinite(v));
        return {
          title: 'A further check and an untested intervention',
          body:
            `<p>A sixth network was trained on the same tree and inputs with fresh latent values. Probes could still predict the original values about as well as from the network trained on them. Error ratios range from ${Math.min(...ratios).toFixed(2)} to ${Math.max(...ratios).toFixed(2)} across sites and size bins, including a probe on the network’s scalar output. This diagnostic uses one draw of replacement values and tests no mechanism.</p>` +
            '<p>Brill and Carlson write that “Causal interventions are needed to validate specific interpretability hypotheses”. Each latent value $z$ contributes to the known target $y$ with weight $1/\\sqrt{1+d_x}$.</p>' +
            '<p>A possible experiment would edit a candidate representation of one latent and measure the predicted $y$ on its descendants and on unrelated points. The generator supplies the expected change. An activation edit could also change other latent values, so the experiment would need to check its specificity.</p>',
        };
      };

      const s8result = 'The network gets better at predicting $y$ during training. Brill and Carlson found that probes estimate latent values $z$ better from trained activations than from raw inputs. We also fit probes to the untrained network. ' +
        trainingSummary + (noGain ? ' A much stronger penalty favored the trained network on the first run.' : '');
      const beats: Beat[] = [
        { at: t0, caption: 'Brill and Carlson train a residual network to predict $y$ from $x$. In our rerun, the network nearly matches a nearest-neighbor reference that can copy targets from the full dataset. Does the network use the latent values in the sum?' },
        { at: t1, caption: 'Brill and Carlson freeze the network and fit a <em>linear probe</em> to its activations. At depth $k$, one probe estimates the ancestor’s value for points in all eligible subclusters. In the small example, P’s probe target is A’s value.' },
        { at: t2, caption: s8result },
        { at: t3, caption: 'On this tree, a probe can estimate latent values from the network’s random ReLU features at initialization. Does the trained network use that latent? The tree predicts exactly whose target would change if one latent value changed. A selective edit inside the network could test that prediction.' },
      ];
      return { beats, end, draw, notes, details, results: [control] };
    };
    const tail = rerun(S6);
    const DUR = tail.end;

    // ═══ 11. camera: frontal, slow push-ins, a few degrees of yaw on the 3-D cloud ══════════════
    const O: RGB = [0, 0, 0];
    // 0.6886 and 1.2243 are the height and the width of the frame at distance 1 (38° lens)
    const dW = (R3 + 1.6) / (BALL * 0.6886); // the ball of clusters fills the height of the left panel
    const dH = clamp(Math.max((2 * hext[0]) / (0.36 * 1.2243), (2 * hext[1]) / (0.36 * 0.6886)), 12, 110); // frames the largest cluster
    const dataY = (DATA_WINDOW.y0 + DATA_WINDOW.y1) / 2;
    const shData = mkShot(0, BLOB, dW, -14, 4, 38, LX, dataY); // 1 shows a flat figure; the camera already waits on the dataset
    const shHero0 = mkShot(S4B + 2.6, Hb, dH, -2, 4), shHero1 = mkShot(S5 - 0.1, Hb, dH * 0.97, 5, 4);
    const shots: Shot[] = [
      shData,
      mkShot(S3 - 0.2, BLOB, dW, -5, 4, 38, LX, dataY), // 2: the ball turns slowly
      mkShot(S3 + 1.4, O, 44.0, 0, 0, 38, 0.15), // 3 and 4: the ring, small, beside its latent tree
      mkShot(S4B, O, 43.4, 0, 0, 38, 0.15),
      shHero0, // 4: into the largest cluster
      shHero1,
      mkShot(S5 + 1.4, O, 34.0, 0, 0, 38, LX), // 5: the ring's points, then their inputs x, in the left panel
      mkShot(S6, O, 33.6, 0, 0, 38, LX), // the offline study slides use flat figures
      mkShot(DUR, O, 33.6, 0, 0, 38, LX),
    ];

    /** The base frame of a shot: world point → frame fractions (x from the left, y from the top), and back at a given depth. */
    const viewOf = (sh: Shot) => {
      const c = new THREE.PerspectiveCamera(sh.fov ?? 38, ASPECT, 0.1, 4000);
      c.position.set(sh.pos[0], sh.pos[1], sh.pos[2]);
      c.lookAt(sh.look[0], sh.look[1], sh.look[2]);
      c.updateMatrixWorld(true);
      c.updateProjectionMatrix();
      const r = new THREE.Vector3(), u = new THREE.Vector3(), b = new THREE.Vector3(), v = new THREE.Vector3();
      c.matrixWorld.extractBasis(r, u, b);
      const th = Math.tan(((sh.fov ?? 38) * Math.PI) / 360);
      return {
        fr: (x: number, y: number, z: number): [number, number] => { v.set(x, y, z).project(c); return [0.5 + v.x / 2, 0.455 - v.y / 2]; },
        depth: (x: number, y: number, z: number) => -((x - c.position.x) * b.x + (y - c.position.y) * b.y + (z - c.position.z) * b.z),
        at: (fx: number, fy: number, dz: number): RGB => {
          const hh = th * dz, hw = hh * ASPECT;
          v.copy(c.position).addScaledVector(b, -dz).addScaledVector(r, (fx - 0.5) * 2 * hw).addScaledVector(u, -(fy - 0.455) * 2 * hh);
          return [v.x, v.y, v.z];
        },
      };
    };

    // 4: the call-out on the largest subcluster goes where the largest cluster leaves the left panel empty, beside the
    // subcluster's core, and a leader line joins the two. Measured in the frame of the camera when the call-out comes.
    const callLead = new LineField(1, { width: 1.5 });
    callLead.object.name = 'percolation-subcluster-leader';
    let callPos: RGB = [callCore[0], callCore[1], callCore[2]], callAlign: 'left' | 'right' = 'left';
    if (callNode >= 0 && hs > 1) {
      const a = shHero0, b = shHero1, f = (T_CALL - a.t) / (b.t - a.t);
      const V = viewOf({ t: 0, pos: [lerp(a.pos[0], b.pos[0], f), lerp(a.pos[1], b.pos[1], f), lerp(a.pos[2], b.pos[2], f)], look: [lerp(a.look[0], b.look[0], f), lerp(a.look[1], b.look[1], f), lerp(a.look[2], b.look[2], f)], fov: a.fov });
      const FW = 1600, FH = 900, LW = 262, LH = 42, PAD = 6; // the call-out's size in frame pixels (two lines)
      const hx = new Float32Array(hs), hy = new Float32Array(hs);
      for (let p = 0; p < hs; p++) { const q = V.fr(W[(ho + p) * 3], W[(ho + p) * 3 + 1], W[(ho + p) * 3 + 2]); hx[p] = q[0] * FW; hy[p] = q[1] * FH; }
      const cq = V.fr(callCore[0], callCore[1], callCore[2]), cx = cq[0] * FW, cy = cq[1] * FH;
      const X0 = 0.02 * FW, X1 = 0.43 * FW, Y0 = TITLE_Y * FH + 22, Y1 = LEG_Y * FH - 22; // inside the panel, clear of its title and legend
      let best = Infinity, bx = X0, by = Y0;
      for (let y = Y0; y + LH <= Y1; y += 8)
        for (let x = X0; x + LW <= X1; x += 8) {
          const ex = clamp(cx, x, x + LW), ey = clamp(cy, y, y + LH), dist = Math.hypot(cx - ex, cy - ey);
          // the call-out stands wholly to one side of the core, so that its leader leaves from its near end; a short
          // leader counts for more than a few points hidden under the call-out's backing
          let cost = 4 * dist + (cx > x - 24 && cx < x + LW + 24 ? 1e5 : 0) + (dist < 24 ? 400 : 0);
          if (cost >= best) continue;
          for (let p = 0; p < hs && cost < best; p++) if (hx[p] > x - PAD && hx[p] < x + LW + PAD && hy[p] > y - PAD && hy[p] < y + LH + PAD) cost += 6;
          if (cost < best) { best = cost; bx = x; by = y; }
        }
      const right = bx + LW / 2 < cx; // the call-out stands left of the subcluster: it ends at its right edge
      callAlign = right ? 'right' : 'left';
      const dz = V.depth(callCore[0], callCore[1], callCore[2]);
      const ax = (right ? bx + LW : bx) / FW, ay = (by + LH / 2) / FH;
      callPos = V.at(ax, ay, dz);
      callLead.push(V.at(ax + (right ? 6 : -6) / FW, ay, dz), callCore, HUES[FT.col[callNode]], 1);
      callLead.commit();
    }
    group.add(callLead.object);

    // ═══ 12. texts that depend on the run ═════════════════════════════════════════════════════════
    // 2: the one check of the generator on screen; its numbers are in the notes
    const captionS2 =
      'In percolation, randomly occupied sites form clusters. At the critical density, clusters have no characteristic size. ' +
      'Brill and Carlson generate cluster sizes directly and use trees to model their connections. ' +
      `Fitting all generated sizes gives a cumulative tail slope of ${slopeRun}. The generator predicts ${slopeTh}.`;

    // ═══ 13. the notes: what each panel is, why it is here, how it was made ════════════════════════
    const pct = (x: number) => `${(100 * x).toFixed(x < 0.1 ? 1 : 0)}%`;
    /** A small probability as maths: 0.0123, or 4.56 × 10^−5. */
    const sci = (x: number) => {
      if (x >= 1e-3) return x.toPrecision(3);
      const e = Math.floor(Math.log10(x)), mnt = x / Math.pow(10, e);
      return `$${mnt.toFixed(2)} \\times 10^{${e}}$`;
    };
    /** A point of the ring at time t: its group (the highest merged latent above it), latents above it, lines so far. */
    const ringState = (p: number, t: number) => {
      const done = mergesDone(t);
      let node = p, up = 0, deg = 0;
      while (T.par[node] >= 0 && T.par[node] - s < done) { node = T.par[node]; up++; }
      for (let j = 0; j < done; j++) if (T.eu[j] === p || T.ev[j] === p) deg++;
      return { node, up, deg, done };
    };
    /** Pointer → chart coordinates (panel units), and chart units per frame pixel. */
    const toPanel = (p: ProbeInput) => {
      const o = p.project(...pt(0, 0)), ax = p.project(...pt(PW, 0)), ay = p.project(...pt(0, PH));
      const a = ax[0] - o[0], b = ay[0] - o[0], c = ax[1] - o[1], d = ay[1] - o[1], det = a * d - b * c;
      if (Math.abs(det) < 1e-9 || o[2] < -1 || o[2] > 1) return null;
      const ex = p.x - o[0], ey = p.y - o[1];
      return { X: ((d * ex - b * ey) / det) * PW, Y: ((-c * ex + a * ey) / det) * PH, upx: PW / Math.max(1e-6, Math.hypot(a, c)) };
    };
    /** Nearest visible CCDF mark, with a screen-pixel tolerance independent of chart scale. */
    const ccdfProbe = (t: number, p: ProbeInput, compareTheory = false): Probe | null => {
      if (ccdfOf(t) <= 0.5) return null;
      const drawn = Math.max(1, Math.ceil(ccIdx.length * sm(T_CC0 + 0.4, T_CC0 + 1.8, t)));
      let best = -1, bd = 12 * 12;
      for (let k = 0; k < drawn; k++) {
        const i = ccIdx[k], pos = pt(ccX(Math.log10(cc.s[i])), ccY(Math.log10(cc.p[i])));
        const at = p.project(...pos);
        if (at[2] < -1 || at[2] > 1) continue;
        const d = (at[0] - p.x) ** 2 + (at[1] - p.y) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      if (best < 0) return null;
      const sv = cc.s[best], pv = cc.p[best], share = Number((100 * pv).toPrecision(3));
      return {
        pos: pt(ccX(Math.log10(sv)), ccY(Math.log10(pv))),
        title: 'Cluster-size distribution',
        rows: [
          { label: 'cluster size', value: `${int(sv)} ${sv === 1 ? 'point' : 'points'}` },
          { label: 'clusters this size or larger', value: `${share}%` },
          ...(compareTheory ? [
            { label: 'clusters at or above this size', value: `${int(Math.round(pv * nPA))} of ${int(nPA)}` },
            { label: 'generator’s predicted share', value: sci(yuleSimonCCDF(sv, rhoTh)) },
            { label: 'measured share / predicted share', value: (pv / yuleSimonCCDF(sv, rhoTh)).toFixed(2) },
          ] : []),
        ],
      };
    };
    /** Opacity of each panel at t, with the same functions as update(). */
    const panelVis = (t: number) => ({
      concepts: treeOf(t), data: dataOf(t), ccdf: ccdfOf(t),
      ring: ringOf(t) * (1 - walkOf(t)), tree: mergeTreeOf(t),
      fam: famOf(t), bars: barsOf(t),
      walk: walkOf(t), build: buildOf(t), box: boxOf(t),
    });
    const M4 = new THREE.Matrix4();
    const chapter: Chapter = {
      id: 'percolation',
      pillar: 'data',
      kicker: 'DATA MODELS AND VALIDATION',
      title: 'CRITICAL PERCOLATION',
      caveat: 'Computed for this presentation',
      source: 'Brill & Carlson · Critical percolation as a synthetic data model for interpretability · arXiv:2606.20347',
      links: [
        { text: 'Brill and Carlson’s paper', url: 'https://arxiv.org/abs/2606.20347' },
        { text: 'Authors’ probe configuration', url: 'https://github.com/tomingebretsencarlson/mechinterp-percolation/blob/a5fd5846350c68ef5528374cbcfce266ed8e622d/configs/interp/probes.yaml' },
      ],
      duration: DUR,
      group,
      shots,
      interaction: {
        pick(t, p) {
          const target = datasetSlide(t) ? 'cloud' : clusterSlide(t) && famOf(t) > 0.5 ? 'cluster' : null;
          return target && cloudHit(p) ? target : null;
        },
        draggable: (region) => region === 'cloud' || region === 'cluster',
        drag(t, target, dx, dy) {
          const rotation = target === 'cloud' && datasetSlide(t) ? cloudRotation
            : target === 'cluster' && clusterSlide(t) ? clusterRotation : null;
          if (!rotation || !camNow) return;
          turn.setFromAxisAngle(_u, dx * 0.006);
          rotation.premultiply(turn);
          turn.setFromAxisAngle(_r, dy * 0.006);
          rotation.premultiply(turn).normalize();
        },
        wheel(t, target, delta) {
          if (datasetSlide(t) && target === 'cloud') cloudZoom = clamp(cloudZoom * Math.exp(-delta * 0.0012), 0.5, 2.4);
          if (clusterSlide(t) && target === 'cluster') clusterZoom = clamp(clusterZoom * Math.exp(-delta * 0.0012), 0.5, 2.4);
        },
        reset: resetCloud,
        changed: () => Math.abs(cloudZoom - 1) > 1e-6 || rotationChanged(cloudRotation) || clusterViewChanged(),
      },
      hover: ccdfProbe,
      details(t) {
        if (t >= S6) return tail.details(t);
        if (t < S2 || t >= S3) return undefined;
        return {
          title: 'Scale and checks of the generated data',
          body:
            `<p>This dataset contains ${int(nPts)} points with ${DIM} input coordinates and keeps every cluster. The paper uses 100 coordinates, with one cluster of 200,000 points and a separate dataset generated with 2 million points before filtering out clusters smaller than 500.</p>` +
            `<p>The cluster sizes come from preferential attachment. A new point starts a cluster with probability $p_{01}$ or joins one in proportion to its size. Fitting the Yule–Simon distribution uses all ${int(nPA)} cluster sizes. At $p_{01}=1/3$, the predicted cumulative tail slope is $-3/2$, as in critical mean-field percolation. The paper’s Figure 8 fits a power law only on cluster sizes of at least 35 points.</p>` +
            `<p>A second check counts points with one neighbor. Their share is ${degShare[0].toFixed(3)} across ${int(degClusters)} clusters of at least ${degMin} points, compared with the large-tree prediction $e^{-1}\\approx0.368$. Both properties are built into the generator. These checks test its implementation.</p>`,
        };
      },
      results: [
        {
          label: 'fitted cumulative tail slope for cluster sizes', measured: slopeRun, reference: `generator prediction ${slopeTh}`, board: true, group: 'check',
          say: `Fitting all cluster sizes gives a cumulative tail slope of ${slopeRun}. The generator predicts ${slopeTh}.`,
        },
        {
          label: `share of points with one neighbor (${int(degClusters)} clusters of at least ${degMin} points)`, measured: degShare[0].toFixed(3), reference: 'theory $e^{-1} \\approx 0.368$', group: 'check',
          say: degOk ? 'Points have one neighbor as often as theory predicts.' : 'Points have one neighbor more or less often than theory predicts.',
        },
        ...tail.results,
      ],
      beats: [
        { at: S1, caption: 'If a network recognizes sparrows, can we also read out the broader class of birds from its activations? Brill and Carlson build a hierarchy into synthetic data so researchers can compare a probe’s answers with known hidden variables.' },
        { at: S2, caption: captionS2 },
        { at: S3, caption: 'Shuffle the points around a ring. Each starts in its own group. Pick a random point and link it to a random point in the next group clockwise. Merging the groups creates a shared hidden variable, or <em>latent</em>.' },
        { at: S4, caption: `Each merge puts two groups under a new parent. A latent’s <em>depth</em> counts its ancestors. At a given depth, each latent defines a <em>subcluster</em> of points below it. Point P belongs to the one under A.` },
        { at: S4B, caption: `Cut the same way, the largest cluster of the dataset splits into subclusters from ${subFilm.min === 1 ? 'single points' : `${int(subFilm.min)} points`} up to ${int(subFilm.largest)} points. ` + 'A branch that ends at the cut contributes a single point. A branch that ends earlier has no representative at that depth.' },
        { at: S5, caption: 'A branching random walk gives each point its input coordinates $x$. Every link takes one step. Its direction depends on the latent hierarchy, so nearby relatives tend to occupy nearby regions of input space.' },
        {
          at: S5B,
          caption: 'Every latent and every point also gets a random value $z$. A point\'s target $y$, the value the network must predict, is the scaled sum along its path from the root. Training uses only $x$ and $y$. Researchers retain the tree and every $z$ as ground truth.',
          eq: 'y_x=\\frac{1}{\\sqrt{1+d_x}}\\sum_{i=0}^{d_x} z_i',
          eqNote: 'the scaling keeps the spread of y the same at every depth',
        },
        ...tail.beats.map((b, i) => (i === 0 ? { ...b, eq: '' } : b)),
      ],
      update(t, api) {
        const cam = api.camera;
        frame(cam);
        if (datasetSlide(t)) cloudPivot.quaternion.copy(cloudRotation);
        else cloudPivot.quaternion.identity();
        cloudPivot.scale.setScalar(datasetSlide(t) ? cloudZoom : 1);
        cloudPivot.updateMatrixWorld(true);
        clusterPivot.quaternion.copy(clusterRotation);
        clusterPivot.scale.setScalar(clusterZoom);
        clusterPivot.updateMatrixWorld(true);
        drawDemo(t, api);
        drawCoalescentExample(t, api);
        drawArch(t);
        if (arch.object.visible) frameCloud(t, cam);
        drawCharts(t, api);
        drawFlat(t, api);
        tail.draw(t, api);
        // titles, legend lines and call-outs of the left panel while it shows the data
        const o = { color: CSS.white } as const;
        const aW = dataOf(t);
        api.label('tc', scr(cam, TITLE_X, TITLE_Y), 'Generated percolation data', {
          ...o, alpha: aW, align: 'left', size: 'title',
          help: `<p>This generated dataset has ${int(nPts)} points with ${DIM} input coordinates per point. Each line links two neighbors in the same cluster.</p>`,
        });
        api.label('lc', scr(cam, TITLE_X, DATA_LEG_Y), `dot = point · line = link<small>${THEME === 'paper' ? 'darker' : 'brighter'} dots = larger clusters</small>`, { ...o, alpha: aW * 0.9, align: 'left', dy: -7 });
        const famT = famOf(t);
        if (famT > 0.01 && hs > 1) {
          api.label('te', scr(cam, TITLE_X, TITLE_Y), 'Largest cluster', {
            ...o, alpha: famT, align: 'left', size: 'title',
            help: '<p>Points in a subcluster share a color, which can recur in other branches. Gray points belong to branches that end above the cut.</p>',
          });
          api.label('le', scr(cam, TITLE_X, LEG_Y), 'colored by subcluster · gray = above the cut', { ...o, alpha: famT * 0.9, align: 'left' });
        }
        // the call-out on the largest subcluster, off the data, with its leader line (only at the film's own cut)
        // Keep the label anchor fixed while its leader follows the same cluster point after a
        // local turn. A core clipped out of the cloud's panel cannot carry a visible callout.
        _v.set(...callCore);
        if (clusterSlide(t) && clusterViewChanged()) _v.applyMatrix4(clusterContent.matrixWorld);
        callLead.positions.set([_v.x, _v.y, _v.z], 3);
        callLead.commit();
        _v.project(cam);
        const callX = 0.5 + _v.x / 2, callY = 0.5 - _v.y / 2;
        const callInPanel = !clusterViewChanged() || (callX > cloudScreen.x && callX < cloudScreen.z && callY > cloudScreen.y && callY < cloudScreen.w && _v.z > -1 && _v.z < 1);
        const aCall = hs > 1 && callNode >= 0 && callInPanel && (notesCutK() === null || notesCutK() === kFilm) ? famT * sm(T_CALL, T_CALL + 0.5, t) : 0;
        showL(callLead, aCall > 0.01);
        callLead.uIntensity.value = aCall;
        if (aCall > 0.01) api.label('call', callPos, `its largest subcluster, ${int(TH.cnt[callNode])} points<br>under one latent at depth $k = ${kFilm}$`, { color: callCss, alpha: aCall, align: callAlign, size: 'md', plate: true });
      },
      set(key, value, isDefault) {
        if (key === 'k') liveK = isDefault ? null : clamp(Math.round(+value), 0, kCap);
      },
      readouts(t): Readout[] {
        if (t >= T_CC0 && t < S3)
          return [
            { label: 'cumulative tail slope fitted to all sizes', value: neg(`-${fittedRho.toFixed(3)}`) },
            { label: `predicted cumulative tail slope${critical ? ' at criticality' : ''}`, value: slopeTh },
            { label: 'clusters of one or two points', value: `${int(nTiny)} of ${int(nPA)}`, detail: true },
            { label: 'points with one, two, three neighbors', value: degShare.map((x) => x.toFixed(3)).join(', '), detail: true, note: `in the ${int(degClusters)} clusters of at least ${degMin} points. The large-tree limit $e^{-1}/(j-1)!$ gives 0.368, 0.368, 0.184.` },
          ];
        if (t >= S4B && t < S5) {
          const k = kAt(t), st = subAt(k);
          return [
            { label: 'cut depth', value: String(k), detail: true },
            { label: 'subclusters, single points included', value: int(st.nSub), detail: true },
            { label: 'largest subcluster', value: `${int(st.largest)} of ${int(hs)} points`, detail: true },
            { label: 'points above the cut, in no subcluster', value: int(st.above), detail: true },
          ];
        }
        if (t >= S5B && t < S6)
          return [
            { label: 'depth of the marked point', value: String(pickD), detail: true },
            { label: 'sum of the values on its path', value: neg(f2(pathSum)), detail: true },
            { label: '$\\sqrt{1 + \\text{depth}}$', value: Math.sqrt(1 + pickD).toFixed(2), detail: true },
            { label: 'its target $y$', value: neg(f2(pickY)), detail: true },
          ];
        return [];
      },

      // ── contextual notes ──
      inspect(t) {
        const out: Hotspot[] = [];
        const cam = camNow;
        if (!cam) return out;
        const v = panelVis(t);
        const left = (): [number, number, number] => scr(cam, 0.031, TITLE_Y - 24 / 900);
        if (v.concepts > 0.5)
          out.push({
            id: 'concepts',
            pos: npt(CT[0].x - 86, CT[0].y),
            title: 'Classes and parts describe different relations',
            body: '<p>A sparrow belongs to the class of birds. Brill and Carlson call this a taxonomic hierarchy. A compositional hierarchy instead describes parts of a whole. Their generated data model the first kind.</p>',
          });
        if (v.data > 0.5)
          out.push({
            id: 'data',
            pos: left(),
            title: 'Which distances have meaning',
            body:
              `<p>Each cluster is projected ${view === 'pca' ? 'along its own three directions of greatest spread' : 'through the same random projection'}. The distance between different clusters has no meaning in this drawing.</p>`,
          });
        if (v.ccdf > 0.5)
          out.push({
            id: 'ccdf',
            pos: pt(0, PH + 71 * UPX),
            title: 'What the size check establishes',
            body: '<p>The size distribution is built into the generator. Agreement with its predicted slope checks that implementation. It does not test whether percolation describes natural data.</p>',
          });
        if (v.ring > 0.5 && t < S4)
          out.push({
            id: 'ring',
            pos: left(),
            title: 'Why larger groups are picked more often',
            body: '<p>Choosing a point uniformly chooses its group in proportion to its size. Brill and Carlson prove that with a uniformly shuffled order of point labels, this samples uniform random labeled trees. The diagram numbers the points by their positions around the ring.</p>',
          });
        if (v.tree > 0.5 && t >= S4 && t < S4B)
          out.push({
            id: 'cut',
            pos: npt(TAX + 2, TYT - 84),
            title: 'Keeping the ancestor relations',
            body: '<p>The paper calls the merge order a “construction device”. The latent hierarchy keeps which latents descend from which, and discards the times when they were created.</p>',
          });
        if (v.bars > 0.5)
          out.push({
            id: 'icicle',
            pos: pt(0, PH + 83 * UPX),
            title: 'Which branches are left out of the drawing',
            body: `<p>Bars representing fewer than ${icMin} points are hidden. Rows 0 to ${kTop - 1} are hidden too. Those first splits mostly peel off a few points, so the rows look almost alike.</p>`,
          });
        if (v.walk > 0.5 && t < S5B)
          out.push({
            id: 'walk',
            pos: left(),
            title: 'Equal steps can look unequal',
            body: `<p>The generator picks a starting point by descending the latent tree. It also assigns directions down that tree. The two children inherit a component along their parent’s direction and receive opposite sideways components. Every graph edge has the same length in the ${DIM}-coordinate input space. Showing only two projected coordinates changes the apparent lengths.</p>`,
          });
        if (v.build > 0.5)
          out.push({
            id: 'build',
            pos: pt(0, PH + 83 * UPX),
            title: 'Why new points cannot be predicted exactly',
            body:
              '<p>Points below one latent share its value and all values above it. Their targets are therefore correlated. Each point also has an independent value of its own, which a model cannot know for a new point.</p>',
          });
        return out.concat(tail.notes(t));
      },
      // ── the notes layer, pinned by a deep link or the frame grabber: the element under a frame pixel ──
      probe(t, p) {
        const cam = camNow;
        if (!cam) return null;
        const v = panelVis(t);

        // the latent tree (3, 4), drawn in frame pixels: the nearest leaf or latent that is drawn (not in its inset)
        if (v.tree > 0.5 && t < S4B) {
          const hang = sm(T_HANG0, T_HANG1, t), mergeTime = teachingStage(t) ? T_MERGE1 + 2 : t;
          let best = -1, bd = 12 * 12;
          for (let node = 0; node < NN; node++) {
            const on = node < s ? mergeTime >= T_RING0 + 0.3 + (2.1 * qOf(node)) / s : (mergeTime - tau[node - s]) / dur[node - s] >= 0.8;
            if (!on) continue;
            const c = p.project(...npt(tX[node] + teachingTreeShift(t), tY(node, hang)));
            const d = (c[0] - p.x) ** 2 + (c[1] - p.y) ** 2;
            if (d < bd) { bd = d; best = node; }
          }
          if (best >= 0) {
            const pos = npt(tX[best] + teachingTreeShift(t), tY(best, hang));
            const cutNow = t >= T_KD0 ? Math.round(cutK(t)) : -1;
            const famRow: Readout[] = [];
            if (cutNow >= 0) {
              const fi = cutIdx[cutNow][best];
              famRow.push({ label: `at the cut, depth ${cutNow}`, value: fi === -2 ? 'above the cut' : fi === -1 ? 'above the cut, in no subcluster' : fi === 0 ? 'in subcluster A' : 'in another subcluster' });
            }
            if (best < s) {
              const st = ringState(best, mergeTime), q0 = qOf(T.start[st.node]), c = T.cnt[st.node];
              return {
                pos,
                title: best === pk && !teachingStage(t) ? 'The marked point' : `Point at position ${qOf(best)} on the ring`,
                rows: [
                  { label: 'its group now', value: c > 1 ? `positions ${q0} to ${q0 + c - 1}` : 'itself only' },
                  { label: 'latents above it so far', value: String(st.up) },
                  { label: 'depth $d_x$ when the tree is complete', value: String(T.depth[best]) },
                ].concat(famRow),
              };
            }
            const j = best - s, ni = T.cnt[T.ca[j]], nj = T.cnt[T.cb[j]], q0 = qOf(T.start[best]), c = T.cnt[best];
            return {
              pos,
              title: `Latent of merge ${j + 1}${best === A && !teachingStage(t) && hlA(t) > 0.3 ? ' (A)' : ''}`,
              rows: [
                { label: 'sizes of the groups joined', value: `$${ni} + ${nj}$` },
                { label: 'points under it', value: `${c}, positions ${q0} to ${q0 + c - 1}` },
                { label: 'its depth', value: String(T.depth[best]) },
              ].concat(famRow),
              note: 'Every point under its bracket lies below this latent.',
            };
          }
        }

        // right half: the chart on screen
        if (v.ccdf > 0.5 || v.bars > 0.5 || v.build > 0.5) {
          const q = toPanel(p);
          if (q && q.X > -0.3 && q.X < (v.bars > 0.5 ? IW : PW) + 0.3 && q.Y > -0.3 && q.Y < PH + 0.3) {
            const { X, Y, upx } = q;
            if (v.ccdf > 0.5) {
              return ccdfProbe(t, p, true);
            }
            if (v.bars > 0.5) {
              // the bar under the pointer, in any row (the rows below the cut are on the chart too)
              const k = kTop + Math.floor(((PH - Y) / PH) * nRows);
              if (k < kTop || k > kBot || X < 0 || X > IW) return null;
              const leaf = (Math.min(hs - 1, Math.floor((X / IW) * hs)) + icRs) % hs;
              if (TH.depth[leaf] < k) return null;
              let node = leaf;
              for (let dd = TH.depth[leaf]; dd > k; dd--) node = TH.par[node];
              const c = TH.cnt[node];
              if (node < hs || c < icMin) return null;
              const x0 = (TH.start[node] - icRs + hs) % hs, j = node - hs;
              return {
                pos: pt(icX(x0 + c / 2), icY(k)),
                title: `Latent at depth $k = ${k}$`,
                rows: [
                  { label: 'points under it', value: `${int(c)} (${pct(c / hs)} of the cluster)` },
                  { label: 'positions in the cyclic order', value: `${int(x0)} to ${int(x0 + c - 1)}` },
                  { label: 'splits at depth $k + 1$ into', value: `$${ti(TH.cnt[TH.ca[j]])} + ${ti(TH.cnt[TH.cb[j]])}$` },
                  { label: 'subclusters at this depth', value: int(subAt(k).nSub) },
                ],
              };
            }
            if (v.build > 0.5) {
              // path sum: the term under the pointer, or the target's own bar
              const shown = Math.min(nbz, Math.floor(nbz * ramp(T_BLD0, T_BLD1, t) + 1e-6));
              if (Math.abs(X - BYX) < 12 * upx && t >= T_BLD1 + 0.5)
                return {
                  pos: pt(BYX, bY(pickY)),
                  title: 'Target $y$ of the marked point',
                  rows: [
                    { label: 'sum of the values on its path', value: f2(pathSum) },
                    { label: `$\\sqrt{1 + d_x}$ with $d_x = ${pickD}$`, value: Math.sqrt(1 + pickD).toFixed(2) },
                    { label: 'target $y$', value: f2(pickY) },
                  ],
                };
              const i = Math.floor((X - BX0) / bPitch);
              if (i < 0 || i >= shown) return null;
              return {
                pos: pt(bX(i), bY(pathZ[i])),
                title: i === pickD ? 'The point itself' : i === aBar ? `Latent A, at depth ${i} on the path` : `Latent at depth ${i} on the path`,
                rows: [
                  { label: `value $z_{${i}}$`, value: f2(pathZ[i]) },
                  { label: i === pickD ? 'points' : 'points under it', value: i === pickD ? '1' : `${int(pathSize[i])} of ${int(s)}` },
                  { label: `running sum$\\,/\\sqrt{${nbz}}$ up to here`, value: f2(runZ[i]) },
                ],
              };
            }
          }
        }

        // left half: the points of the ring's cluster (on the ring or at their inputs), or the dataset cloud
        let best = -1, kind = 0, bd = 12 * 12; // kind 0: a point of the ring's cluster, 1: a point of the dataset
        if ((v.ring > 0.5 && !teachingStage(t)) || v.walk > 0.5) {
          const P3 = nodes.positions;
          for (let q = 0; q < s; q++) {
            if (t < T_RING0 + (2.1 * qOf(q)) / s) continue;
            const sp = p.project(P3[q * 3], P3[q * 3 + 1], P3[q * 3 + 2]);
            if (sp[2] < -1 || sp[2] > 1) continue;
            const d = (sp[0] - p.x) ** 2 + (sp[1] - p.y) ** 2;
            if (d < bd) { bd = d; best = q; }
          }
        }
        const panel = arch.object.visible ? cloudHit(p) : null;
        if (panel) {
          // own projection loop: up to 60,000 points per frame
          const heroOnly = t >= S4B;
          M4.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).multiply(heroOnly && clusterSlide(t) && clusterViewChanged() ? clusterContent.matrixWorld : cloudContent.matrixWorld);
          const e = M4.elements;
          const fx = e[0], fy = e[4], fz = e[8], fw = e[12], gx = e[1], gy = e[5], gz = e[9], gw = e[13], hx = e[3], hy = e[7], hz = e[11], hw = e[15];
          const fullW = (panel.x1 - panel.x0) / (cloudScreen.z - cloudScreen.x);
          const fullH = (panel.y1 - panel.y0) / (cloudScreen.w - cloudScreen.y);
          const offsetX = panel.x0 - fullW * cloudScreen.x, offsetY = panel.y0 - fullH * cloudScreen.y;
          const i0 = heroOnly ? ho : 0, i1 = heroOnly ? ho + hs : n;
          let bd2 = Math.min(bd, 100), bi = -1;
          for (let i = i0; i < i1; i++) {
            const c = cl[i];
            if (rc[c] < 0.3 || c === demo) continue;
            const x = W[i * 3], y = W[i * 3 + 1], z = W[i * 3 + 2];
            const w = hx * x + hy * y + hz * z + hw;
            if (w <= 0) continue;
            const ux = 0.5 + (fx * x + fy * y + fz * z + fw) / (2 * w);
            const uy = 0.5 - (gx * x + gy * y + gz * z + gw) / (2 * w);
            if (ux <= cloudScreen.x || ux >= cloudScreen.z || uy <= cloudScreen.y || uy >= cloudScreen.w) continue;
            const sx = ux * fullW + offsetX, sy = uy * fullH + offsetY;
            const d = (sx - p.x) * (sx - p.x) + (sy - p.y) * (sy - p.y);
            if (d < bd2) { bd2 = d; bi = i; }
          }
          if (bi >= 0) { best = bi; kind = 1; }
        }
        if (best < 0) return null;

        if (kind === 0) {
          const q = best, P3 = nodes.positions, pos: [number, number, number] = [P3[q * 3], P3[q * 3 + 1], P3[q * 3 + 2]];
          const gi = doff + q;
          if (v.walk > 0.5) {
            const fi = cutIdx[kd][q];
            return {
              pos,
              title: q === pk ? 'The marked point' : 'Point of the illustrated cluster',
              rows: [
                { label: 'depth $d_x$ (latents above it)', value: String(T.depth[q]) },
                { label: `at the cut, depth ${kd}`, value: fi === -1 ? 'above the cut, in no subcluster' : fi === 0 ? 'in subcluster A' : 'in another subcluster' },
                { label: 'lines at this point', value: String(data.degree[gi]) },
                { label: 'target $y_x$', value: data.y[gi].toFixed(3) },
              ],
              note: 'Its position shows two projected coordinates of its input $x$.',
            };
          }
          if (t >= T_KD0 - 0.4) {
            const k = Math.round(cutK(t)), fi = cutIdx[k][q];
            let ancestor = q;
            while (T.depth[ancestor] > k) ancestor = T.par[ancestor];
            return {
              pos,
              title: q === pk ? 'The marked point' : `Point at position ${qOf(q)}`,
              rows: [
                { label: 'cut depth', value: String(k) },
                { label: 'subcluster', value: fi === -1 ? 'none: its branch ends above the cut' : fi === 0 ? `A, ${T.cnt[ancestor]} points` : `${T.cnt[ancestor]} points` },
                { label: 'depth $d_x$', value: String(T.depth[q]) },
                { label: 'lines at this point', value: String(data.degree[gi]) },
              ],
            };
          }
          const st = ringState(q, t), q0 = qOf(T.start[st.node]), c = T.cnt[st.node];
          const rows: Readout[] = [
            { label: 'its group now', value: c > 1 ? `positions ${q0} to ${q0 + c - 1}` : 'itself only' },
            { label: 'group size', value: `${c} of ${s}` },
            { label: 'groups left', value: String(s - st.done) },
            { label: 'lines at this point so far', value: String(st.deg) },
          ];
          if (hlA(t) > 0.3) rows.push({ label: 'below latent A', value: inA[q] ? 'yes' : 'no' });
          return { pos, title: q === pk ? 'The marked point' : `Point at position ${qOf(q)}`, rows };
        }

        const i = best, c = cl[i], sz = size[c], dxv = data.leafDepth[i], yv = data.y[i];
        _v.set(W[i * 3], W[i * 3 + 1], W[i * 3 + 2]).applyMatrix4(clusterSlide(t) && clusterViewChanged() && c === hero ? clusterContent.matrixWorld : cloudContent.matrixWorld);
        const pos: [number, number, number] = [_v.x, _v.y, _v.z];
        const rows: Readout[] = [];
        if (v.fam > 0.5 && c === hero) {
          const k = kAt(t), lp = i - ho;
          rows.push({ label: 'depth $d_x$', value: String(dxv) });
          if (TH.depth[lp] < k) rows.push({ label: `subcluster at $k = ${k}$`, value: 'none ($d_x < k$)' });
          else {
            let node = lp;
            for (let dd = TH.depth[lp]; dd > k; dd--) node = TH.par[node];
            rows.push({ label: `subcluster at $k = ${k}$`, value: `${int(TH.cnt[node])} points (${pct(TH.cnt[node] / hs)})` });
          }
          rows.push({ label: 'target $y_x$', value: yv.toFixed(3) });
          return { pos, title: 'Point of the largest cluster', rows, note: 'Its subcluster is the set of points under its latent at depth $k$.' };
        }
        rows.push({ label: 'cluster size $s$', value: c === hero ? `${int(sz)} (the largest)` : int(sz) });
        rows.push({ label: 'depth $d_x$', value: String(dxv) });
        if (sz > 1) rows.push({ label: 'asymptotic mean depth $\\sqrt{\\pi s/2}$', value: Math.sqrt((Math.PI * sz) / 2).toFixed(1) });
        rows.push({ label: 'lines at this point', value: String(data.degree[i]) });
        rows.push({ label: 'target $y_x$', value: yv.toFixed(3) });
        return { pos, title: c === hero ? 'Point of the largest cluster' : `Point of cluster ${int(c)}`, rows };
      },
    };
    return chapter;
  },
};

export default percolation;
