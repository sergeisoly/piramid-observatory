import * as THREE from 'three/webgpu';
import type { Chapter, ChapterFactory, Hotspot, Probe } from '../engine/types';
import { C, CSS, diverge, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { mulberry32, gaussian, smooth, easeInOut } from '../lib/math';
import { resolveParams } from '../engine/params';

/**
 * Coarse-graining, shown on a toy. A random tree is sampled the way chapter 1 builds its clusters (a
 * cyclic coalescent, re-implemented here). A greedy rule then replaces every connected group of at least
 * b units by one unit, level after level, until one unit is left. Each level is drawn as a square of its
 * own, from the finest on the left to the coarsest on the right. On the first slide one group per level is
 * followed, with a line from each member to the unit it becomes.
 *
 * Slide 7.2 gives every node a made-up value: noise of its own plus a shift shared by one branch. A unit
 * shows the mean value of its nodes. A plot above the small squares shows how far the shift stands out above
 * the noise left on a unit, level by level: it grows up to the branch's own level (7.2) and shrinks above it
 * (7.3). On the top unit the branch's part of the mean is exactly its shift times its share of the nodes, and
 * how far it stands out is its peak times the square root of that share (7.3). Slide 7.4 adds a small branch,
 * which peaks lower and ends below the noise. Nothing here is computed about a network.
 */
const T_OPEN = 0.4, T_VAL = 8.2, T_ABOVE = 12.4, T_SMALL = 16.0, DUR = 20.2;
// slide 7.1: the levels come in one after another, and all have landed before slide 7.2 begins
const GROW0 = 1.3, GROW_LEN = 1.2, GROW_END = T_VAL - 0.7;
// the shift on the branch (and, with the other sign, on the small branch), in units of the noise's spread
const SHIFT = 0.6;
// a unit's mean prints at full color from this size on, at this share of the ramp's darkest ink
const VSCALE = 0.9, CK = 0.72;
const PAPER = 'https://arxiv.org/abs/2602.05184';

// the figure is laid out in pixels of the base frame; the camera below shows 50 of them per world unit
const PX = 1 / 50;
// framing.ts uses one box for all four slides to keep the figure from re-zooming.
const CAM_X = -0.84, CAM_Y = 1.0, CAM_Z = 26.45, FOV = 38;

const fmt = (n: number) => n.toLocaleString('en-US');
/** A value with its sign: two decimals, three below 0.095 (so that a small share does not print as 0.00). */
const val = (x: number) => {
  const s = Math.abs(x) >= 0.095 ? Math.abs(x).toFixed(2) : Math.abs(x).toFixed(3);
  return (x < 0 && +s > 0 ? '−' : '') + s;
};
const signed = (x: number) => (x > 0 ? '+' : '') + val(x);
/** A ratio: two decimals below 10, one from 10 on. */
const ratio = (x: number) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));

/**
 * The zero line of a field h sampled on an n × n grid from (x0, y0) with spacing `step` (marching squares),
 * as segments x0, y0, x1, y1 one after another. A saddle cell is resolved by the mean of its corners.
 */
function zeroLine(h: Float32Array, n: number, x0: number, y0: number, step: number): number[] {
  const out: number[] = [];
  const cut = (p: number, q: number, ax: number, ay: number, bx: number, by: number) => {
    const s = p / (p - q);
    out.push(ax + (bx - ax) * s, ay + (by - ay) * s);
  };
  for (let j = 0; j + 1 < n; j++)
    for (let i = 0; i + 1 < n; i++) {
      // corners: a (i, j), b (i + 1, j), c (i + 1, j + 1), d (i, j + 1)
      const a = h[j * n + i], b = h[j * n + i + 1], c = h[(j + 1) * n + i + 1], d = h[(j + 1) * n + i];
      const k = (a > 0 ? 1 : 0) | (b > 0 ? 2 : 0) | (c > 0 ? 4 : 0) | (d > 0 ? 8 : 0);
      if (k === 0 || k === 15) continue;
      const X = x0 + i * step, Y = y0 + j * step, X1 = X + step, Y1 = Y + step;
      const ab = () => cut(a, b, X, Y, X1, Y), bc = () => cut(b, c, X1, Y, X1, Y1);
      const dc = () => cut(d, c, X, Y1, X1, Y1), ad = () => cut(a, d, X, Y, X, Y1);
      const joined = a + b + c + d > 0;
      switch (k) {
        case 1: case 14: ab(); ad(); break;
        case 2: case 13: ab(); bc(); break;
        case 3: case 12: ad(); bc(); break;
        case 4: case 11: bc(); dc(); break;
        case 6: case 9: ab(); dc(); break;
        case 7: case 8: ad(); dc(); break;
        case 5: if (joined) { ab(); bc(); ad(); dc(); } else { ab(); ad(); bc(); dc(); } break;
        case 10: if (joined) { ab(); ad(); bc(); dc(); } else { ab(); bc(); ad(); dc(); } break;
      }
    }
  return out;
}

/**
 * Greedy bottom-up grouping of a rooted tree given parents in topological (parents-first) order: a
 * subtree closes into one unit as soon as it holds B units; the group at the root takes what is left.
 */
function groupTree(parent: Int32Array, B: number) {
  const m = parent.length;
  const acc = new Int32Array(m).fill(1), cut = new Uint8Array(m);
  for (let x = m - 1; x > 0; x--) {
    if (acc[x] >= B) cut[x] = 1;
    else acc[parent[x]] += acc[x];
  }
  const unitOf = new Int32Array(m);
  const qparent: number[] = [];
  let nb = 0;
  for (let x = 0; x < m; x++) {
    if (x === 0 || cut[x]) {
      unitOf[x] = nb++;
      qparent.push(x === 0 ? 0 : unitOf[parent[x]]);
    } else unitOf[x] = unitOf[parent[x]];
  }
  return { unitOf, qparent: Int32Array.from(qparent), nb };
}

/** Cell (x, y) of the d-th step of a Hilbert curve that fills an n × n grid (n a power of two). */
function hilbert(n: number, d: number): [number, number] {
  let x = 0, y = 0, t = d;
  for (let s = 1; s < n; s *= 2) {
    const rx = 1 & (t / 2), ry = 1 & (t ^ rx);
    if (ry === 0) {
      if (rx === 1) { x = s - 1 - x; y = s - 1 - y; }
      const tmp = x; x = y; y = tmp;
    }
    x += s * rx; y += s * ry;
    t = Math.floor(t / 4);
  }
  return [x, y];
}

const scales: ChapterFactory = {
  id: 'scales',
  label: 'coarse-graining · levels of a random tree',
  controls: [
    { key: 'n', label: '$N$  tree nodes', kind: 'select', value: 4096, options: [1024, 2048, 4096, 8192].map((v) => ({ value: v, label: v.toLocaleString('en-US') })), hint: 'Size of the random tree that gets coarse-grained.' },
    { key: 'block', label: '$b$  units per group', kind: 'range', value: 3, min: 2, max: 8, step: 1, hint: 'The rule closes a connected group into one unit once it holds at least $b$ units; the group left at the root may be smaller. Larger $b$ means fewer, coarser levels.' },
    { key: 'seed', label: 'seed', kind: 'range', value: 2026, min: 1, max: 4000, step: 1, hint: 'A different tree with the same statistics.' },
  ],
  async build(ctx, params) {
    const P = resolveParams(scales.controls, params);
    const N = P.n as number, BLOCK = P.block as number;
    const rnd = mulberry32(P.seed as number);
    const group = new THREE.Group();

    // ── 1. cyclic coalescent: a random tree on N nodes whose shape has the law of a uniform labeled tree ──
    const eu = new Int32Array(N - 1), ev = new Int32Array(N - 1);
    {
      const nxt = new Int32Array(N), st = new Int32Array(N).fill(1), uf = new Int32Array(N), head = new Int32Array(N);
      for (let i = 0; i < N; i++) { nxt[i] = (i + 1) % N; uf[i] = i; head[i] = i; }
      const find = (x: number) => { while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
      for (let k = 0; k < N - 1; k++) {
        const u = (rnd() * N) | 0;
        const a = head[find(u)], b = nxt[a];
        const v = (b + ((rnd() * st[b]) | 0)) % N;
        eu[k] = u; ev[k] = v;
        st[a] += st[b]; nxt[a] = nxt[b];
        const ru = find(u), rv = find(v);
        uf[rv] = ru;
        head[ru] = a;
      }
    }
    const adjHead = new Int32Array(N).fill(-1), adjNext = new Int32Array(2 * (N - 1)), adjTo = new Int32Array(2 * (N - 1));
    for (let k = 0, e = 0; k < N - 1; k++) {
      adjTo[e] = ev[k]; adjNext[e] = adjHead[eu[k]]; adjHead[eu[k]] = e++;
      adjTo[e] = eu[k]; adjNext[e] = adjHead[ev[k]]; adjHead[ev[k]] = e++;
    }
    // root the tree at its centroid; relabel nodes in BFS order
    const bfsFrom = (root: number) => {
      const order = new Int32Array(N), par = new Int32Array(N).fill(-1);
      let qh = 0, qt = 0;
      order[qt++] = root; par[root] = root;
      while (qh < qt) {
        const x = order[qh++];
        for (let e = adjHead[x]; e >= 0; e = adjNext[e]) {
          const y = adjTo[e];
          if (par[y] >= 0) continue;
          par[y] = x; order[qt++] = y;
        }
      }
      const sub = new Int32Array(N).fill(1);
      for (let i = N - 1; i > 0; i--) sub[par[order[i]]] += sub[order[i]];
      return { order, par, sub };
    };
    let t0 = bfsFrom(0), root = 0;
    for (;;) {
      let heavy = -1;
      for (let e = adjHead[root]; e >= 0; e = adjNext[e]) {
        const y = adjTo[e];
        if (t0.par[y] === root && y !== root && t0.sub[y] > N / 2) heavy = y;
      }
      if (heavy < 0) break;
      root = heavy;
      t0 = bfsFrom(root);
    }
    const rank = new Int32Array(N);
    for (let i = 0; i < N; i++) rank[t0.order[i]] = i;
    const par0 = new Int32Array(N), depth = new Int32Array(N), sub0 = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      par0[i] = rank[t0.par[t0.order[i]]];
      sub0[i] = t0.sub[t0.order[i]];
      depth[i] = i === 0 ? 0 : depth[par0[i]] + 1;
    }
    await ctx.log('tree ✓');

    // ── 2. drawing of the finest level: the nodes in depth-first order along a Hilbert curve, so that every
    // branch of the tree (a run of that order) fills a compact patch of an evenly filled square ──
    const childHead = new Int32Array(N).fill(-1), childNext = new Int32Array(N).fill(-1);
    for (let x = N - 1; x > 0; x--) { childNext[x] = childHead[par0[x]]; childHead[par0[x]] = x; }
    const side = 1 << Math.ceil(Math.log2(Math.sqrt(N)));
    const px = new Float32Array(N), pz = new Float32Array(N);
    {
      const stack = [0];
      let k = 0;
      while (stack.length) {
        const x = stack.pop()!;
        const [hx, hy] = hilbert(side, k++);
        px[x] = (hx + 0.5) / side - 0.5; pz[x] = (hy + 0.5) / side - 0.5;
        const kids: number[] = [];
        for (let c = childHead[x]; c >= 0; c = childNext[c]) kids.push(c);
        for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]);
      }
    }

    // ── 3. levels: group the tree, then the tree of units, until one unit is left ──
    const unitOfLeaf: Int32Array[] = [Int32Array.from({ length: N }, (_, i) => i)];
    const parents: Int32Array[] = [par0];
    const counts: number[] = [N];
    while (counts[counts.length - 1] > 1 && counts.length < 16) {
      const prev = parents[parents.length - 1];
      const { unitOf, qparent, nb } = groupTree(prev, BLOCK);
      const leafMap = new Int32Array(N), last = unitOfLeaf[unitOfLeaf.length - 1];
      for (let i = 0; i < N; i++) leafMap[i] = unitOf[last[i]];
      unitOfLeaf.push(leafMap);
      parents.push(qparent);
      counts.push(nb);
    }
    const L = counts.length;
    const size: Int32Array[] = counts.map((c, l) => {
      const s = new Int32Array(c);
      for (let i = 0; i < N; i++) s[unitOfLeaf[l][i]]++;
      return s;
    });
    const seq = counts.map(fmt).join(' → ');
    await ctx.log(`levels ${seq}`);
    const off: number[] = [];
    let total = 0;
    for (let l = 0; l < L; l++) { off.push(total); total += counts[l]; }
    // unit → the unit one level up that takes it in, and unit → how many units of the level below it holds
    const ups: Int32Array[] = [], kids: Int32Array[] = [new Int32Array(0)];
    for (let l = 0; l + 1 < L; l++) {
      const up = new Int32Array(counts[l]);
      for (let i = 0; i < N; i++) up[unitOfLeaf[l][i]] = unitOfLeaf[l + 1][i];
      const kc = new Int32Array(counts[l + 1]);
      for (let b = 0; b < counts[l]; b++) kc[up[b]]++;
      ups.push(up);
      kids.push(kc);
    }

    // ── 4. the values: noise on every node, a shift on one branch, the other shift on a small branch ──
    // The branch: the unit, on any level below the top two, closest to a sixth of the tree in size (not the
    // root's unit). Between its own level and the top its shift then falls by about √(1/6) ≈ 0.4 on any tree.
    let lvB = 1, uB = -1;
    {
      let best = Infinity;
      for (let l = 1; l <= L - 2; l++)
        for (let b = 1; b < counts[l]; b++) {
          const d = Math.abs(Math.log(size[l][b] / (N / 6)));
          if (d < best) { best = d; lvB = l; uB = b; }
        }
    }
    const hasB = uB >= 0;
    // the small branch: D_MIN to D_MAX nodes: enough to stand out three times above the noise on its own unit, and few enough, so that the top unit keeps its shift below the noise
    // (SHIFT · n / √N < 1), on a level below the branch's
    const D_MIN = Math.ceil((3 / SHIFT) ** 2), D_MAX = Math.floor((0.94 * Math.sqrt(N)) / SHIFT);
    const hasD = hasB && lvB >= 2;
    const inB = new Uint8Array(N);
    if (hasB) for (let i = 0; i < N; i++) inB[i] = unitOfLeaf[lvB][i] === uB ? 1 : 0;
    // turn or mirror the square (one of its eight symmetries) so that the branch sits at its upper right: its
    // label then has room beside the finest square, above the next one
    {
      let bx = 0, by = 0, nb = 0;
      for (let i = 0; i < N; i++) if (inB[i]) { bx += px[i]; by += pz[i]; nb++; }
      bx /= Math.max(1, nb); by /= Math.max(1, nb);
      const sym = (k: number, x: number, y: number): [number, number] => {
        const a = k & 4 ? [y, x] : [x, y];
        return [k & 1 ? -a[0] : a[0], k & 2 ? -a[1] : a[1]];
      };
      let bestK = 0, bestS = -Infinity;
      for (let k = 0; k < 8; k++) { const [x, y] = sym(k, bx, by); const s = x + 0.7 * y; if (s > bestS) { bestS = s; bestK = k; } }
      for (let i = 0; i < N; i++) [px[i], pz[i]] = sym(bestK, px[i], pz[i]);
    }
    // every unit at the mean position of its nodes (the drawing's only mean of positions)
    const fx = new Float32Array(total), fz = new Float32Array(total);
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < N; i++) {
        const j = off[l] + unitOfLeaf[l][i];
        fx[j] += px[i]; fz[j] += pz[i];
      }
      for (let b = 0; b < counts[l]; b++) { const j = off[l] + b; fx[j] /= size[l][b]; fz[j] /= size[l][b]; }
    }
    // each level is scaled to fill its square: its units' box, centered, half-width `ext`
    const box = counts.map((c, l) => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let b = 0; b < c; b++) { const j = off[l] + b; x0 = Math.min(x0, fx[j]); x1 = Math.max(x1, fx[j]); y0 = Math.min(y0, fz[j]); y1 = Math.max(y1, fz[j]); }
      return { mx: (x0 + x1) / 2, my: (y0 + y1) / 2, ext: Math.max(1e-6, (x1 - x0) / 2, (y1 - y0) / 2) };
    });
    // the small branch: a whole subtree of the tree (no part of it went to another unit), so that its nodes are
    // one run of the depth-first order and one patch; outside the branch, on a level below the branch's, of
    // D_MIN to D_MAX nodes, away from the middle of its square and toward its upper left (under its label, and
    // apart from the branch at the upper right), near 50 nodes
    let uD = -1, lvD = 1;
    if (hasD) {
      let best = Infinity;
      for (let pass = 0; pass < 2 && uD < 0; pass++)
        for (let l = 1; l < lvB; l++) {
          const rep = new Int32Array(counts[l]).fill(-1); // the unit's first node in BFS order: the top of its subtree
          for (let i = 0; i < N; i++) if (rep[unitOfLeaf[l][i]] < 0) rep[unitOfLeaf[l][i]] = i;
          const { mx, my, ext } = box[l];
          for (let b = 1; b < counts[l]; b++) {
            const s = size[l][b];
            if (s < D_MIN || s > D_MAX || inB[rep[b]] || sub0[rep[b]] !== s) continue;
            const j = off[l] + b, x = (fx[j] - mx) / ext, y = (fz[j] - my) / ext;
            if (pass === 0 && Math.max(Math.abs(x), Math.abs(y)) < 0.45) continue;
            let da = Math.abs(Math.atan2(y, x) - 0.75 * Math.PI);
            if (da > Math.PI) da = 2 * Math.PI - da;
            const score = da + 0.6 * Math.abs(Math.log(s / 50));
            if (score < best) { best = score; uD = b; lvD = l; }
          }
        }
    }
    const inD = new Uint8Array(N);
    if (uD >= 0) for (let i = 0; i < N; i++) inD[i] = unitOfLeaf[lvD][i] === uD ? 1 : 0;
    const showD = uD >= 0;
    let nB = 0, nD = 0;
    for (let i = 0; i < N; i++) { nB += inB[i]; nD += inD[i]; }

    const g = gaussian(mulberry32(((P.seed as number) * 7919 + 17) >>> 0));
    const noise = new Float32Array(N);
    for (let i = 0; i < N; i++) noise[i] = g();
    // per unit: the mean of the noise, the branch's share of the shift and the small branch's share
    const nMean = new Float32Array(total), bPart = new Float32Array(total), dPart = new Float32Array(total);
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < N; i++) {
        const j = off[l] + unitOfLeaf[l][i];
        nMean[j] += noise[i]; bPart[j] += inB[i]; dPart[j] += inD[i];
      }
      for (let b = 0; b < counts[l]; b++) {
        const j = off[l] + b, s = size[l][b];
        nMean[j] /= s; bPart[j] *= SHIFT / s; dPart[j] *= -SHIFT / s;
      }
    }
    // Log the root mean square of the sampled noise means across each level's units.
    const noiseRms = counts.map((c, l) => {
      let s = 0;
      for (let b = 0; b < c; b++) s += nMean[off[l] + b] ** 2;
      return Math.sqrt(s / c);
    });
    // For the small branch's outline, find the unit containing its first node at each level.
    const dUnit = counts.map((_, l) => (showD ? unitOfLeaf[l][inD.indexOf(1)] : -1));
    // on every level, the units that hold nodes of the branch (of the small branch)
    const member = (inSet: Uint8Array) => counts.map((c, l) => { const m = new Uint8Array(c); for (let i = 0; i < N; i++) if (inSet[i]) m[unitOfLeaf[l][i]] = 1; return m; });
    const bMember = member(inB), dMember = member(inD);
    /**
     * How far a shift stands out above the noise on a level: the shift a unit carries, over the noise averaging
     * leaves on a unit of n nodes (1/√n). Below the branch's own level the branch is several units, each with the
     * whole shift; their sizes enter as a root mean square. From its own level on, one unit holds it.
     */
    const stand = (inSet: Uint8Array, mem: Uint8Array[], lvOwn: number, shift: number) =>
      counts.map((c, l) => {
        if (l >= lvOwn) { const u = unitOfLeaf[l][inSet.indexOf(1)], s = size[l][u]; return Math.abs((shift * (inSetCount(inSet) / s))) * Math.sqrt(s); }
        let s2 = 0, k = 0;
        for (let b = 0; b < c; b++) if (mem[l][b]) { s2 += size[l][b]; k++; }
        return Math.abs(shift) * Math.sqrt(s2 / Math.max(1, k));
      });
    const inSetCount = (inSet: Uint8Array) => { let n = 0; for (let i = 0; i < N; i++) n += inSet[i]; return n; };
    const standB = hasB ? stand(inB, bMember, lvB, SHIFT) : counts.map(() => 0);
    const standD = showD ? stand(inD, dMember, lvD, SHIFT) : counts.map(() => 0);
    const bTop = bPart[off[L - 1]];
    // The wording on slide 7.2 checks whether the ratio rises at every level up to the branch's own.
    const riseB = hasB && standB.every((v, l) => l === 0 || l > lvB || v > standB[l - 1]);
    await ctx.log(`noise ${noiseRms.map((x) => x.toFixed(3)).join(' → ')}`);

    // ── 5. layout: one square per level, finest on the left; squares shrink with the number of units ──
    const hPx = counts.map((c, l) => (l === 0 ? 200 : c === 1 ? 0 : Math.max(14, 80 * Math.pow(c / counts[Math.min(1, L - 1)], 0.25))));
    const slot = (l: number) => (counts[l] === 1 ? 52 : 2 * hPx[l]);
    const cxPx: number[] = [];
    let xPx = 0;
    for (let l = 0; l < L; l++) {
      if (l > 0) xPx += l === 1 ? 34 : counts[l] === 1 ? 34 : 22;
      cxPx.push(xPx + slot(l) / 2);
      xPx += slot(l);
    }
    const K = Math.min(1, 1050 / xPx) * PX; // world units per layout pixel
    const BASE = -150; // the squares stand on this line (layout pixels below the figure's center line)
    const cx = cxPx.map((x) => (x - xPx / 2) * K);
    const cy = counts.map((c, l) => (BASE + (c === 1 ? 26 : hPx[l])) * K);
    const hW = hPx.map((h) => h * K);
    /** Where the point (x, z) of the finest drawing lands in the square of level l. */
    const inSq = (l: number, x: number, z: number): [number, number, number] => {
      const { mx, my, ext } = box[l], k = hW[l] / ext;
      return [cx[l] + (x - mx) * k, cy[l] + (z - my) * k, 0];
    };
    const home = (l: number, b: number) => (counts[l] === 1 ? ([cx[l], cy[l], 0] as [number, number, number]) : inSq(l, fx[off[l] + b], fz[off[l] + b]));
    // a unit begins its flight at the mean of its nodes as drawn in the square to its left
    const from = (l: number, b: number) => inSq(l - 1, fx[off[l] + b], fz[off[l] + b]);

    // dots: about the spacing of their level, and a little larger for a unit that holds more nodes
    const dotPx = new Float32Array(total);
    for (let l = 0; l < L; l++) {
      const spacing = counts[l] === 1 ? 36 : (2 * hPx[l]) / Math.sqrt(counts[l]);
      const base = Math.min(22, Math.max(3.4, (l === 0 ? 0.66 : 0.62) * spacing));
      const m = N / counts[l];
      for (let b = 0; b < counts[l]; b++) dotPx[off[l] + b] = l === 0 ? base : base * Math.min(1.5, Math.max(0.7, Math.cbrt(size[l][b] / m)));
    }
    const INK = C.white;
    // the light gray every unit keeps under its color, so that a unit near zero still shows; a touch of the
    // ochre takes the blue out of the key ink, so that this gray does not read as a pale teal
    const GRAY: RGB = [INK[0] * 0.1 + C.amber[0] * 0.05, INK[1] * 0.1 + C.amber[1] * 0.05, INK[2] * 0.1 + C.amber[2] * 0.05];
    const pts = new PointField({ count: total, size: 1 });
    for (let j = 0; j < total; j++) pts.sizes[j] = (dotPx[j] * K) / 0.52; // the visible disc is about half the sprite
    group.add(pts.object);
    const target = new Float32Array(total * 3), start = new Float32Array(total * 3);
    for (let l = 0; l < L; l++)
      for (let b = 0; b < counts[l]; b++) {
        const j = off[l] + b, h = home(l, b), s = l === 0 ? h : from(l, b);
        target.set(h, j * 3); start.set(s, j * 3);
      }

    /**
     * The outline of a set of nodes on level l, below the level where the set is one unit: there every unit lies
     * wholly inside the set or wholly outside it, and the outline runs halfway between the set's units and the
     * others (it closes a little outside the set's outermost units). Segments x0, y0, x1, y1 one after another.
     */
    const outline = (l: number, inSet: Uint8Array): number[] => {
      const c = counts[l], mine = new Uint8Array(c);
      for (let i = 0; i < N; i++) if (inSet[i]) mine[unitOfLeaf[l][i]] = 1;
      // the spacing of the units: the grid's pitch on the finest level, the mean spacing above it
      const pitch = l === 0 ? hW[0] / box[0].ext / side : (2 * hW[l]) / Math.sqrt(c);
      const R = (l === 0 ? 0.8 : 1.3) * pitch, step = pitch / 4, pad = R + 2 * step;
      const x0 = cx[l] - hW[l] - pad, y0 = cy[l] - hW[l] - pad;
      const n = Math.ceil((2 * (hW[l] + pad)) / step) + 1;
      // distance from each grid point to the nearest unit of the set and the nearest other unit, within 2R
      const dB = new Float32Array(n * n).fill(Infinity), dN = new Float32Array(n * n).fill(Infinity);
      const rc = Math.ceil((2 * R) / step);
      for (let b = 0; b < c; b++) {
        const [hx, hy] = home(l, b), d = mine[b] ? dB : dN;
        const ci = Math.round((hx - x0) / step), cj = Math.round((hy - y0) / step);
        for (let j = Math.max(0, cj - rc); j <= Math.min(n - 1, cj + rc); j++)
          for (let i = Math.max(0, ci - rc); i <= Math.min(n - 1, ci + rc); i++) {
            const dd = Math.hypot(x0 + i * step - hx, y0 + j * step - hy);
            if (dd < d[j * n + i]) d[j * n + i] = dd;
          }
      }
      // positive nearer the set than any other unit and within R of it
      const h = new Float32Array(n * n);
      for (let k = 0; k < n * n; k++) h[k] = dB[k] === Infinity ? -R : Math.min(dN[k] === Infinity ? Infinity : (dN[k] - dB[k]) / 2, R - dB[k]);
      return zeroLine(h, n, x0, y0, step);
    };

    // outlines that follow the small branch up the levels: around its patch below its own level (the finest
    // level's corner patch would push a circle past the square's edge), then a ring around the one unit that holds it
    const RING_SEG = 40;
    const rings: LineField[] = [];
    const ringAt: { c: [number, number, number]; r: number }[] = [];
    if (showD) {
      for (let l = 0; l < L; l++) {
        let c: [number, number, number], r: number;
        if (l >= lvD) {
          c = home(l, dUnit[l]);
          r = (dotPx[off[l] + dUnit[l]] / 2 + 5) * K;
        } else {
          let sx = 0, sy = 0, n = 0;
          const seen = new Uint8Array(counts[l]);
          for (let i = 0; i < N; i++) if (inD[i] && !seen[unitOfLeaf[l][i]]) { seen[unitOfLeaf[l][i]] = 1; const h = home(l, unitOfLeaf[l][i]); sx += h[0]; sy += h[1]; n++; }
          c = [sx / n, sy / n, 0];
          r = 0;
          for (let b = 0; b < counts[l]; b++) if (seen[b]) { const h = home(l, b); r = Math.max(r, Math.hypot(h[0] - c[0], h[1] - c[1]) + (dotPx[off[l] + b] / 2 + 4) * K); }
        }
        ringAt.push({ c, r });
        const segs: number[] = [];
        if (l < lvD) segs.push(...outline(l, inD));
        else for (let s = 0; s < RING_SEG; s++) {
          const a0 = (s / RING_SEG) * Math.PI * 2, a1 = ((s + 1) / RING_SEG) * Math.PI * 2;
          segs.push(c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0), c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1));
        }
        const lf = new LineField(Math.max(1, segs.length / 4));
        for (let s = 0; s + 3 < segs.length; s += 4) lf.push([segs[s], segs[s + 1], 0], [segs[s + 2], segs[s + 3], 0], C.cyan, 1);
        lf.commit();
        rings.push(lf);
        group.add(lf.object);
      }
    }

    // ochre outlines that follow the branch up the levels. Below its own level every unit lies wholly inside the
    // branch or wholly outside it, and the outline runs halfway between the branch's units and the others (it
    // closes a little outside the branch's outermost units). From its own level on, a ring marks the one unit
    // that holds it, as for the small branch, and a little wider where the small branch's ring shares the unit.
    const bLines: LineField[] = [];
    if (hasB) {
      const firstB = inB.indexOf(1);
      for (let l = 0; l < L; l++) {
        let segs: number[] = [];
        if (l < lvB) {
          segs = outline(l, inB);
        } else {
          const u = unitOfLeaf[l][firstB], c = home(l, u);
          const r = (dotPx[off[l] + u] / 2 + (showD && l > lvB ? 9 : 5)) * K;
          for (let s = 0; s < RING_SEG; s++) {
            const a0 = (s / RING_SEG) * Math.PI * 2, a1 = ((s + 1) / RING_SEG) * Math.PI * 2;
            segs.push(c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0), c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1));
          }
        }
        const lf = new LineField(Math.max(1, segs.length / 4));
        for (let s = 0; s + 3 < segs.length; s += 4) lf.push([segs[s], segs[s + 1], 0], [segs[s + 2], segs[s + 3], 0], C.amber, 1);
        lf.commit();
        bLines.push(lf);
        group.add(lf.object);
      }
    }

    // ── 6. time ──
    const RISE_GAP = (GROW_END - GROW0 - GROW_LEN) / Math.max(1, L - 2); // level l starts its flight at GROW0 + (l − 1) · RISE_GAP
    const flight = (l: number, t: number) => (l === 0 ? 1 : easeInOut(smooth(GROW0 + (l - 1) * RISE_GAP, GROW0 + (l - 1) * RISE_GAP + GROW_LEN, t)));
    const landed = (l: number) => GROW0 + (l - 1) * RISE_GAP + GROW_LEN;
    // the newest level that counts as built at time t: one that has done 60 % of its flight (readouts, notes and the probe use it)
    const levelAt = (t: number) => Math.max(0, Math.min(L - 1, Math.floor((t - GROW0 - GROW_LEN * 0.6) / RISE_GAP) + 1));
    // color comes in level after level, from the finest to the coarsest
    const tint = (l: number, t: number) => easeInOut(smooth(T_VAL + 0.2 + 0.14 * l, T_VAL + 1.3 + 0.14 * l, t));
    const small = (t: number) => (showD ? easeInOut(smooth(T_SMALL + 0.2, T_SMALL + 1.3, t)) : 0);
    const ringOn = (l: number, t: number) => smooth(T_SMALL + 0.5 + 0.16 * l, T_SMALL + 1.0 + 0.16 * l, t);
    // the branch's outlines come in with its label on slide 7.2, from the finest level to the coarsest
    const outlineOn = (l: number, t: number) => smooth(T_VAL + 0.9 + 0.16 * l, T_VAL + 1.4 + 0.16 * l, t);

    // ── 7. one lineage, followed on the first slide: on every level, the group that becomes one unit of the
    // next level, and the links from its members to that unit. A node away from both branches, at the lower
    // right of the finest square (next to the square it moves to). ──
    let n0 = -1;
    {
      let best = Infinity;
      for (let i = 0; i < N; i++) {
        if (hasB && unitOfLeaf[lvB][i] === uB) continue;
        if (showD && inD[i]) continue;
        if (L > 1 && kids[1][unitOfLeaf[1][i]] !== 3) continue;
        if (showD && L > lvD && unitOfLeaf[lvD][i] === uD) continue;
        const { mx, my, ext } = box[0];
        const x = (px[i] - mx) / ext, y = (pz[i] - my) / ext;
        const sc = (x - 0.7) ** 2 + (y + 0.85) ** 2; // low in the square: its links then run along the squares' common base
        if (sc < best) { best = sc; n0 = i; }
      }
      if (n0 < 0) n0 = N - 1;
    }
    const lin = counts.map((_, l) => unitOfLeaf[l][n0]);
    /** Is unit b of level l in the group that becomes the lineage's unit on level l + 1? */
    const inLin = (l: number, b: number) => l + 1 < L && ups[l][b] === lin[l + 1];
    const linLines = new LineField(64, { width: 1.6 });
    group.add(linLines.object);
    // leaders from two labels to what they name: "becomes one unit" (slide 7.1) and the small branch (7.4)
    const leaders = new LineField(4);
    group.add(leaders.object);
    // the lineage's nodes on the finest level print a little larger
    for (let b = 0; b < N; b++) if (inLin(0, b)) pts.sizes[b] *= 1.4;

    // ── 8. the plot: how far each shift stands out above the noise, level by level (log scale) ──
    // Its points stand above the squares they belong to; the finest level's point stands at the left edge of
    // the second square.
    const plotX = counts.map((_, l) => (l === 0 ? (cxPx[1] - hPx[1] - 14 - xPx / 2) * K : cx[l]));
    const PY0 = 74, PY1 = 236; // layout pixels above the center line
    const VMIN = showD ? Math.min(0.3, 0.8 * Math.min(...standD)) : 0.3;
    const VMAX = Math.max(22, 1.3 * Math.max(...standB, ...standD));
    const plotY = (v: number) => (PY0 + ((Math.log(Math.max(VMIN, v)) - Math.log(VMIN)) / (Math.log(VMAX) - Math.log(VMIN))) * (PY1 - PY0)) * K;
    const axes = new LineField(64);
    {
      // the noise line (1×), dashed, and a faint line at 10×
      const xa = plotX[0] - 8 * K, xb = plotX[L - 1] + 8 * K, y1 = plotY(1), y10 = plotY(10);
      const n = 26;
      for (let s = 0; s < n; s += 1) if (s % 2 === 0) axes.push([xa + ((xb - xa) * s) / n, y1, 0], [xa + ((xb - xa) * (s + 1)) / n, y1, 0], INK, 0.55);
      axes.push([xa, y10, 0], [xb, y10, 0], INK, 0.16);
      axes.push([xa, plotY(VMIN), 0], [xa, plotY(VMAX), 0], INK, 0.3);
      axes.commit();
    }
    group.add(axes.object);
    // drop lines: the plot's noise line serves as its axis of levels. From each level's place on it a faint dotted
    // line runs down to that level's square (they start at the noise line, so they cross no label of the plot). The
    // finest level's point stands beside its square, so it gets a short level line to the square's edge.
    // Each level's dashes come in with its point (update() redraws them with the point's strength).
    const dropSeg: number[][] = counts.map(() => []); // per level: x0, y0, x1, y1 of every dash
    {
      const dotted = (l: number, x0: number, y0: number, x1: number, y1: number) => {
        const len = Math.hypot(x1 - x0, y1 - y0), n = Math.max(1, Math.round(len / (6 * K)));
        for (let q = 0; q < n; q += 1) {
          const u0 = q / n, u1 = (q + 0.45) / n;
          dropSeg[l].push(x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0, x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1);
        }
      };
      for (let l = 0; l < L; l++) {
        if (l === 0) dotted(0, plotX[0] - 5 * K, plotY(standB[0]), cx[0] + hW[0] + 3 * K, plotY(standB[0]));
        else dotted(l, plotX[l], plotY(1) + 4 * K, plotX[l], counts[l] === 1 ? cy[l] + (dotPx[off[l]] / 2 + 12) * K : cy[l] + hW[l] + 4 * K);
      }
    }
    const drops = new LineField(Math.max(1, dropSeg.reduce((n, d) => n + d.length / 4, 0)));
    group.add(drops.object);
    const curveB = new LineField(L, { width: 2 }), curveD = new LineField(L, { width: 2 });
    group.add(curveB.object, curveD.object);
    const marks = new PointField({ count: 2 * L, size: 1 });
    for (let j = 0; j < 2 * L; j++) marks.sizes[j] = (6.5 * K) / 0.52;
    for (let l = 0; l < L; l++) {
      marks.setPos(l, plotX[l], plotY(standB[l]), 0);
      marks.setPos(L + l, plotX[l], plotY(standD[l]), 0);
    }
    marks.commit(true, true, true);
    group.add(marks.object);
    /** How much of a curve is drawn at time t: up to the branch's own level on slide 7.2, the rest on 7.3. */
    const curveBTo = (t: number) => {
      const a = smooth(T_VAL + 1.4, T_VAL + 1.4 + 0.25 * lvB, t) * lvB;
      return t < T_ABOVE + 0.3 ? a : a + smooth(T_ABOVE + 0.3, T_ABOVE + 1.3, t) * (L - 1 - lvB);
    };
    const curveDTo = (t: number) => showD ? smooth(T_SMALL + 0.6, T_SMALL + 1.8, t) * (L - 1) : 0;
    const drawCurve = (lf: LineField, v: number[], upto: number, col: RGB) => {
      lf.clear();
      for (let l = 0; l + 1 < L && l < upto; l++) {
        const f = Math.min(1, upto - l), a: [number, number, number] = [plotX[l], plotY(v[l]), 0];
        const b: [number, number, number] = [plotX[l] + (plotX[l + 1] - plotX[l]) * f, plotY(v[l]) + (plotY(v[l + 1]) - plotY(v[l])) * f, 0];
        lf.push(a, b, col, 0.95);
      }
      lf.commit();
    };

    // how many units of the level below the smallest unit holds, away from the root and at the root (checked, not assumed)
    let minGroup = Infinity, rootMin = Infinity;
    for (let l = 1; l < L; l++) {
      rootMin = Math.min(rootMin, kids[l][0]);
      for (let b = 1; b < counts[l]; b++) minGroup = Math.min(minGroup, kids[l][b]);
    }
    const rootToo = rootMin >= BLOCK; // in the default run every unit, the root's included, holds at least BLOCK
    const numWord = (n: number) => (['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][n] ?? fmt(n));
    const units = (n: number) => (n === 1 ? 'one unit' : `${fmt(n)} units`);

    // label anchors: under each square, over it, and beside the branches
    const under = (l: number): [number, number, number] => [cx[l], BASE * K, 0];
    const topOf = (l: number): [number, number, number] => [cx[l], cy[l] + hW[l] + (counts[l] === 1 ? 14 : 0) * K, 0];
    // the branch's label stands beside the finest square, just above the second square
    let bAnchor = under(0);
    if (hasB) bAnchor = [cx[0] + hW[0] + 6 * K, cy[1] + hW[1] + 30 * K, 0];
    // the small branch's label stands over the square of its own level, above its ring there
    let dAnchor = under(0);
    if (showD) {
      const { c, r } = ringAt[lvD];
      dAnchor = [cx[lvD] - hW[lvD] - 9 * K, Math.max(cy[lvD] + hW[lvD], c[1] + r) + 24 * K, 0];
    }

    const chapter: Chapter = {
      id: 'scales',
      pillar: 'frame',
      kicker: 'Across the teams',
      title: 'COARSE-GRAINING',
      source: 'Greenspan et al. · Towards Worst-Case Guarantees with Scale-Aware Interpretability · arXiv:2602.05184',
      links: [{ text: 'arXiv:2602.05184', url: PAPER }],
      duration: DUR,
      group,
      beats: [
        {
          at: T_OPEN,
          caption:
            'In statistical physics, <em>coarse-graining</em> groups small parts into larger units. Greenspan and co-authors ask which details of a network we could group together and still predict its behavior. ' +
            'Here each group reports the mean of its nodes. At the coarsest scale, one number describes the whole tree.',
        },
        {
          at: T_VAL,
          caption:
            'Give every node independent noise and add the same positive shift to one branch. Within that branch, averaging preserves the shift and reduces the noise. ' +
            (riseB ? 'The shift stands out most when a single unit covers the entire branch.' : 'The shift stands out more clearly when a single unit covers the branch.'),
        },
        {
          at: T_ABOVE,
          caption:
            'Once a unit takes in nodes outside the branch, its mean shift falls in proportion to the fraction of nodes that carry it. ' +
            'The noise also falls, but more slowly. A shift that grew clearer with each merge inside the branch now becomes harder to detect.',
        },
        {
          at: T_SMALL,
          caption:
            (showD
              ? `Removing the small branch’s shift changes its own mean by ${val(SHIFT)}, but the whole-tree mean by only ${val(SHIFT * nD / N)}. `
              : `Removing the highlighted branch’s shift changes its own mean by ${val(SHIFT)}, but the whole-tree mean by only ${val(bTop)}. `) +
            'Which detail can be discarded depends on the quantity an explanation must preserve. Greenspan and co-authors propose bounding that change for a specified network behavior.',
        },
      ],
      // one fixed view, straight on: the figure does not move while its labels are read
      shots: [
        { t: 0, pos: [CAM_X, CAM_Y, CAM_Z], look: [CAM_X, CAM_Y, 0], fov: FOV },
        { t: DUR, pos: [CAM_X, CAM_Y, CAM_Z], look: [CAM_X, CAM_Y, 0], fov: FOV },
      ],
      update(t, api) {
        const Pp = pts.positions, Cc = pts.colors;
        const kd = small(t);
        const linOn = smooth(T_OPEN + 0.5, T_OPEN + 1.2, t) * (1 - smooth(T_VAL, T_VAL + 1.0, t));
        for (let l = 0; l < L; l++) {
          const el = flight(l, t);
          const on = l === 0 ? smooth(T_OPEN, T_OPEN + 0.8, t) : smooth(GROW0 + (l - 1) * RISE_GAP, GROW0 + (l - 1) * RISE_GAP + 0.4 * GROW_LEN, t);
          const w = tint(l, t);
          const kInk = (l === 0 ? 0.55 : 0.75) * (1 - w);
          for (let b = 0; b < counts[l]; b++) {
            const j = off[l] + b, j3 = j * 3;
            Pp[j3] = start[j3] + (target[j3] - start[j3]) * el;
            Pp[j3 + 1] = start[j3 + 1] + (target[j3 + 1] - start[j3 + 1]) * el;
            Pp[j3 + 2] = 0;
            // the lineage prints in full ink on the first slide; a group darkens once the unit it becomes starts to form
            const lineage = (inLin(l, b) && (l === 0 || t >= GROW0 + l * RISE_GAP)) || lin[l] === b;
            const kk = kInk * (lineage ? 1 + 0.9 * linOn : 1 - 0.25 * linOn);
            let r = INK[0] * kk, gg = INK[1] * kk, bb = INK[2] * kk;
            if (w > 0) {
              const d: RGB = diverge((nMean[j] + bPart[j] + dPart[j] * kd) / VSCALE);
              r += (d[0] * CK + GRAY[0]) * w; gg += (d[1] * CK + GRAY[1]) * w; bb += (d[2] * CK + GRAY[2]) * w;
            }
            Cc[j3] = r * on; Cc[j3 + 1] = gg * on; Cc[j3 + 2] = bb * on;
          }
        }
        pts.commit(true, true, false);
        rings.forEach((lf, l) => (lf.uIntensity.value = ringOn(l, t)));
        bLines.forEach((lf, l) => (lf.uIntensity.value = outlineOn(l, t) * (1 - 0.5 * kd)));

        // the lineage's links: every member of a group to the unit it becomes, drawn as that unit forms
        linLines.clear();
        if (linOn > 0.01)
          for (let l = 0; l + 1 < L; l++) {
            const vis = smooth(GROW0 + l * RISE_GAP, GROW0 + l * RISE_GAP + 0.4, t);
            if (vis < 0.01) continue;
            const q3 = (off[l + 1] + lin[l + 1]) * 3, z: [number, number, number] = [Pp[q3], Pp[q3 + 1], 0];
            for (let b = 0; b < counts[l]; b++) if (inLin(l, b)) { const j3 = (off[l] + b) * 3; linLines.push([Pp[j3], Pp[j3 + 1], 0], z, INK, 0.7 * vis * linOn); }
          }
        linLines.commit();
        const aG1 = L > 1 ? smooth(landed(1) - 0.2, landed(1) + 0.4, t) * (1 - smooth(T_VAL - 0.3, T_VAL + 0.5, t)) : 0;
        leaders.clear();
        if (L > 1) {
          const u = home(1, lin[1]), r = (dotPx[off[1] + lin[1]] / 2 + 3) * K;
          if (aG1 > 0.01) leaders.push([u[0], u[1] + r, 0], [u[0], cy[1] + hW[1] + 12 * K, 0], INK, 0.8 * aG1);
        }
        if (showD) {
          const aDl = smooth(T_SMALL + 0.8, T_SMALL + 1.6, t), { c, r } = ringAt[lvD];
          if (aDl > 0.01) leaders.push([dAnchor[0] + 4 * K, dAnchor[1] - 22 * K, 0], [c[0] - r * 0.6, c[1] + r * 0.8, 0], C.cyan, aDl);
        }
        leaders.commit();

        // the plot
        const aP = smooth(T_VAL + 1.0, T_VAL + 1.8, t);
        axes.uIntensity.value = aP;
        drawCurve(curveB, standB, curveBTo(t), C.amber);
        drawCurve(curveD, standD, curveDTo(t), C.cyan);
        const ub = curveBTo(t), ud = curveDTo(t), dOn = showD ? smooth(T_SMALL + 0.4, T_SMALL + 0.8, t) : 0;
        for (let l = 0; l < L; l++) {
          marks.setColor(l, C.amber, 0.95 * smooth(l - 0.3, l, ub) * aP);
          marks.setColor(L + l, C.cyan, 0.95 * smooth(l - 0.3, l, ud) * dOn);
        }
        marks.commit(false, true, false);
        drops.clear();
        for (let l = 0; l < L; l++) {
          const k = 0.32 * smooth(l - 0.3, l, ub) * aP, d = dropSeg[l];
          if (k > 0.002) for (let q = 0; q + 3 < d.length; q += 4) drops.push([d[q], d[q + 1], 0], [d[q + 2], d[q + 3], 0], INK, k);
        }
        drops.commit();

        // labels in the words of the plan; each comes up with the part it names
        const lab = { size: 'md' as const, color: CSS.white };
        const a0 = smooth(T_OPEN, T_OPEN + 0.8, t);
        api.label('l0', under(0), `${fmt(counts[0])} tree nodes`, { ...lab, align: 'below', dy: 10, alpha: a0 });
        api.label('title', topOf(0), 'Averaging nodes on a tree · illustration', { size: 'md', color: CSS.white, align: 'center', dy: -12, alpha: a0, help: `<p>Each unit groups connected nodes or units from the level before. A group becomes one unit once it contains at least ${numWord(BLOCK)}${rootToo ? '.' : ', apart from any smaller group left at the root.'}</p><p>The nodes are arranged so they all fit in a square. Their positions on screen can differ from their connections in the random tree.</p>` });
        for (let l = 1; l < L; l++) {
          const al = smooth(landed(l) - 0.4, landed(l) + 0.4, t);
          if (l < L - 1) api.label(`c${l}`, under(l), units(counts[l]), { ...lab, align: 'below', dy: 10, alpha: al });
          else api.label('lT', under(l), `${units(counts[l])}`, { ...lab, align: 'below', dx: 24, dy: 10, alpha: al });
        }
        // slide 7.1: the rule, and one group followed into the unit it becomes
        if (L > 1) {
          const g0 = home(0, lin[0]);
          api.label('g0', g0, `a connected group<br>of ${numWord(kids[1][lin[1]])} nodes`, { ...lab, align: 'right', dx: -12, alpha: smooth(T_OPEN + 0.6, T_OPEN + 1.3, t) * (1 - smooth(T_VAL - 0.3, T_VAL + 0.5, t)), plate: true });
          // over the second square, clear of its units, with a leader down to the unit it names
          api.label('g1', [home(1, lin[1])[0], cy[1] + hW[1] + 14 * K, 0], 'becomes one unit', { ...lab, align: 'center', dy: -2, alpha: aG1, plate: true });
        }
        // slide 7.2: the color key, the two parts of every node's value, and the plot
        const aV = smooth(T_VAL + 0.8, T_VAL + 1.6, t);
        if (aV > 0.01) {
          api.label('key', [cx[L - 1] + 26 * K, 12 * K, 0], `Unit mean<br><span style="color:${CSS.cyan}">teal −</span>&nbsp;&nbsp;gray ≈ 0&nbsp;&nbsp;<span style="color:${CSS.amber}">ochre +</span>`, { size: 'sm', color: CSS.white, align: 'right', alpha: aV, plate: true });
          if (hasB) api.label('shift', bAnchor, `Ochre branch · ${signed(SHIFT)} per node`, { ...lab, color: CSS.amber, align: 'left', alpha: aV, plate: true });
          const wholeTreeHelp = t >= T_ABOVE ? ` Its contribution to the whole-tree mean is ${val(SHIFT)} × ${fmt(nB)} ÷ ${fmt(N)} ≈ ${val(bTop)}.` : '';
          const smallBranchHelp = showD && t >= T_SMALL ? ` The teal branch shifts ${fmt(nD)} nodes by ${signed(-SHIFT)} each.` : '';
          api.label('pt', [plotX[0], plotY(VMAX), 0], 'Signal-to-noise (log scale)', { size: 'md', color: CSS.white, align: 'left', dx: 8, dy: -8, alpha: aP, help:
            '<p>The ratio divides the magnitude of a unit’s mean shift by the noise standard deviation. Below the branch’s own level, the curve summarizes the ratios of its units as a root mean square. The dashed line marks a ratio of one.</p>' +
            `<p>The ochre branch shifts ${fmt(nB)} nodes by ${signed(SHIFT)} each.${wholeTreeHelp}${smallBranchHelp}</p>` });
          api.label('p1', [plotX[L - 1] + 8 * K, plotY(1), 0], '1×', { size: 'sm', color: CSS.white, align: 'left', dx: 6, alpha: aP });
          api.label('p10', [plotX[L - 1] + 8 * K, plotY(10), 0], '10×', { size: 'sm', color: CSS.slate, align: 'left', dx: 6, alpha: aP });
          if (hasB) api.label('own', [plotX[lvB], plotY(standB[lvB]), 0], `Peak · ${ratio(standB[lvB])}×`, { ...lab, color: CSS.amber, align: 'center', dy: -8, alpha: smooth(T_VAL + 1.4 + 0.25 * lvB, T_VAL + 1.8 + 0.25 * lvB, t) });
        }
        // slide 7.3: what the top unit keeps of the branch
        const aA = smooth(T_ABOVE + 0.8, T_ABOVE + 1.5, t) * (1 - smooth(T_SMALL - 0.2, T_SMALL + 0.5, t));
        // left of the top unit's point, under the curve: the label keeps the figure's right edge where slide 7.2 has it
        if (hasB && aA > 0.01) api.label('dil', [plotX[L - 1], plotY(standB[L - 1]), 0], `Whole tree · ${ratio(standB[L - 1])}×`, { ...lab, color: CSS.amber, align: 'right', dx: -10, dy: 19, alpha: aA });
        // slide 7.4: the small branch, followed up the levels by its rings and its curve
        const aD = smooth(T_SMALL + 0.8, T_SMALL + 1.6, t);
        if (showD && aD > 0.01) {
          api.label('small', dAnchor, `Teal branch · ${signed(-SHIFT)} per node`, { ...lab, color: CSS.cyan, align: 'left', alpha: aD });
          api.label('dTop', [plotX[L - 1], plotY(standD[L - 1]), 0], `${ratio(standD[L - 1])}×`, { ...lab, color: CSS.cyan, align: 'below', dy: 8, alpha: aD });
        }
      },
      results: [
        {
          label: 'how far the branch\'s shift stands out above the noise: finest level → its own level → top',
          measured: `${ratio(standB[0])} → ${ratio(standB[lvB])} → ${ratio(standB[L - 1])}`,
          reference: 'illustration',
          claim: 'the signal-to-noise ratio peaks at the branch’s own level and falls whenever the containing unit adds unshifted nodes',
          kind: 'illustration',
          group: 'picture',
          say: 'On this tree, adding unshifted nodes to the branch average lowers its signal-to-noise ratio.',
        },
      ],
      details(t) {
        if (t < T_SMALL) return undefined;
        return {
          title: 'What a behavioral guarantee would need',
          body:
            '<p>Renormalization tracks how quantities change as we group and rescale them. Greenspan and co-authors describe a feature as <em>dangerously irrelevant</em> when it shrinks under coarse-graining yet its fluctuations still have a substantial effect on a large-scale observable.</p>' +
            '<p>In this tree we can calculate exactly how the shift affects either mean. For a network, the authors propose holding the coarse description fixed and bounding how far a chosen output can move under a specified set of changes to the omitted details.</p>',
        };
      },

      // ── the notes layer ──
      inspect(t) {
        const hs: Hotspot[] = [];
        if (t < T_VAL)
          hs.push({
            id: 'tree',
            pos: [cx[0] - hW[0], cy[0], 0],
            dx: -12,
            title: 'Neighbors in the tree',
            body: '<p>Nearby dots on the square need not be connected in the tree. Grouping follows the tree connections; the square arranges the dots so they can all be seen.</p>',
          });
        if (t >= T_VAL + 1.8 && t < T_ABOVE && hasB)
          hs.push({
            id: 'values',
            // at the plot's upper right, clear of the branch's outline on the finest square
            pos: [plotX[L - 1], plotY(VMAX), 0],
            dx: 46,
            title: 'The noise scale',
            body:
              `<p>The random values on different nodes are independent, each with standard deviation 1. Their mean over $n$ nodes has standard deviation $1/\\sqrt{n}$. The curve divides the magnitude of the branch's contribution by this expected noise scale, rather than by the random noise value drawn for a particular unit.</p>` +
              `<p>Below the level where the branch forms one unit, the curve shows the root mean square of this ratio across its units. From that level onward, it follows the single unit containing the branch. The curve is derived from the averaging rule; changing the noise draw alone would leave it unchanged.</p>`,
          });
        if (t >= T_ABOVE + 0.8 && t < T_SMALL && hasB)
          hs.push({
            id: 'top',
            // right of the coarsest unit, clear of its rings
            pos: [cx[L - 1], cy[L - 1], 0],
            dx: 34,
            dy: 10,
            title: 'Two different dilution factors',
            body:
              `<p>When every node in a branch of $n$ nodes receives a shift $\\delta$, its contribution to the mean over all $N$ nodes is $\\delta n/N$. Its ratio to the noise is $|\\delta|n/\\sqrt{N}$, because the noise scale there is $1/\\sqrt{N}$. Relative to the branch's own unit, the mean contribution falls by $n/N$ and the signal-to-noise ratio by $\\sqrt{n/N}$.</p>` +
              `<p>Setting this branch's shift to zero lowers the whole-tree mean by about ${val(bTop)}. The nodes stay in the average. Deleting them would change the denominator and every other node's contribution, so this calculation would no longer apply. The exact effect here concerns one averaged number in a tree, with no neural network.</p>`,
          });
        if (showD && t >= T_SMALL + 0.8)
          hs.push({
            id: 'small',
            pos: dAnchor,
            dx: -14,
            dy: -10,
            title: 'Why this branch falls below one noise standard deviation',
            body:
              `<p>The small branch was chosen to make its shift at least three noise standard deviations locally and less than one for the whole tree. This example gives ${ratio(standD[lvD])} locally and ${ratio(standD[L - 1])} at the top. Those are signal-to-noise ratios for prescribed shifts on this tree. They do not estimate how often a learned feature has this property.</p>`,
          });
        return hs;
      },
      probe(t, p) {
        // the visible unit nearest to the pointer, on any level that has come in
        const P3 = pts.positions, lv = levelAt(t);
        let best = -1, bl = 0, bd = 12 * 12; // within 12 px (7 px for the dense tree)
        for (let l = 0; l <= lv; l++) {
          const tol = l === 0 ? 7 * 7 : 12 * 12;
          for (let b = 0; b < counts[l]; b++) {
            const j3 = (off[l] + b) * 3, s = p.project(P3[j3], P3[j3 + 1], P3[j3 + 2]);
            if (s[2] < -1 || s[2] > 1) continue;
            const d = (s[0] - p.x) * (s[0] - p.x) + (s[1] - p.y) * (s[1] - p.y);
            if (d < bd && d < tol) { bd = d; best = b; bl = l; }
          }
        }
        if (best < 0) return null;
        const b = best, l = bl, s = size[l][b], j3 = (off[l] + b) * 3;
        const rows = [{ label: 'nodes of the tree inside', value: fmt(s) }];
        if (l === 0) rows.push({ label: 'depth from the root', value: String(depth[b]) });
        else rows.push({ label: `units of level ${l - 1} it holds`, value: fmt(kids[l][b]) });
        if (l + 1 < L) rows.push({ label: `its unit on level ${l + 1} holds`, value: `${fmt(size[l + 1][ups[l][b]])} nodes` });
        if (t >= T_VAL + 1) rows.push({ label: 'its mean value', value: val(nMean[off[l] + b] + bPart[off[l] + b] + small(t) * dPart[off[l] + b]) });
        const out: Probe = {
          pos: [P3[j3], P3[j3 + 1], P3[j3 + 2]],
          title: l === 0 ? 'A node of the tree' : `A unit on level ${l} of ${L - 1}`,
          rows,
          note: l === 0 ? undefined : 'A dot grows with the number of nodes inside.',
        };
        return out;
      },
    };
    return chapter;
  },
};

export default scales;
