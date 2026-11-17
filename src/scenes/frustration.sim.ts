/**
 * Numerics for the `frustration` chapter: a miniature of the sparse-denoising model from
 * Vaintrob, "A tale of three theories: sparsity, frustration, and statistical field theory".
 *
 *   x = x* + ξ,  x* ∈ {0,1}^d two-hot,  ξ ~ N(0, σ² I),  target x*
 *   x̂ = D φ(E x),  E ∈ R^{h×d},  D ∈ R^{d×h},  φ = tanh³,  no biases
 *   L = E[ λ_on Σ_{on}(x̂_j − 1)² + Σ_{off} x̂_j² ] + κ_E‖E‖² + κ_D‖D‖²
 *
 * Adam, 5 % linear warm-up then cosine decay to 0, fresh samples every step. Everything is
 * seeded and deterministic. No DOM, no three.js: this file is plain arithmetic.
 *
 * The chapter uses a leak map (leakMap) to show where a single active feature writes to the output.
 */
import { mulberry32 } from '../lib/math';

export interface NetCfg {
  d: number; // features (input = output dimension)
  h: number; // hidden neurons
  sigma2: number; // input noise variance
  lamOn: number; // weight of the active coordinates in the loss
  kE: number; // L2 coefficient on E
  kD: number; // L2 coefficient on D
  sdE: number; // init standard deviation of E
  sdD: number; // init standard deviation of D
  seed: number;
}

/** Standard normal sampler (Marsaglia polar method) on a seeded uniform RNG. */
export function polarGaussian(rnd: () => number) {
  let spare = 0, has = false;
  return () => {
    if (has) {
      has = false;
      return spare;
    }
    let u = 0, v = 0, s = 0;
    do {
      u = rnd() * 2 - 1;
      v = rnd() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    has = true;
    return u * m;
  };
}

/** The two-layer denoiser with its Adam state. E is row-major h×d, D is row-major d×h. */
export class Net {
  readonly d: number;
  readonly h: number;
  readonly n: number;
  readonly E: Float64Array;
  readonly D: Float64Array;
  private gE: Float64Array;
  private gD: Float64Array;
  private mE!: Float64Array;
  private vE!: Float64Array;
  private mD!: Float64Array;
  private vD!: Float64Array;
  private b1t = 1;
  private b2t = 1;
  private x: Float64Array;
  private a: Float64Array;
  private th: Float64Array;
  private dz: Float64Array;
  private rnd: () => number;
  private g: () => number;
  private sig: number;

  constructor(readonly cfg: NetCfg) {
    const d = (this.d = cfg.d), h = (this.h = cfg.h), n = (this.n = d * h);
    if (d % 4 || h % 4) throw new Error('d and h must be multiples of 4');
    this.rnd = mulberry32(cfg.seed);
    this.g = polarGaussian(this.rnd);
    this.sig = Math.sqrt(cfg.sigma2);
    this.E = new Float64Array(n);
    this.D = new Float64Array(n);
    for (let k = 0; k < n; k++) this.E[k] = this.g() * cfg.sdE;
    for (let k = 0; k < n; k++) this.D[k] = this.g() * cfg.sdD;
    this.gE = new Float64Array(n);
    this.gD = new Float64Array(n);
    this.x = new Float64Array(d);
    this.a = new Float64Array(h);
    this.th = new Float64Array(h);
    this.dz = new Float64Array(h);
    this.resetAdam();
  }

  resetAdam() {
    const n = this.n;
    this.mE = new Float64Array(n);
    this.vE = new Float64Array(n);
    this.mD = new Float64Array(n);
    this.vD = new Float64Array(n);
    this.b1t = 1;
    this.b2t = 1;
  }

  /** One Adam step on a fresh minibatch. Returns the minibatch loss (without the L2 terms). */
  step(eta: number, B = 32) {
    const { d, h, E, D, gE, gD, x, a, th, dz, rnd, g, sig } = this;
    const { lamOn, kE, kD } = this.cfg;
    gE.fill(0);
    gD.fill(0);
    let loss = 0;
    const s2 = 2 / B;
    for (let b = 0; b < B; b++) {
      // sample: two distinct active features plus gaussian noise on every coordinate
      const j1 = (rnd() * d) | 0;
      let j2 = (rnd() * (d - 1)) | 0;
      if (j2 >= j1) j2++;
      for (let j = 0; j < d; j++) x[j] = g() * sig;
      x[j1] += 1;
      x[j2] += 1;
      // hidden layer: a = tanh³(E x)
      for (let i = 0; i < h; i++) {
        let z0 = 0, z1 = 0, z2 = 0, z3 = 0;
        const o = i * d;
        for (let j = 0; j < d; j += 4) {
          z0 += E[o + j] * x[j];
          z1 += E[o + j + 1] * x[j + 1];
          z2 += E[o + j + 2] * x[j + 2];
          z3 += E[o + j + 3] * x[j + 3];
        }
        const t = Math.tanh(z0 + z1 + z2 + z3);
        th[i] = t;
        a[i] = t * t * t;
      }
      // output, re-balanced squared error, gradient w.r.t. D and back-propagated signal
      dz.fill(0);
      for (let j = 0; j < d; j++) {
        let y0 = 0, y1 = 0, y2 = 0, y3 = 0;
        const o = j * h;
        for (let i = 0; i < h; i += 4) {
          y0 += D[o + i] * a[i];
          y1 += D[o + i + 1] * a[i + 1];
          y2 += D[o + i + 2] * a[i + 2];
          y3 += D[o + i + 3] * a[i + 3];
        }
        const y = y0 + y1 + y2 + y3;
        const on = j === j1 || j === j2;
        const r = on ? y - 1 : y, w = on ? lamOn : 1;
        loss += w * r * r;
        const gj = s2 * w * r;
        for (let i = 0; i < h; i++) {
          gD[o + i] += gj * a[i];
          dz[i] += D[o + i] * gj;
        }
      }
      // gradient w.r.t. E through φ'(z) = 3 tanh² (1 − tanh²)
      for (let i = 0; i < h; i++) {
        const t2 = th[i] * th[i];
        const q = dz[i] * 3 * t2 * (1 - t2), o = i * d;
        for (let j = 0; j < d; j++) gE[o + j] += q * x[j];
      }
    }
    // Adam (β1 = 0.9, β2 = 0.999) with the L2 terms added to the gradient
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.b1t *= b1;
    this.b2t *= b2;
    const c1 = 1 / (1 - this.b1t), c2 = 1 / (1 - this.b2t);
    const { mE, vE, mD, vD } = this;
    for (let i = 0; i < h; i++)
      for (let j = 0; j < d; j++) {
        const ke = i * d + j, kd = j * h + i;
        let gr = gE[ke] + 2 * kE * E[ke];
        mE[ke] = b1 * mE[ke] + (1 - b1) * gr;
        vE[ke] = b2 * vE[ke] + (1 - b2) * gr * gr;
        E[ke] -= (eta * mE[ke] * c1) / (Math.sqrt(vE[ke] * c2) + eps);
        gr = gD[kd] + 2 * kD * D[kd];
        mD[kd] = b1 * mD[kd] + (1 - b1) * gr;
        vD[kd] = b2 * vD[kd] + (1 - b2) * gr * gr;
        D[kd] -= (eta * mD[kd] * c1) / (Math.sqrt(vD[kd] * c2) + eps);
      }
    return loss / B;
  }
}

/** How well a trained network does its task, measured on fresh samples drawn the way training draws them. */
export interface TaskScore {
  samples: number;
  noise: number; // sd of the input noise on every coordinate (σ)
  meanOn: number; // mean output at the two active features (target 1)
  rmsOnErr: number; // root-mean-square error at the active features
  rmsOff: number; // root-mean-square output at the inactive features (target 0; the input there has rms σ)
  rmse: number; // root-mean-square error of the output against the clean list, over all coordinates
  rmsIn: number; // the same for the noisy input itself (about σ)
  top2: number; // fraction of samples whose two largest outputs are exactly the two active features
  top2In: number; // the same for the noisy input itself
  rmsZero: number; // the error of an output of all zeros (√(2/d))
  scale: number; // minimizes expected squared error of c·x for this data distribution
  rmsScale: number; // the error of c·x
}

/** One input pushed through the network: the noisy input, the hidden activations and the output. */
export interface Pass {
  on: [number, number];
  x: Float64Array;
  a: Float64Array;
  y: Float64Array;
}

/** x̂ = D tanh³(E x) for one input `x`, written into `a` (hidden) and `y` (output). */
export function forward(net: { d: number; h: number; E: ArrayLike<number>; D: ArrayLike<number> }, x: ArrayLike<number>, a: Float64Array, y: Float64Array) {
  const { d, h, E, D } = net;
  for (let i = 0; i < h; i++) {
    let z = 0;
    const o = i * d;
    for (let j = 0; j < d; j++) z += E[o + j] * x[j];
    const t = Math.tanh(z);
    a[i] = t * t * t;
  }
  for (let j = 0; j < d; j++) {
    let v = 0;
    const o = j * h;
    for (let i = 0; i < h; i++) v += D[o + i] * a[i];
    y[j] = Number.isFinite(v) ? v : 0;
  }
}

/**
 * Score `net` on `samples` fresh two-hot inputs from its own task, with an RNG of its own (`seed`), so the
 * score never touches the training stream. Pure arithmetic on the current weights; nothing is updated.
 * Also returns the first test sample for the figure, regardless of its prediction.
 */
export function scoreTask(net: Pick<Net, 'cfg' | 'E' | 'D' | 'd' | 'h'>, samples: number, seed: number): TaskScore & { shown: Pass } {
  const { d, h } = net;
  const rnd = mulberry32(seed), g = polarGaussian(rnd), sig = Math.sqrt(net.cfg.sigma2);
  const x = new Float64Array(d), a = new Float64Array(h), y = new Float64Array(d);
  // Population-optimal origin-constrained scalar for a two-hot signal plus independent Gaussian noise.
  const prevalence = 2 / d, scale = prevalence / (prevalence + net.cfg.sigma2);
  let sOn = 0, sOnErr = 0, sOff = 0, sIn = 0, top = 0, topIn = 0, scaledError = 0;
  if (!Number.isInteger(samples) || samples < 1) throw new Error('At least one test sample is required');
  let shown!: Pass;
  for (let b = 0; b < samples; b++) {
    const j1 = (rnd() * d) | 0;
    let j2 = (rnd() * (d - 1)) | 0;
    if (j2 >= j1) j2++;
    for (let j = 0; j < d; j++) x[j] = g() * sig;
    x[j1] += 1;
    x[j2] += 1;
    forward(net, x, a, y);
    let m1 = -1, m2 = -1, n1 = -1, n2 = -1;
    for (let j = 0; j < d; j++) {
      if (n1 < 0 || x[j] > x[n1]) { n2 = n1; n1 = j; } else if (n2 < 0 || x[j] > x[n2]) n2 = j;
      const on = j === j1 || j === j2, tg = on ? 1 : 0;
      sIn += (x[j] - tg) * (x[j] - tg);
      scaledError += (scale * x[j] - tg) ** 2;
      if (on) {
        sOn += y[j];
        sOnErr += (y[j] - 1) * (y[j] - 1);
      } else sOff += y[j] * y[j];
      if (m1 < 0 || y[j] > y[m1]) { m2 = m1; m1 = j; } else if (m2 < 0 || y[j] > y[m2]) m2 = j;
    }
    const hit = (m1 === j1 && m2 === j2) || (m1 === j2 && m2 === j1);
    if (hit) top++;
    if ((n1 === j1 && n2 === j2) || (n1 === j2 && n2 === j1)) topIn++;
    if (b === 0) shown = { on: [Math.min(j1, j2), Math.max(j1, j2)], x: x.slice(), a: a.slice(), y: y.slice() };
  }
  const n = samples;
  return {
    samples,
    noise: sig,
    meanOn: sOn / (2 * n),
    rmsOnErr: Math.sqrt(sOnErr / (2 * n)),
    rmsOff: Math.sqrt(sOff / ((d - 2) * n)),
    rmse: Math.sqrt((sOnErr + sOff) / (d * n)),
    rmsIn: Math.sqrt(sIn / (d * n)),
    top2: top / n,
    top2In: topIn / n,
    rmsZero: Math.sqrt(2 / d),
    scale,
    rmsScale: Math.sqrt(scaledError / (d * n)),
    shown,
  };
}

/** 5 % linear warm-up, then cosine decay to exactly 0 at the last step. */
export function lrAt(s: number, total: number, lr: number) {
  const warm = Math.floor(0.05 * total);
  return lr * (s < warm ? (s + 1) / warm : 0.5 * (1 + Math.cos((Math.PI * (s - warm)) / (total - warm))));
}

/**
 * A recorded training run. Weights are stored per PAIR p = i·d + j (neuron i, feature j):
 * E[k·n + p] = E_ij and D[k·n + p] = D_ji at snapshot k.
 */
export interface Track {
  d: number;
  h: number;
  n: number;
  K: number;
  total: number;
  step: Float32Array; // optimizer step of each snapshot
  loss: Float32Array; // minibatch loss averaged over ±8 steps around the snapshot
  lr: Float32Array;
  E: Float32Array;
  D: Float32Array;
}

/** K snapshot steps from 0 to total, packed towards the start where the weights move most. */
export function snapSteps(total: number, K: number, power: number) {
  const out: number[] = [];
  for (let k = 0; k < K; k++) {
    let s = Math.round(total * Math.pow(k / (K - 1), power));
    if (k > 0 && s <= out[k - 1]) s = out[k - 1] + 1;
    out.push(Math.min(total, s));
  }
  out[K - 1] = total;
  return out;
}

/** Train `net` for `total` steps and record snapshots. `tick` is awaited every `chunk` steps. */
export async function record(net: Net, total: number, lr: number, snaps: number[], tick: (done: number) => Promise<void>, chunk = 600): Promise<Track> {
  const { d, h, n } = net, K = snaps.length;
  const tr: Track = {
    d, h, n, K, total,
    step: new Float32Array(K), loss: new Float32Array(K), lr: new Float32Array(K),
    E: new Float32Array(K * n), D: new Float32Array(K * n),
  };
  const trace = new Float32Array(total);
  const grab = (k: number, s: number) => {
    tr.step[k] = s;
    tr.lr[k] = s >= total ? 0 : lrAt(s, total, lr);
    const o = k * n;
    for (let i = 0; i < h; i++)
      for (let j = 0; j < d; j++) {
        tr.E[o + i * d + j] = net.E[i * d + j];
        tr.D[o + i * d + j] = net.D[j * h + i];
      }
  };
  let k = 0;
  for (let s = 0; s < total; s++) {
    if (k < K && snaps[k] === s) grab(k++, s);
    trace[s] = net.step(lrAt(s, total, lr));
    if ((s + 1) % chunk === 0 && s + 1 < total) await tick(s + 1);
  }
  while (k < K) grab(k++, total);
  for (let q = 0; q < K; q++) {
    const s = tr.step[q], a = Math.max(0, s - 8), b = Math.min(total, s + 8);
    let sum = 0;
    for (let t = a; t < b; t++) sum += trace[t];
    tr.loss[q] = sum / Math.max(1, b - a);
  }
  return tr;
}

/**
 * The leak map of a network: L[j·d + k] = Σ_i D_ki tanh³(E_ij), the output at feature k when feature j alone is
 * on (value 1, no noise). Its diagonal is each feature's own output; everything off the diagonal is a leak.
 * Reordering the neurons, or turning one neuron's weights over (E_i· → −E_i·, D_·i → −D_·i; tanh³ is odd), leaves
 * every entry unchanged, so two separately trained networks can be compared entry by entry.
 * `linear` drops the tanh³ (the plain product D E), a version that ignores the activation.
 */
export function leakMap(d: number, h: number, E: ArrayLike<number>, D: ArrayLike<number>, linear = false, eOff = 0, dOff = eOff) {
  const L = new Float64Array(d * d), a = new Float64Array(h);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < h; i++) {
      const e = E[eOff + i * d + j], t = Math.tanh(e);
      a[i] = linear ? e : t * t * t;
    }
    for (let k = 0; k < d; k++) {
      let v = 0;
      for (let i = 0; i < h; i++) v += D[dOff + k * h + i] * a[i];
      L[j * d + k] = v;
    }
  }
  return L;
}

/** The off-diagonal entries of a d × d map, in a fixed order (row by row). */
export function offDiag(L: ArrayLike<number>, d: number) {
  const o = new Float64Array(d * (d - 1));
  let q = 0;
  for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) if (j !== k) o[q++] = L[j * d + k];
  return o;
}

export const meanOf = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
};
export const rmsOf = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / a.length);
};

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>) {
  const n = a.length, ma = meanOf(a), mb = meanOf(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / Math.sqrt(saa * sbb || 1);
}

/**
 * The post's analogy: three Ising spins on a triangle with antiferromagnetic bonds,
 * H = Σ_{⟨ab⟩} σ_a σ_b. Enumerates all 2³ states and counts the ground states.
 */
export interface TriState {
  spins: [number, number, number];
  unsat: number; // bonds whose two spins are aligned
  energy: number;
}
export const TRI_BONDS: [number, number][] = [[0, 1], [1, 2], [2, 0]];

export function triangleStates() {
  const states: TriState[] = [];
  for (let k = 0; k < 8; k++) {
    const spins: [number, number, number] = [k & 1 ? -1 : 1, k & 2 ? -1 : 1, k & 4 ? -1 : 1];
    let unsat = 0, energy = 0;
    for (const [a, b] of TRI_BONDS) {
      energy += spins[a] * spins[b];
      if (spins[a] === spins[b]) unsat++;
    }
    states.push({ spins, unsat, energy });
  }
  const minUnsat = Math.min(...states.map((s) => s.unsat));
  return { states, minUnsat, nGround: states.filter((s) => s.unsat === minUnsat).length };
}
