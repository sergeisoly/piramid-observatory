import { mulberry32, gaussian } from '../lib/math';

/**
 * The miniature behind the MEAN FIELD chapter (the film's own settings, not the post's experiment).
 *
 *   f(x) = s Σ_i a_i φ(w_i·x),   φ(z) = max(0, tanh z),   θ_i = (w_i1, w_i2, a_i),   prior θ ~ N(0, 1)
 *   data: 8×8 midpoint grid on [−1,1]², ⟨·⟩ = mean over the 64 grid points, r = y − f
 *
 * Two scalings of the same Gibbs posterior family (b = per-neuron coupling, s = readout scale):
 *   mean-field  s = 1/N,   p(θ) ∝ ρ0(θ) exp(−βN ⟨r²⟩)   →  b = 2β        (features move)
 *   lazy / NNGP s = 1/√N,  p(θ) ∝ ρ0(θ) exp(−β  ⟨r²⟩)   →  b = 2β/√N     (textbook contrast)
 * Both have the same ratio of prior function variance to observation noise.
 *
 * Langevin dynamics (Euler–Maruyama):
 *   θ_i ← θ_i + η (b ⟨ r ∇_θi (a_i φ(w_i·x)) ⟩ − θ_i) + sqrt(2η) ξ
 *
 * For a FROZEN field r the neurons decouple and the stationary single-neuron law is
 *   ρ_r(w, a) ∝ exp(−|θ|²/2 + b a c_r(w)),   c_r(w) = ⟨ r φ(w·x) ⟩,
 * i.e. a | w ~ N(b c_r(w), 1) and ρ_r(w) ∝ exp(−|w|²/2 + b² c_r(w)²/2). The deterministic
 * self-consistency equation is r = y − sN·E_{ρ_r}[a φ(w·x)], solved below by Newton.
 */
export const G = 8;
export const M = G * G;

export type TargetId = 'x1x2' | 'x1sq-x2sq' | 'x1';

/**
 * `label` is plain text (it is shown in a <select>, which cannot typeset); `tex` is the same
 * formula as TeX without dollars; `rule` is a sentence fragment with its maths in dollars.
 */
export const TARGETS: Record<TargetId, { label: string; tex: string; fn: (x1: number, x2: number) => number; lobes: string; rule: string }> = {
  x1x2: { label: 'x₁x₂', tex: 'x_1x_2', fn: (a, b) => a * b, lobes: 'four lobes', rule: '$\\operatorname{sign}(a) = \\operatorname{sign}(w_1w_2)$' },
  'x1sq-x2sq': { label: 'x₁² − x₂²', tex: 'x_1^2 - x_2^2', fn: (a, b) => a * a - b * b, lobes: 'four lobes', rule: '$\\operatorname{sign}(a) = \\operatorname{sign}(w_1^2 - w_2^2)$' },
  x1: { label: 'x₁', tex: 'x_1', fn: (a) => a, lobes: 'two lobes', rule: '$\\operatorname{sign}(a) = \\operatorname{sign}(w_1)$' },
};

export interface Data {
  X1: Float64Array;
  X2: Float64Array;
  Y: Float64Array;
  varY: number;
}

export function makeData(target: TargetId): Data {
  const X1 = new Float64Array(M), X2 = new Float64Array(M), Y = new Float64Array(M);
  let v = 0;
  for (let j = 0; j < G; j++)
    for (let i = 0; i < G; i++) {
      const k = j * G + i;
      X1[k] = -1 + (2 * i + 1) / G;
      X2[k] = -1 + (2 * j + 1) / G;
      Y[k] = TARGETS[target].fn(X1[k], X2[k]);
      v += Y[k] * Y[k];
    }
  return { X1, X2, Y, varY: v / M };
}

export interface RunOpts {
  seed: number;
  n: number; // number of neurons in this population
  pre: number; // prior-only (b = 0) steps before step 0
  steps: number;
  every: number; // snapshot stride
  eta: number;
  b: number; // per-neuron coupling (2β mean-field, 2β/√N lazy)
  out: number; // readout scale applied to Σ a φ of this population
  frozen: ArrayLike<number> | null; // null = coupled through its own residual
}

export interface RunData {
  n: number;
  pre: number;
  steps: number;
  every: number;
  nSnap: number;
  /** nSnap × n × 3 (w1, w2, a); snapshot j is step j·every − pre. */
  snaps: Float32Array;
  /** nSnap × M: y − f of THIS population read as a network (for the coupled run this is the field itself). */
  field: Float32Array;
  /** Per step (index = step + pre). `loss` = ⟨(y − f)²⟩ of this population read as a network. */
  loss: Float32Array;
  ea2: Float32Array;
  mw: Float32Array;
  /** Time average of y − f over the second half of the run. */
  rAvg: Float64Array;
  /** Second-half time averages. */
  avg: { loss: number; ea2: number; mw: number };
}

/**
 * Euler–Maruyama Langevin sampler. `frozen = null` couples the neurons through their own
 * residual (the "gas"); otherwise every neuron sees the fixed field `frozen` and the neurons
 * are independent (the cavity test). The first `pre` steps run with b = 0 (prior only).
 */
export function langevin(d: Data, o: RunOpts) {
  const { n, pre, steps, every, eta, frozen } = o;
  const { X1, X2, Y } = d;
  const g = gaussian(mulberry32(o.seed));
  const th = new Float64Array(n * 3);
  for (let i = 0; i < n * 3; i++) th[i] = g();
  const total = pre + steps;
  const nSnap = Math.floor(total / every) + 1;
  const data: RunData = {
    n, pre, steps, every, nSnap,
    snaps: new Float32Array(nSnap * n * 3),
    field: new Float32Array(nSnap * M),
    loss: new Float32Array(total + 1),
    ea2: new Float32Array(total + 1),
    mw: new Float32Array(total + 1),
    rAvg: new Float64Array(M),
    avg: { loss: 0, ea2: 0, mw: 0 },
  };
  const ph = new Float64Array(n * M), dph = new Float64Array(n * M);
  const f = new Float64Array(M), r = new Float64Array(M), own = new Float64Array(M);
  const sq = Math.sqrt(2 * eta);
  const half = pre + Math.floor(steps / 2);
  const LIM = 1e3; // NaN guard only: the drift is bounded, so no parameter comes near this at any setting
  let s = 0, nAvg = 0;

  /** Run up to `count` more steps; returns true when the run is complete. */
  function advance(count: number) {
    for (let c = 0; c < count && s <= total; c++, s++) {
      // forward pass: features and network output on the 64 data points
      f.fill(0);
      for (let i = 0; i < n; i++) {
        const w1 = th[i * 3], w2 = th[i * 3 + 1], a = th[i * 3 + 2], q = i * M;
        for (let k = 0; k < M; k++) {
          const z = w1 * X1[k] + w2 * X2[k];
          let p = 0, dp = 0;
          if (z > 0) { p = Math.tanh(z); dp = 1 - p * p; }
          ph[q + k] = p; dph[q + k] = dp;
          f[k] += a * p;
        }
      }
      let loss = 0;
      for (let k = 0; k < M; k++) {
        own[k] = Y[k] - f[k] * o.out;
        loss += own[k] * own[k];
        r[k] = frozen ? frozen[k] : own[k];
      }
      loss /= M;
      let ea2 = 0, mw = 0;
      for (let i = 0; i < n; i++) {
        ea2 += th[i * 3 + 2] * th[i * 3 + 2];
        mw += Math.hypot(th[i * 3], th[i * 3 + 1]);
      }
      ea2 /= n; mw /= n;
      data.loss[s] = loss; data.ea2[s] = ea2; data.mw[s] = mw;
      if (s % every === 0) {
        const j = s / every;
        data.snaps.set(th, j * n * 3);
        data.field.set(own, j * M);
      }
      if (s >= half) {
        for (let k = 0; k < M; k++) data.rAvg[k] += own[k];
        data.avg.loss += loss; data.avg.ea2 += ea2; data.avg.mw += mw;
        nAvg++;
      }
      if (s === total) {
        for (let k = 0; k < M; k++) data.rAvg[k] /= nAvg;
        data.avg.loss /= nAvg; data.avg.ea2 /= nAvg; data.avg.mw /= nAvg;
        s++;
        return true;
      }
      // Langevin step
      const b = s < pre ? 0 : o.b;
      for (let i = 0; i < n; i++) {
        const q = i * M, aa = th[i * 3 + 2];
        let g1 = 0, g2 = 0, ga = 0;
        if (b !== 0)
          for (let k = 0; k < M; k++) {
            const rk = r[k];
            ga += rk * ph[q + k];
            const cc = rk * aa * dph[q + k];
            g1 += cc * X1[k]; g2 += cc * X2[k];
          }
        let v1 = th[i * 3] + eta * ((b * g1) / M - th[i * 3]) + sq * g();
        let v2 = th[i * 3 + 1] + eta * ((b * g2) / M - th[i * 3 + 1]) + sq * g();
        let v3 = aa + eta * ((b * ga) / M - aa) + sq * g();
        if (!(v1 > -LIM)) v1 = -LIM; else if (v1 > LIM) v1 = LIM;
        if (!(v2 > -LIM)) v2 = -LIM; else if (v2 > LIM) v2 = LIM;
        if (!(v3 > -LIM)) v3 = -LIM; else if (v3 > LIM) v3 = LIM;
        th[i * 3] = v1; th[i * 3 + 1] = v2; th[i * 3 + 2] = v3;
      }
    }
    return s > total;
  }
  return { data, advance };
}

/** Dense solve of A x = b (Gaussian elimination with partial pivoting); A and b are overwritten. */
function solve(A: Float64Array, b: Float64Array, n: number) {
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(A[j * n + i]) > Math.abs(A[piv * n + i])) piv = j;
    if (piv !== i) {
      for (let k = 0; k < n; k++) { const t = A[i * n + k]; A[i * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
      const t = b[i]; b[i] = b[piv]; b[piv] = t;
    }
    const dd = A[i * n + i];
    for (let j = i + 1; j < n; j++) {
      const q = A[j * n + i] / dd;
      if (q === 0) continue;
      for (let k = i; k < n; k++) A[j * n + k] -= q * A[i * n + k];
      b[j] -= q * b[i];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= A[i * n + k] * b[k];
    b[i] = s / A[i * n + i];
  }
  return b;
}

/**
 * Exact single-neuron theory on a GW×GW quadrature grid over w ∈ [−L, L]² (a is integrated
 * analytically). `b` is the per-neuron coupling and `sN` = (readout scale)·N. Gives moments
 * of ρ_r, i.i.d. samples of ρ_r, and the Newton solution of r = y − sN·E_{ρ_r}[a φ].
 */
export function singleNeuronTheory(d: Data, b: number, sN: number, GW = 96, L = 4.5) {
  const { X1, X2, Y } = d;
  const nc = GW * GW, hw = (2 * L) / GW;
  const PHI = new Float32Array(nc * M);
  const W1 = new Float32Array(nc), W2 = new Float32Array(nc), PRI = new Float64Array(nc);
  for (let j = 0; j < GW; j++)
    for (let i = 0; i < GW; i++) {
      const c = j * GW + i, w1 = -L + (i + 0.5) * hw, w2 = -L + (j + 0.5) * hw;
      W1[c] = w1; W2[c] = w2; PRI[c] = -(w1 * w1 + w2 * w2) / 2;
      for (let k = 0; k < M; k++) {
        const z = w1 * X1[k] + w2 * X2[k];
        PHI[c * M + k] = z > 0 ? Math.tanh(z) : 0;
      }
    }

  /** Cell probabilities p (w-marginal of ρ_r) and c_r(w) per cell. */
  function density(r: ArrayLike<number>) {
    const cc = new Float64Array(nc), p = new Float64Array(nc);
    let mx = -Infinity;
    for (let c = 0; c < nc; c++) {
      let s = 0;
      const q = c * M;
      for (let k = 0; k < M; k++) s += r[k] * PHI[q + k];
      s /= M;
      cc[c] = s;
      const l = PRI[c] + 0.5 * b * b * s * s;
      p[c] = l;
      if (l > mx) mx = l;
    }
    let Z = 0;
    for (let c = 0; c < nc; c++) { p[c] = Math.exp(p[c] - mx); Z += p[c]; }
    for (let c = 0; c < nc; c++) p[c] /= Z;
    return { p, cc };
  }

  /** E[a²], E|w|, F(x) = sN·E[a φ(w·x)] and (optionally) the covariance of a φ(w·x) over the data points. */
  function moments(r: ArrayLike<number>, wantCov = false) {
    const { p, cc } = density(r);
    let ea2 = 0, mw = 0;
    const F = new Float64Array(M);
    const Cov = wantCov ? new Float64Array(M * M) : null;
    for (let c = 0; c < nc; c++) {
      const pc = p[c];
      if (pc < 1e-14) continue;
      const m = b * cc[c], q = c * M;
      ea2 += pc * (1 + m * m);
      mw += pc * Math.hypot(W1[c], W2[c]);
      const pm = pc * m;
      for (let k = 0; k < M; k++) F[k] += pm * PHI[q + k];
      if (Cov) {
        const e2 = pc * (1 + m * m);
        for (let k = 0; k < M; k++) {
          const a = e2 * PHI[q + k];
          if (a === 0) continue;
          const row = k * M;
          for (let l = k; l < M; l++) Cov[row + l] += a * PHI[q + l];
        }
      }
    }
    if (Cov)
      for (let k = 0; k < M; k++)
        for (let l = k; l < M; l++) {
          const v = Cov[k * M + l] - F[k] * F[l];
          Cov[k * M + l] = v; Cov[l * M + k] = v;
        }
    for (let k = 0; k < M; k++) F[k] *= sN;
    return { ea2, mw, F, Cov };
  }

  const resid = (r: ArrayLike<number>, F: ArrayLike<number>, out: Float64Array) => {
    let nr = 0;
    for (let k = 0; k < M; k++) { out[k] = r[k] - Y[k] + F[k]; nr += out[k] * out[k]; }
    return Math.sqrt(nr / M);
  };

  /** Damped Newton on R(r) = r − y + sN·E_{ρ_r}[aφ]; Jacobian = I + sN·(b/M)·Cov_{ρ_r}(aφ). */
  function fixedPoint(r0: ArrayLike<number>, maxIter = 14) {
    let r = Float64Array.from(r0);
    const R = new Float64Array(M), Rt = new Float64Array(M);
    let res = Infinity, iters = 0;
    for (; iters < maxIter; iters++) {
      const { F, Cov } = moments(r, true);
      res = resid(r, F, R);
      if (!(res > 1e-9)) break;
      const J = new Float64Array(M * M);
      for (let k = 0; k < M; k++)
        for (let l = 0; l < M; l++) J[k * M + l] = (k === l ? 1 : 0) + ((sN * b) / M) * Cov![k * M + l];
      const step = solve(J, Float64Array.from(R), M);
      // backtracking: accept the longest step that does not increase the residual
      let lam = 1, accepted = false;
      for (let h = 0; h < 7; h++, lam *= 0.5) {
        const trial = new Float64Array(M);
        for (let k = 0; k < M; k++) trial[k] = r[k] - lam * step[k];
        const rt = resid(trial, moments(trial).F, Rt);
        if (isFinite(rt) && rt < res) { r = trial; accepted = true; break; }
      }
      if (!accepted) break;
    }
    const m = moments(r);
    res = resid(r, m.F, R);
    let loss = 0;
    for (let k = 0; k < M; k++) loss += r[k] * r[k];
    return { r, loss: loss / M, ea2: m.ea2, mw: m.mw, iters, res, ok: isFinite(res) && res < 1e-6 };
  }

  /** n i.i.d. samples (w1, w2, a) of ρ_r: w from the grid marginal (jittered inside its cell), a | w exact. */
  function sample(r: ArrayLike<number>, n: number, seed: number) {
    const rnd = mulberry32(seed), g = gaussian(rnd);
    const { p } = density(r);
    const cdf = new Float64Array(nc);
    let acc = 0;
    for (let c = 0; c < nc; c++) { acc += p[c]; cdf[c] = acc; }
    const out = new Float32Array(n * 3);
    for (let q = 0; q < n; q++) {
      const u = rnd() * acc;
      let lo = 0, hi = nc - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      const w1 = W1[lo] + (rnd() - 0.5) * hw, w2 = W2[lo] + (rnd() - 0.5) * hw;
      let c = 0;
      for (let k = 0; k < M; k++) {
        const z = w1 * X1[k] + w2 * X2[k];
        if (z > 0) c += r[k] * Math.tanh(z);
      }
      out[q * 3] = w1; out[q * 3 + 1] = w2; out[q * 3 + 2] = (b * c) / M + g();
    }
    return out;
  }

  return { moments, fixedPoint, sample };
}

export interface Plan {
  N: number;
  K: number; // size of the fresh (test) populations
  beta: number;
  lazy: boolean;
  eta: number;
  b: number;
  out: number; // readout scale of the width-N network
  every: number;
  preMain: number;
  stepsMain: number;
  preTest: number;
  stepsTest: number;
}

/**
 * Step sizes and step counts for a given (N, β, scaling). At the film's settings this is
 * exactly η = 0.01, 2500 coupled steps and 1500 steps for each test population. For larger β
 * the step size shrinks as 1/β (the stiffest function-space mode relaxes at a rate ∝ β) and
 * the step counts are capped so that any setting builds in a couple of seconds.
 */
export function plan(N: number, beta: number, lazy: boolean, preTimeMain: number, preTimeTest: number): Plan {
  const eta = Math.min(0.01, 1 / beta);
  const every = Math.max(1, Math.round(0.05 / eta));
  const snap = (x: number) => Math.max(every, Math.round(x / every) * every);
  const stepsMain = snap(Math.min(25 / eta, 1.6e6 / N));
  const stepsTest = snap(Math.min(15 / eta, 3200));
  return {
    N, K: 256, beta, lazy, eta, every,
    b: lazy ? (2 * beta) / Math.sqrt(N) : 2 * beta,
    out: lazy ? 1 / Math.sqrt(N) : 1 / N,
    preMain: snap(preTimeMain / eta), stepsMain,
    preTest: snap(preTimeTest / eta), stepsTest,
  };
}
