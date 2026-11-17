/**
 * A miniature of Causal Perturbative Elicitation (Mack, Panickssery & Turner, arXiv:2606.29604):
 * Rank-one weight edits searched by gradient iteration with optional soft orthogonalization.
 * The update maximizes the squared norm of a single input's causal-map response. This keeps the
 * paper's soft-orthogonalization step but omits its separate similarity penalty and learned output factors.
 *
 * The "model slice" between the edited source layer and the target layer is
 *     Z(h) = h + W σ(V h − τ)
 * with K planted detector directions v_j (rows of V, orthonormal) that are OFF at the baseline
 * activation h0, each writing to its own output direction w_j. A rank-one edit of the source
 * projection, W_O ↦ W_O + R·b aᵀ, changes the activation to h = h0 + R·b (a·s) for input s, and the
 * causal map is Δ(φ) = Z(h) − Z(h0) with φ = (a, b), ‖a‖ = ‖b‖ = 1.
 *
 * Pure numerics, no rendering. Everything is deterministic given the two samplers passed in.
 */

export interface ToySpec {
  d: number; // residual width
  K: number; // number of planted mechanisms
  tau: number; // detector threshold
  sharp: number; // softplus sharpness
  gain: number; // ‖w_j‖
  sNorm: number; // ‖s‖, the input read by the edit
  R: number; // perturbation radius (finite!)
  h0Norm: number;
  offLo: number; // v_j·h0 is drawn uniformly from [offLo, offHi]: how far below threshold each mechanism sleeps
  offHi: number;
}

export interface Toy extends ToySpec {
  V: Float64Array; // K × d, orthonormal rows
  W: Float64Array; // K × d, row j = output direction of detector j (norm = gain)
  h0: Float64Array;
  s: Float64Array;
  sHat: Float64Array;
  pre0: Float64Array; // v_j·h0 − τ (all negative: asleep)
  act0: Float64Array;
}

const dotAt = (a: Float64Array, ao: number, b: Float64Array, bo: number, n: number) => {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
  return s;
};
const normalizeAt = (a: Float64Array, o: number, n: number) => {
  const l = Math.sqrt(dotAt(a, o, a, o, n));
  if (l > 0) for (let i = 0; i < n; i++) a[o + i] /= l;
  return l;
};

export function makeToy(rnd: () => number, g: () => number, spec: ToySpec): Toy {
  const { d, K } = spec;
  const V = new Float64Array(K * d), W = new Float64Array(K * d);
  for (let j = 0; j < K; j++) {
    for (let i = 0; i < d; i++) V[j * d + i] = g();
    for (let k = 0; k < j; k++) {
      const c = dotAt(V, j * d, V, k * d, d);
      for (let i = 0; i < d; i++) V[j * d + i] -= c * V[k * d + i];
    }
    normalizeAt(V, j * d, d);
  }
  for (let j = 0; j < K; j++) {
    for (let i = 0; i < d; i++) W[j * d + i] = g();
    normalizeAt(W, j * d, d);
    for (let i = 0; i < d; i++) W[j * d + i] *= spec.gain;
  }
  // baseline activation: a random vector with a planted (negative) offset along every detector
  const h0 = new Float64Array(d);
  for (let i = 0; i < d; i++) h0[i] = g();
  for (let j = 0; j < K; j++) {
    const c = dotAt(h0, 0, V, j * d, d);
    for (let i = 0; i < d; i++) h0[i] -= c * V[j * d + i];
  }
  normalizeAt(h0, 0, d);
  for (let i = 0; i < d; i++) h0[i] *= spec.h0Norm;
  for (let j = 0; j < K; j++) {
    const off = spec.offLo + (spec.offHi - spec.offLo) * rnd();
    for (let i = 0; i < d; i++) h0[i] += off * V[j * d + i];
  }
  const sHat = new Float64Array(d);
  for (let i = 0; i < d; i++) sHat[i] = g();
  normalizeAt(sHat, 0, d);
  const s = sHat.map((x) => x * spec.sNorm);
  const pre0 = new Float64Array(K), act0 = new Float64Array(K);
  for (let j = 0; j < K; j++) {
    pre0[j] = dotAt(V, j * d, h0, 0, d) - spec.tau;
    act0[j] = softplus(pre0[j], spec.sharp);
  }
  return { ...spec, V, W, h0, s, sHat, pre0, act0 };
}

export const softplus = (x: number, k: number) => (x * k > 30 ? x : Math.log1p(Math.exp(x * k)) / k);
const dsoftplus = (x: number, k: number) => 1 / (1 + Math.exp(-x * k));

/** Δ(φ) for the edit (a, b) stored at offset o of A / B. Writes Δ into `out`, detector state into pre / act. */
export function causalMap(toy: Toy, A: Float64Array, B: Float64Array, o: number, out: Float64Array, pre: Float64Array, act: Float64Array) {
  const { d, K, V, W } = toy;
  const as = toy.R * dotAt(A, o, toy.s, 0, d);
  for (let i = 0; i < d; i++) out[i] = as * B[o + i];
  for (let j = 0; j < K; j++) {
    const p = toy.pre0[j] + as * dotAt(V, j * d, B, o, d);
    pre[j] = p;
    act[j] = softplus(p, toy.sharp);
    const c = act[j] - toy.act0[j];
    for (let i = 0; i < d; i++) out[i] += c * W[j * d + i];
  }
  return as;
}

export interface Snapshot {
  /** m × K signed cosines between each candidate edit b aᵀ and each planted edit v_j ŝᵀ (Frobenius). */
  cos: Float32Array;
  /** ‖Δ‖ of each candidate. */
  dn: Float32Array;
}

export interface RunOpts {
  m: number;
  iters: number;
  repel: boolean;
  T: number; // soft-orthogonalization temperature
  inner: number; // inner iterations of Algorithm 2
  /** Keep the strongest `links` repulsion weights per candidate and iteration as diagnostics. */
  links?: number;
}

export interface Run {
  m: number;
  iters: number;
  sub: number; // snapshots per iteration (2 with repulsion: after the gradient jump, after the push)
  snaps: Snapshot[]; // [0] = random initialization
  A0: Float64Array; B0: Float64Array; // m × d initial factors
  A: Float64Array; B: Float64Array; // m × d final factors
  /** per iteration: (i, j, W_ij) triples of the strongest repulsions in the first inner pass */
  links: Float32Array[];
  /** per iteration, per candidate: ‖G_φ‖ (the "loudness" used as the logit bias) */
  loud: Float32Array[];
}

/** Gradient iteration on the toy slice, with Algorithm 2's push when repel is true. */
export function runSOGI(toy: Toy, g: () => number, o: RunOpts): Run {
  const { d, K } = toy, m = o.m;
  const A = new Float64Array(m * d), B = new Float64Array(m * d);
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < d; c++) A[i * d + c] = g();
    normalizeAt(A, i * d, d);
    for (let c = 0; c < d; c++) B[i * d + c] = g();
    normalizeAt(B, i * d, d);
  }
  const A0 = A.slice(), B0 = B.slice();
  const u = new Float64Array(d), q = new Float64Array(d), pre = new Float64Array(K), act = new Float64Array(K);
  const GA = new Float64Array(m * d), GB = new Float64Array(m * d), gn = new Float64Array(m), lb = new Float64Array(m);
  const SA = new Float64Array(m * d), SB = new Float64Array(m * d), S = new Float64Array(m);

  const snap = (): Snapshot => {
    const cos = new Float32Array(m * K), dn = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      causalMap(toy, A, B, i * d, u, pre, act);
      dn[i] = Math.sqrt(dotAt(u, 0, u, 0, d));
      const ah = dotAt(A, i * d, toy.sHat, 0, d);
      for (let j = 0; j < K; j++) cos[i * K + j] = ah * dotAt(B, i * d, toy.V, j * d, d);
    }
    return { cos, dn };
  };

  const snaps: Snapshot[] = [snap()];
  const links: Float32Array[] = [], loud: Float32Array[] = [];
  const nl = o.links ?? 0;
  for (let it = 0; it < o.iters; it++) {
    // u_i ← Δ(Rφ_i);  φ_i ← ∇_φ ⟨u_i, Δ(Rφ_i)⟩  — the gradient REPLACES the iterate (power iteration)
    for (let i = 0; i < m; i++) {
      const as = causalMap(toy, A, B, i * d, u, pre, act);
      q.set(u);
      for (let j = 0; j < K; j++) {
        const c = dsoftplus(pre[j], toy.sharp) * dotAt(toy.W, j * d, u, 0, d);
        for (let k = 0; k < d; k++) q[k] += c * toy.V[j * d + k];
      }
      const bq = dotAt(B, i * d, q, 0, d);
      let n2 = 0;
      for (let k = 0; k < d; k++) {
        const gb = as * q[k], ga = toy.R * bq * toy.s[k];
        GB[i * d + k] = gb; GA[i * d + k] = ga;
        n2 += gb * gb + ga * ga;
      }
      gn[i] = Math.sqrt(n2);
    }
    A.set(GA); B.set(GB);
    for (let i = 0; i < m; i++) { normalizeAt(A, i * d, d); normalizeAt(B, i * d, d); }
    loud.push(Float32Array.from(gn));
    snaps.push(snap());
    if (!o.repel) continue;

    // Algorithm 2: soft orthogonalization with log ‖G_φj‖ as the logit bias
    for (let i = 0; i < m; i++) lb[i] = Math.log(gn[i] + 1e-12);
    const lk = new Float32Array(m * nl * 3);
    for (let k = 0; k < o.inner; k++) {
      SA.fill(0); SB.fill(0);
      for (let i = 0; i < m; i++) {
        let mx = -Infinity;
        for (let j = 0; j < m; j++) {
          if (j === i) continue;
          S[j] = (dotAt(A, i * d, A, j * d, d) + dotAt(B, i * d, B, j * d, d)) / o.T + lb[j];
          if (S[j] > mx) mx = S[j];
        }
        let z = 0;
        for (let j = 0; j < m; j++) if (j !== i) { S[j] = Math.exp(S[j] - mx); z += S[j]; }
        for (let j = 0; j < m; j++) {
          if (j === i) continue;
          const w = S[j] / z;
          S[j] = w;
          for (let c = 0; c < d; c++) { SA[i * d + c] += w * A[j * d + c]; SB[i * d + c] += w * B[j * d + c]; }
        }
        if (k === 0 && nl) {
          S[i] = -1;
          for (let r = 0; r < nl; r++) {
            let best = -1, bw = 0;
            for (let j = 0; j < m; j++) if (S[j] > bw) { bw = S[j]; best = j; }
            const p = (i * nl + r) * 3;
            lk[p] = i; lk[p + 1] = best; lk[p + 2] = bw;
            if (best >= 0) S[best] = -1;
          }
        }
      }
      for (let i = 0; i < m; i++) {
        // φ_i ← φ_i − Σ_j W_ij φ_j, then project back to the torus (unit a, unit b)
        let na = 0, nb = 0;
        for (let c = 0; c < d; c++) {
          SA[i * d + c] = A[i * d + c] - SA[i * d + c];
          SB[i * d + c] = B[i * d + c] - SB[i * d + c];
          na += SA[i * d + c] ** 2; nb += SB[i * d + c] ** 2;
        }
        // with a single prompt every a_i is ±ŝ, so the a-part can cancel exactly: keep it then
        if (na > 1e-12) { const l = Math.sqrt(na); for (let c = 0; c < d; c++) A[i * d + c] = SA[i * d + c] / l; }
        if (nb > 1e-12) { const l = Math.sqrt(nb); for (let c = 0; c < d; c++) B[i * d + c] = SB[i * d + c] / l; }
      }
    }
    links.push(lk);
    snaps.push(snap());
  }
  return { m, iters: o.iters, sub: o.repel ? 2 : 1, snaps, A0, B0, A: A.slice(), B: B.slice(), links, loud };
}

export interface Tally {
  found: number; // planted mechanisms with at least one candidate at cos > thr
  counts: Int32Array; // candidates per mechanism
  owner: Int32Array; // per candidate: mechanism index or −1
  idle: number; // candidates on no mechanism
  mean: number; // mean ‖Δ‖
  maxPile: number;
}

export function tally(run: Run, K: number, idx: number, thr = 0.9): Tally {
  const s = run.snaps[idx], m = run.m;
  const counts = new Int32Array(K), owner = new Int32Array(m).fill(-1);
  let idle = 0, mean = 0;
  for (let i = 0; i < m; i++) {
    let best = -1, bv = thr;
    for (let j = 0; j < K; j++) if (s.cos[i * K + j] > bv) { bv = s.cos[i * K + j]; best = j; }
    owner[i] = best;
    if (best >= 0) counts[best]++; else idle++;
    mean += s.dn[i] / m;
  }
  let found = 0, maxPile = 0;
  for (let j = 0; j < K; j++) { if (counts[j] > 0) found++; if (counts[j] > maxPile) maxPile = counts[j]; }
  return { found, counts, owner, idle, mean, maxPile };
}
