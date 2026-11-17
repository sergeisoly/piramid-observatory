/**
 * Numerics for the `entk` chapter. Callers supply the seeded samplers. The code has no DOM dependencies
 * and can also run under Node.
 *
 * Model     f(x) = W2 (W1 x)²   — quadratic MLP, no biases, x = onehot(a) ⊕ onehot(b)
 * Kernel    K̄(x,x') = (1/p) Σ_c Σ_θ ∂f_c(x)/∂θ · ∂f_c(x')/∂θ   (class-averaged empirical NTK)
 *                    = (x·x') · (4/p) Σ_k s_k h_k(x) h_k(x')  +  Σ_k h_k(x)² h_k(x')²,   s_k = Σ_c W2_ck²
 */

export interface TrainOpts {
  p: number; // modulus
  n: number; // hidden width
  epochs: number;
  lr: number;
  wd: number;
  trainFrac: number;
  sigma1: number; // std of the W1 entries at initialization
  sigma2: number; // std of the W2 entries at initialization
  checkpoints: number[]; // epochs at which the weights are kept (0 = initialization)
}

export interface TrainResult {
  p: number;
  n: number;
  N: number;
  isTrain: Uint8Array; // N
  nTrain: number;
  /** length epochs + 1; entry e is measured with the weights after e optimizer steps. */
  trainAcc: Float32Array;
  testAcc: Float32Array;
  loss: Float32Array;
  /** (epochs + 1) × N: 1 where argmax f(x) = (a + b) mod p. */
  correct: Uint8Array;
  /** Weights at each requested checkpoint. W1 is stored column-major: W1[col * n + k]. */
  snaps: { epoch: number; W1: Float64Array; W2: Float64Array }[];
}

/**
 * Full-batch AdamW on MSE to one-hot targets. Deterministic given the two samplers.
 * Returns a resumable trainer so that the caller can yield to the browser between chunks:
 * `advance(k)` runs k more epochs and returns true once all epochs are done.
 */
export function makeTrainer(o: TrainOpts, rnd: () => number, gauss: () => number): { result: TrainResult; advance(k: number): boolean } {
  const { p, n, epochs } = o;
  const N = p * p, D = 2 * p;
  // ── data split (Fisher–Yates on the seeded RNG) ──
  const perm = new Int32Array(N);
  for (let i = 0; i < N; i++) perm[i] = i;
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  const nTrain = Math.round(o.trainFrac * N);
  const isTrain = new Uint8Array(N);
  for (let i = 0; i < nTrain; i++) isTrain[perm[i]] = 1;

  // ── parameters ──
  const W1 = new Float64Array(D * n); // column-major: W1[col * n + k]
  const W2 = new Float64Array(p * n); // row-major:    W2[c * n + k]
  for (let i = 0; i < W1.length; i++) W1[i] = gauss() * o.sigma1;
  for (let i = 0; i < W2.length; i++) W2[i] = gauss() * o.sigma2;
  const m1 = new Float64Array(W1.length), v1 = new Float64Array(W1.length), g1 = new Float64Array(W1.length);
  const m2 = new Float64Array(W2.length), v2 = new Float64Array(W2.length), g2 = new Float64Array(W2.length);

  const H = new Float64Array(N * n), S = new Float64Array(N * n);
  const E = new Float64Array(N * p); // dL/df on the training rows
  const trainAcc = new Float32Array(epochs + 1), testAcc = new Float32Array(epochs + 1), loss = new Float32Array(epochs + 1);
  const correct = new Uint8Array((epochs + 1) * N);
  const snaps: TrainResult['snaps'] = [];
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const gScale = 2 / (nTrain * p);
  const fx = new Float64Array(p), ds = new Float64Array(n);

  const result: TrainResult = { p, n, N, isTrain, nTrain, trainAcc, testAcc, loss, correct, snaps };
  let ep = 0, done = false;
  const epoch = () => {
    if (o.checkpoints.includes(ep)) snaps.push({ epoch: ep, W1: W1.slice(), W2: W2.slice() });
    // forward on every input, error signal on the training ones
    let nTr = 0, nTe = 0, L = 0;
    for (let x = 0; x < N; x++) {
      const a = (x / p) | 0, b = x - a * p, y = (a + b) % p;
      const ca = a * n, cb = (p + b) * n, xo = x * n;
      for (let k = 0; k < n; k++) {
        const h = W1[ca + k] + W1[cb + k];
        H[xo + k] = h;
        S[xo + k] = h * h;
      }
      let best = -Infinity, arg = 0;
      for (let c = 0; c < p; c++) {
        const wo = c * n;
        let f0 = 0, f1 = 0, f2 = 0, f3 = 0;
        for (let k = 0; k < n; k += 4) {
          f0 += W2[wo + k] * S[xo + k];
          f1 += W2[wo + k + 1] * S[xo + k + 1];
          f2 += W2[wo + k + 2] * S[xo + k + 2];
          f3 += W2[wo + k + 3] * S[xo + k + 3];
        }
        const f = f0 + f1 + f2 + f3;
        fx[c] = f;
        if (f > best) { best = f; arg = c; }
      }
      const ok = arg === y ? 1 : 0;
      correct[ep * N + x] = ok;
      if (isTrain[x]) {
        nTr += ok;
        for (let c = 0; c < p; c++) {
          const r = fx[c] - (c === y ? 1 : 0);
          L += r * r;
          E[x * p + c] = r * gScale;
        }
      } else nTe += ok;
    }
    trainAcc[ep] = nTr / nTrain;
    testAcc[ep] = nTe / (N - nTrain);
    loss[ep] = L / (nTrain * p);
    if (ep === epochs) {
      done = true;
      return;
    }

    // backward: dL/dW2 = Eᵀ S,  dL/dh = 2 h ∘ (E W2),  dL/dW1 scatters into the two active columns
    g1.fill(0);
    g2.fill(0);
    for (let x = 0; x < N; x++) {
      if (!isTrain[x]) continue;
      const a = (x / p) | 0, b = x - a * p;
      const ca = a * n, cb = (p + b) * n, xo = x * n, eo = x * p;
      ds.fill(0);
      for (let c = 0; c < p; c++) {
        const e = E[eo + c], wo = c * n;
        for (let k = 0; k < n; k += 2) {
          g2[wo + k] += e * S[xo + k];
          ds[k] += e * W2[wo + k];
          g2[wo + k + 1] += e * S[xo + k + 1];
          ds[k + 1] += e * W2[wo + k + 1];
        }
      }
      for (let k = 0; k < n; k++) {
        const dh = 2 * H[xo + k] * ds[k];
        g1[ca + k] += dh;
        g1[cb + k] += dh;
      }
    }
    // AdamW (decoupled weight decay, bias-corrected moments)
    const t = ep + 1, c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t), decay = 1 - o.lr * o.wd;
    for (let i = 0; i < W1.length; i++) {
      const g = g1[i];
      m1[i] = b1 * m1[i] + (1 - b1) * g;
      v1[i] = b2 * v1[i] + (1 - b2) * g * g;
      W1[i] = W1[i] * decay - (o.lr * (m1[i] / c1)) / (Math.sqrt(v1[i] / c2) + eps);
    }
    for (let i = 0; i < W2.length; i++) {
      const g = g2[i];
      m2[i] = b1 * m2[i] + (1 - b1) * g;
      v2[i] = b2 * v2[i] + (1 - b2) * g * g;
      W2[i] = W2[i] * decay - (o.lr * (m2[i] / c1)) / (Math.sqrt(v2[i] / c2) + eps);
    }
    ep++;
  };
  return {
    result,
    advance(k: number) {
      for (let i = 0; i < k && !done; i++) epoch();
      return done;
    },
  };
}

/** Convenience wrapper: train to the end in one call. */
export function trainModAdd(o: TrainOpts, rnd: () => number, gauss: () => number): TrainResult {
  const t = makeTrainer(o, rnd, gauss);
  t.advance(o.epochs + 1);
  return t.result;
}

export type Layer = 'both' | 'K1' | 'K2';

/**
 * Class-averaged empirical NTK over all p² inputs (row-major N×N, input index = a·p + b).
 * `layer` restricts the parameter sum to the first-layer (K1) or second-layer (K2) weights.
 */
export function entkKernel(p: number, n: number, W1: Float64Array, W2: Float64Array, layer: Layer = 'both'): Float64Array {
  const use1 = layer !== 'K2', use2 = layer !== 'K1';
  const N = p * p;
  const s = new Float64Array(n);
  for (let c = 0; c < p; c++) for (let k = 0; k < n; k++) s[k] += W2[c * n + k] * W2[c * n + k];
  const H = new Float64Array(N * n), Hs = new Float64Array(N * n), S = new Float64Array(N * n);
  for (let x = 0; x < N; x++) {
    const a = (x / p) | 0, b = x - a * p;
    for (let k = 0; k < n; k++) {
      const h = W1[a * n + k] + W1[(p + b) * n + k];
      H[x * n + k] = h;
      Hs[x * n + k] = h * s[k] * (4 / p);
      S[x * n + k] = h * h;
    }
  }
  const K = new Float64Array(N * N);
  for (let x = 0; x < N; x++) {
    const a = (x / p) | 0, b = x - a * p, xo = x * n;
    for (let y = x; y < N; y++) {
      const a2 = (y / p) | 0, b2 = y - a2 * p, yo = y * n;
      const share = (a === a2 ? 1 : 0) + (b === b2 ? 1 : 0); // x·x'
      let v = 0;
      if (use2) {
        let k2a = 0, k2b = 0;
        for (let k = 0; k < n; k += 2) {
          k2a += S[xo + k] * S[yo + k];
          k2b += S[xo + k + 1] * S[yo + k + 1];
        }
        v = k2a + k2b;
      }
      if (share && use1) {
        let k1a = 0, k1b = 0;
        for (let k = 0; k < n; k += 2) {
          k1a += Hs[xo + k] * H[yo + k];
          k1b += Hs[xo + k + 1] * H[yo + k + 1];
        }
        v += share * (k1a + k1b);
      }
      K[x * N + y] = v;
      K[y * N + x] = v;
    }
  }
  return K;
}

export interface Eig {
  n: number;
  values: Float64Array; // descending
  vectors: Float64Array; // row r = unit eigenvector r
}

/**
 * Householder reduction of a real symmetric matrix to tridiagonal form (in place; `a` becomes the
 * orthogonal transform; `d` and `e` receive the diagonal and the sub-diagonal).
 */
function tred2(a: Float64Array, n: number, d: Float64Array, e: Float64Array) {
  for (let i = n - 1; i > 0; i--) {
    const l = i - 1, io = i * n;
    let h = 0, scale = 0;
    if (l > 0) {
      for (let k = 0; k <= l; k++) scale += Math.abs(a[io + k]);
      if (scale === 0) e[i] = a[io + l];
      else {
        for (let k = 0; k <= l; k++) {
          a[io + k] /= scale;
          h += a[io + k] * a[io + k];
        }
        let f = a[io + l];
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        a[io + l] = f - g;
        f = 0;
        for (let j = 0; j <= l; j++) {
          const jo = j * n;
          a[jo + i] = a[io + j] / h;
          g = 0;
          for (let k = 0; k <= j; k++) g += a[jo + k] * a[io + k];
          for (let k = j + 1; k <= l; k++) g += a[k * n + j] * a[io + k];
          e[j] = g / h;
          f += e[j] * a[io + j];
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          const jo = j * n;
          f = a[io + j];
          e[j] = g = e[j] - hh * f;
          for (let k = 0; k <= j; k++) a[jo + k] -= f * e[k] + g * a[io + k];
        }
      }
    } else e[i] = a[io + l];
    d[i] = h;
  }
  d[0] = 0;
  e[0] = 0;
  const col = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const l = i - 1, io = i * n;
    if (d[i] !== 0) {
      for (let k = 0; k <= l; k++) col[k] = a[k * n + i];
      for (let j = 0; j <= l; j++) {
        let g = 0;
        for (let k = 0; k <= l; k++) g += a[io + k] * a[k * n + j];
        for (let k = 0; k <= l; k++) a[k * n + j] -= g * col[k];
      }
    }
    d[i] = a[io + i];
    a[io + i] = 1;
    for (let j = 0; j <= l; j++) a[j * n + i] = a[io + j] = 0;
  }
}

/** Implicit-shift QL on a tridiagonal matrix; `zt` holds the accumulated vectors as ROWS. */
function tqli(d: Float64Array, e: Float64Array, n: number, zt: Float64Array) {
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  for (let l = 0; l < n; l++) {
    let iter = 0;
    for (;;) {
      let m = l;
      for (; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= 2.3e-16 * dd) break;
      }
      if (m === l) break;
      if (iter++ > 80) throw new Error('tqli: no convergence');
      let g = (d[l + 1] - d[l]) / (2 * e[l]);
      let r = Math.hypot(g, 1);
      g = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));
      let s = 1, c = 1, pp = 0;
      let i = m - 1;
      for (; i >= l; i--) {
        let f = s * e[i];
        const b = c * e[i];
        e[i + 1] = r = Math.hypot(f, g);
        if (r === 0) {
          d[i + 1] -= pp;
          e[m] = 0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - pp;
        r = (d[i] - g) * s + 2 * c * b;
        d[i + 1] = g + (pp = s * r);
        g = c * r - b;
        const r0 = i * n, r1 = r0 + n;
        for (let k = 0; k < n; k++) {
          f = zt[r1 + k];
          const z0 = zt[r0 + k];
          zt[r1 + k] = s * z0 + c * f;
          zt[r0 + k] = c * z0 - s * f;
        }
      }
      if (r === 0 && i >= l) continue;
      d[l] -= pp;
      e[l] = g;
      e[m] = 0;
    }
  }
}

/**
 * Full eigendecomposition of a symmetric matrix (Householder + implicit QL), eigenvalues
 * descending, eigenvectors as rows. Exact to machine precision for n in the hundreds.
 */
export function eigFull(K: Float64Array, n: number): Eig {
  const a = K.slice();
  const d = new Float64Array(n), e = new Float64Array(n);
  tred2(a, n, d, e);
  // transpose so that vectors are rows (cache-friendly Givens updates)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const t = a[i * n + j];
      a[i * n + j] = a[j * n + i];
      a[j * n + i] = t;
    }
  tqli(d, e, n, a);
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => d[j] - d[i]);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  order.forEach((src, dst) => {
    values[dst] = d[src];
    vectors.set(a.subarray(src * n, src * n + n), dst * n);
  });
  return { n, values, vectors };
}

export type Family = 'a' | 'b' | 'sum' | 'diff';

/** Unit-norm, mean-free Fourier mode over the p² inputs. `phase` 0 = cos, 1 = sin. */
export function fourierMode(p: number, fam: Family, k: number, phase: 0 | 1): Float64Array {
  const N = p * p, f = new Float64Array(N);
  const w = (2 * Math.PI * k) / p, amp = Math.sqrt(2 / N);
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      const arg = w * (fam === 'a' ? a : fam === 'b' ? b : fam === 'sum' ? a + b : a - b);
      f[a * p + b] = amp * (phase ? Math.sin(arg) : Math.cos(arg));
    }
  return f;
}

/** ‖Qᵀ f‖² for the rows [r0, r1) of an eigenvector matrix, and optionally the projection Q Qᵀ f. */
export function projectOnto(eig: Eig, r0: number, r1: number, f: Float64Array, out?: Float64Array): number {
  const n = eig.n, V = eig.vectors;
  if (out) out.fill(0);
  let s = 0;
  for (let r = r0; r < r1; r++) {
    let c = 0;
    const ro = r * n;
    for (let i = 0; i < n; i++) c += V[ro + i] * f[i];
    s += c * c;
    if (out) for (let i = 0; i < n; i++) out[i] += c * V[ro + i];
  }
  return s;
}

/** ‖Qᵀ F‖_F² between the eigen-rows [r0, r1) and a whole Fourier family (2·⌊p/2⌋ orthonormal modes). */
export function familyOverlap(eig: Eig, r0: number, r1: number, p: number, fam: Family): number {
  let s = 0;
  for (let k = 1; k <= (p - 1) >> 1; k++) {
    s += projectOnto(eig, r0, r1, fourierMode(p, fam, k, 0));
    s += projectOnto(eig, r0, r1, fourierMode(p, fam, k, 1));
  }
  return s;
}
