import * as THREE from 'three/webgpu';
import {
  float, vec3, uniform, instanceIndex, instancedBufferAttribute, positionLocal, normalLocal, mix, uint,
} from 'three/tsl';
import { C, THEME, PAPER_LIN, type RGB } from '../engine/palette';

/** Relative luminance of paper printed with ink `a` at density k (the output stage's arithmetic). */
const printedY = (a: RGB, k: number) => 0.2126 * PAPER_LIN[0] * Math.exp(-k * a[0]) + 0.7152 * PAPER_LIN[1] * Math.exp(-k * a[1]) + 0.0722 * PAPER_LIN[2] * Math.exp(-k * a[2]);
/**
 * Two inks for the two signs of a matrix, matched in lightness: the darker ink is laid on more thinly
 * until both print equally dark at the same |value|. Without it one sign would look heavier than the
 * other. Thinning the darker ink (the key ink, when a team ink is the other sign) keeps both hues true;
 * thickening the lighter one would print it as a brown or a near-black.
 */
function matched(neg: RGB, pos: RGB): [RGB, RGB] {
  if (THEME !== 'paper' || neg === pos) return [neg, pos];
  const K = 1.0; // a mid density of the tiles (their ink runs from 0 to 1.8, see `lum` below)
  const yn = printedY(neg, K), yp = printedY(pos, K);
  const [dark, target] = yn < yp ? [neg, yp] : [pos, yn];
  let lo = 0.02, hi = 1;
  for (let i = 0; i < 30; i++) { const g = (lo + hi) / 2; if (printedY(dark, K * g) < target) hi = g; else lo = g; }
  const g = (lo + hi) / 2, thin = (c: RGB): RGB => [c[0] * g, c[1] * g, c[2] * g];
  return yn < yp ? [thin(neg), pos] : [neg, thin(pos)];
}

interface MatrixOpts {
  rows: number;
  cols: number;
  cell?: number; // world size of one cell
  gap?: number; // 0..1 fraction of the cell left empty
  height?: number; // world height of |value| = 1 (0 = flat tiles)
  neg?: RGB; // color for −1
  pos?: RGB; // color for +1
}

/**
 * A matrix as a field of glowing tiles/bars lying in the XZ plane, centered on the origin.
 * Row r runs along +Z, column c along +X. Write `values` (row-major, nominally in [−1, 1]),
 * then `commit()`. Color diverges neg → dark → pos; bar height follows |value|.
 */
export class MatrixField {
  readonly values: Float32Array;
  readonly object: THREE.Mesh;
  private readonly uGain = uniform(1);
  readonly uOpacity = uniform(1);
  private vA: THREE.InstancedBufferAttribute;

  constructor(o: MatrixOpts) {
    const R = o.rows, Cn = o.cols;
    const cell = o.cell ?? 0.1;
    const gap = o.gap ?? 0.12;
    this.values = new Float32Array(R * Cn);
    this.vA = new THREE.InstancedBufferAttribute(this.values, 1);
    this.vA.setUsage(THREE.DynamicDrawUsage);
    const uHeight = uniform(o.height ?? 0);
    const [neg, pos] = matched(o.neg ?? C.violet, o.pos ?? C.amber);

    const v: any = (instancedBufferAttribute(this.vA) as any).mul(this.uGain);
    const a = v.abs().clamp(0, 1.5);
    const col = uint(instanceIndex).mod(uint(Cn)).toFloat();
    const row = uint(instanceIndex).div(uint(Cn)).toFloat();
    const cx = col.sub((Cn - 1) / 2).mul(cell);
    const cz = row.sub((R - 1) / 2).mul(cell);
    const h = a.mul(uHeight).add(cell * 0.04);
    const local: any = positionLocal;
    const p = vec3(
      local.x.mul(cell * (1 - gap)).add(cx),
      local.y.add(0.5).mul(h),
      local.z.mul(cell * (1 - gap)).add(cz),
    );
    const tint = mix(vec3(neg[0], neg[1], neg[2]), vec3(pos[0], pos[1], pos[2]), v.greaterThan(0).select(float(1), float(0)));
    const top = (normalLocal as any).y.max(0).mul(0.6).add(0.4);
    const lum = a.mul(a).mul(1.8).add(0.015);

    const m = new THREE.MeshBasicNodeMaterial();
    m.positionNode = p;
    m.colorNode = tint.mul(lum).mul(top).mul(this.uOpacity);
    m.transparent = true;
    m.depthWrite = false;
    m.blending = THREE.AdditiveBlending;
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), m, R * Cn);
    mesh.frustumCulled = false;
    this.object = mesh;
  }
  commit() {
    this.vA.needsUpdate = true;
  }
}
