import * as THREE from 'three/webgpu';
import { vec3, uniform, instancedBufferAttribute, positionLocal } from 'three/tsl';

/**
 * Flat rectangles in the XY plane (the cells of the sign matrix), one instance each, with
 * per-instance center, width/height and color. Local helper of the `frustration` chapter;
 * it follows the same instanced-attribute pattern as the kit's MatrixField.
 */
export class TileField {
  readonly count: number;
  readonly positions: Float32Array; // x, y, z of each center
  readonly sizes: Float32Array; // width, height
  readonly colors: Float32Array; // r, g, b (palette entry × amount)
  readonly object: THREE.InstancedMesh;
  readonly uOpacity = uniform(1);
  private pA: THREE.InstancedBufferAttribute;
  private sA: THREE.InstancedBufferAttribute;
  private cA: THREE.InstancedBufferAttribute;

  constructor(count: number) {
    this.count = count;
    this.positions = new Float32Array(count * 3);
    this.sizes = new Float32Array(count * 2);
    this.colors = new Float32Array(count * 3);
    this.pA = new THREE.InstancedBufferAttribute(this.positions, 3);
    this.sA = new THREE.InstancedBufferAttribute(this.sizes, 2);
    this.cA = new THREE.InstancedBufferAttribute(this.colors, 3);
    for (const a of [this.pA, this.sA, this.cA]) a.setUsage(THREE.DynamicDrawUsage);
    const p: any = instancedBufferAttribute(this.pA), s: any = instancedBufferAttribute(this.sA), c: any = instancedBufferAttribute(this.cA);
    const local: any = positionLocal;
    const m = new THREE.MeshBasicNodeMaterial();
    m.positionNode = vec3(local.x.mul(s.x).add(p.x), local.y.mul(s.y).add(p.y), p.z);
    m.colorNode = c.mul(this.uOpacity);
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = false;
    m.side = THREE.DoubleSide;
    m.blending = THREE.AdditiveBlending;
    const mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), m, count);
    mesh.frustumCulled = false;
    this.object = mesh;
  }

  set(i: number, x: number, y: number, w: number, h: number, c: ArrayLike<number>, k = 1, z = 0) {
    const p = this.positions, s = this.sizes, q = this.colors;
    p[i * 3] = x; p[i * 3 + 1] = y; p[i * 3 + 2] = z;
    s[i * 2] = w; s[i * 2 + 1] = h;
    q[i * 3] = c[0] * k; q[i * 3 + 1] = c[1] * k; q[i * 3 + 2] = c[2] * k;
  }
  hide(i: number) {
    this.sizes[i * 2] = 0;
    this.sizes[i * 2 + 1] = 0;
  }
  commit() {
    this.pA.needsUpdate = true;
    this.sA.needsUpdate = true;
    this.cA.needsUpdate = true;
  }
  /** Draw only the first n tiles. */
  set drawCount(n: number) {
    const k = Math.max(0, Math.min(this.count, Math.floor(n)));
    this.object.count = k;
    this.object.visible = k > 0;
  }
}
