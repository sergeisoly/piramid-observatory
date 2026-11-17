import * as THREE from 'three/webgpu';
import {
  float, vec3, vec4, uv, uniform, instanceIndex, instancedBufferAttribute,
  modelViewMatrix, hash, time, sin, mix, smoothstep, exp,
} from 'three/tsl';
import { screenWindow } from './window';

/**
 * Lens state shared by every point sprite. The aperture is 0 (everything in focus): data marks
 * should be sharp at any depth. The uniforms remain so a look with depth of field is one number away.
 */
export const FX = {
  focus: uniform(12.0), // world-space distance of the focal plane
  aperture: uniform(0.0), // blur radius gained per world unit away from the focal plane
  shimmer: uniform(0.0), // 0 = steady marks; 1 = the per-point twinkle scenes may request
};

interface GlowOpts {
  falloff?: number; // gaussian sharpness of the core (higher = tighter)
  twinkle?: number; // 0..1 shimmer depth
  bokeh?: number; // multiplier on the global aperture (0 = always sharp)
  opacity?: number;
}

/**
 * Shared look for every point: a crisp disc (about half the sprite's width) with a thin
 * anti-aliased edge and a faint soft skirt. Reads as a plot marker, not a light source.
 */
function glowSprite(center: any, color: any, size: any, o: GlowOpts = {}) {
  const uOpacity = uniform(o.opacity ?? 1);
  const uBokeh = uniform(o.bokeh ?? 1);
  const seed = hash(instanceIndex);

  const viewZ = modelViewMatrix.mul(vec4(center, 1.0)).z.negate();
  const blur = viewZ.sub(FX.focus).abs().mul(FX.aperture).mul(uBokeh);
  const scale = size.add(blur);
  const sharp = size.div(scale.max(1e-5));
  const energy = sharp.pow(1.6).toVarying();
  const soft = sharp.oneMinus().toVarying();

  const r = uv().sub(0.5).length().mul(2.0);
  const marker = smoothstep(0.58, 0.46, r).mul(0.9).add(exp(r.mul(r).mul(-7.0)).mul(smoothstep(1.0, 0.6, r)).mul(0.22));
  const disc = smoothstep(1.0, 0.88, r).mul(0.55);
  let alpha: any = mix(marker, disc, smoothstep(0.3, 0.85, soft)).mul(energy).mul(uOpacity);
  if (o.twinkle) {
    const tw = FX.shimmer.mul(o.twinkle * 0.5);
    alpha = alpha.mul(sin(time.mul(2.4).add(seed.mul(63.0))).mul(tw).add(float(1).sub(tw)));
  }

  const { uWindow, uFeather, factor } = screenWindow(); // 1 everywhere unless a scene sets uWindow
  alpha = alpha.mul(factor);

  const material = new THREE.SpriteNodeMaterial();
  material.positionNode = center;
  material.scaleNode = scale;
  material.colorNode = color;
  material.opacityNode = alpha;
  material.transparent = true;
  material.depthWrite = false;
  material.blending = THREE.AdditiveBlending;
  return { material, uOpacity, uWindow, uFeather };
}

interface PointOpts extends GlowOpts {
  count: number;
  size?: number; // world-unit diameter multiplier
  drift?: number; // idle wander amplitude (world units), cosmetic
  driftSpeed?: number;
}

/**
 * CPU-authored glowing points. Write into `positions` / `colors` / `sizes`, then `commit()`.
 * Good up to ~60k points updated every frame.
 */
export class PointField {
  readonly count: number;
  readonly positions: Float32Array;
  readonly colors: Float32Array;
  readonly sizes: Float32Array;
  readonly object: THREE.Sprite;
  readonly uOpacity: any;
  /** Screen-UV rectangle (x0, y0, x1, y1) the points are confined to; see kit/window.ts. */
  readonly uWindow: any;
  readonly uFeather: any;
  private pA: THREE.InstancedBufferAttribute;
  private cA: THREE.InstancedBufferAttribute;
  private sA: THREE.InstancedBufferAttribute;

  constructor(o: PointOpts) {
    const n = (this.count = o.count);
    this.positions = new Float32Array(n * 3);
    this.colors = new Float32Array(n * 3).fill(1);
    this.sizes = new Float32Array(n).fill(1);
    this.pA = new THREE.InstancedBufferAttribute(this.positions, 3);
    this.cA = new THREE.InstancedBufferAttribute(this.colors, 3);
    this.sA = new THREE.InstancedBufferAttribute(this.sizes, 1);
    for (const a of [this.pA, this.cA, this.sA]) a.setUsage(THREE.DynamicDrawUsage);

    const pos: any = instancedBufferAttribute(this.pA);
    const col: any = instancedBufferAttribute(this.cA);
    const sz: any = instancedBufferAttribute(this.sA);
    const uSize = uniform(o.size ?? 0.08);

    let center = pos;
    if (o.drift) {
      const ph = hash(instanceIndex).mul(6.2831853);
      const tt = time.mul(o.driftSpeed ?? 0.15);
      center = pos.add(
        vec3(sin(tt.add(ph)), sin(tt.mul(1.31).add(ph.mul(1.7))), sin(tt.mul(0.83).add(ph.mul(2.3)))).mul(o.drift),
      );
    }
    const g = glowSprite(center, col, sz.mul(uSize), o);
    this.uOpacity = g.uOpacity;
    this.uWindow = g.uWindow;
    this.uFeather = g.uFeather;
    this.object = new THREE.Sprite(g.material);
    (this.object as any).count = n;
    this.object.frustumCulled = false;
  }

  setPos(i: number, x: number, y: number, z: number) {
    const p = this.positions, j = i * 3;
    p[j] = x; p[j + 1] = y; p[j + 2] = z;
  }
  setColor(i: number, c: ArrayLike<number>, k = 1) {
    const p = this.colors, j = i * 3;
    p[j] = c[0] * k; p[j + 1] = c[1] * k; p[j + 2] = c[2] * k;
  }
  /** Upload changed buffers to the GPU. */
  commit(pos = true, col = true, size = true) {
    if (pos) this.pA.needsUpdate = true;
    if (col) this.cA.needsUpdate = true;
    if (size) this.sA.needsUpdate = true;
  }
  /** Draw only the first n points. */
  set drawCount(n: number) {
    (this.object as any).count = Math.max(0, Math.min(this.count, Math.floor(n)));
  }
}
