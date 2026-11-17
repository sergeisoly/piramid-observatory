import * as THREE from 'three/webgpu';
import { attribute, uniform } from 'three/tsl';
import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { screenWindow } from './window';

/**
 * Stroke widths are given in design pixels (a 1600 × 900 frame). The engine reports how many
 * device pixels one design pixel covers, and every live stroke material follows it, so strokes
 * keep their weight relative to the figure in any window.
 */
const strokes = new Set<{ m: any; width: number }>();
let strokeScale = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
export function setStrokeScale(devicePxPerDesignPx: number) {
  strokeScale = devicePxPerDesignPx;
  for (const s of strokes) s.m.linewidth = Math.max(1, s.width * strokeScale);
}

export interface LineOpts {
  /** Stroke width in design pixels. 1 (default) draws hairlines; anything larger draws screen-space strokes. */
  width?: number;
}

/**
 * A batch of additive line segments with per-vertex color. Write with `seg()` / `push()` (or
 * straight into `positions` / `colors`), then `commit()`. Brightness is carried by the color
 * magnitude. Use the default hairline for axes, ticks and grids, and `width: 2` for data curves.
 */
export class LineField {
  readonly max: number;
  readonly positions: Float32Array;
  readonly colors: Float32Array;
  readonly object: THREE.Object3D;
  readonly uIntensity = uniform(1);
  /** Screen-UV rectangle (x0, y0, x1, y1) the segments are confined to; see kit/window.ts. */
  readonly uWindow: any;
  readonly uFeather: any;
  private pA?: THREE.BufferAttribute;
  private cA?: THREE.BufferAttribute;
  private geo?: THREE.BufferGeometry;
  private fat?: LineSegmentsGeometry;
  private n = 0;

  constructor(maxSegments: number, opts: LineOpts = {}) {
    this.max = maxSegments;
    this.positions = new Float32Array(maxSegments * 6);
    this.colors = new Float32Array(maxSegments * 6);
    const width = opts.width ?? 1;
    const win = screenWindow();
    this.uWindow = win.uWindow;
    this.uFeather = win.uFeather;
    if (width > 1.01) {
      const g = (this.fat = new LineSegmentsGeometry());
      g.setPositions(this.positions);
      g.setColors(this.colors);
      (g.attributes.instanceStart as any).data.setUsage(THREE.DynamicDrawUsage);
      (g.attributes.instanceColorStart as any).data.setUsage(THREE.DynamicDrawUsage);
      const m = new THREE.Line2NodeMaterial({ vertexColors: true, worldUnits: false, dashed: false } as any);
      m.linewidth = Math.max(1, width * strokeScale);
      const entry = { m, width };
      strokes.add(entry);
      m.addEventListener('dispose', () => strokes.delete(entry));
      m.colorNode = (this.uIntensity as any).mul(win.factor); // multiplies the per-vertex colors
      m.blending = THREE.AdditiveBlending;
      m.depthWrite = false;
      m.depthTest = false;
      m.alphaToCoverage = false;
      const obj = new LineSegments2(g, m);
      obj.frustumCulled = false;
      obj.visible = false;
      g.instanceCount = 0;
      this.object = obj;
    } else {
      this.pA = new THREE.BufferAttribute(this.positions, 3);
      this.cA = new THREE.BufferAttribute(this.colors, 3);
      this.pA.setUsage(THREE.DynamicDrawUsage);
      this.cA.setUsage(THREE.DynamicDrawUsage);
      this.geo = new THREE.BufferGeometry();
      this.geo.setAttribute('position', this.pA);
      this.geo.setAttribute('color', this.cA);
      const m = new THREE.LineBasicNodeMaterial();
      m.colorNode = (attribute('color', 'vec3') as any).mul(this.uIntensity).mul(win.factor);
      m.transparent = true;
      m.depthWrite = false;
      m.blending = THREE.AdditiveBlending;
      const obj = new THREE.LineSegments(this.geo, m);
      obj.frustumCulled = false;
      obj.visible = false;
      this.geo.setDrawRange(0, 0);
      this.object = obj;
    }
  }

  /** Number of segments currently drawn. */
  get count() {
    return this.n;
  }
  set count(n: number) {
    this.n = Math.max(0, Math.min(this.max, Math.floor(n)));
    if (this.fat) this.fat.instanceCount = this.n;
    else this.geo!.setDrawRange(0, this.n * 2);
    this.object.visible = this.n > 0; // WebGPU warns on zero-vertex draws
  }

  /** Write segment i from a to b. `cb` defaults to `ca`; `k` scales brightness. */
  seg(i: number, a: ArrayLike<number>, b: ArrayLike<number>, ca: ArrayLike<number>, k = 1, cb: ArrayLike<number> = ca, kb = k) {
    const p = this.positions, c = this.colors, j = i * 6;
    p[j] = a[0]; p[j + 1] = a[1]; p[j + 2] = a[2];
    p[j + 3] = b[0]; p[j + 4] = b[1]; p[j + 5] = b[2];
    c[j] = ca[0] * k; c[j + 1] = ca[1] * k; c[j + 2] = ca[2] * k;
    c[j + 3] = cb[0] * kb; c[j + 4] = cb[1] * kb; c[j + 5] = cb[2] * kb;
  }

  /** Append a segment and return its index. */
  push(a: ArrayLike<number>, b: ArrayLike<number>, ca: ArrayLike<number>, k = 1, cb: ArrayLike<number> = ca, kb = k) {
    const i = this.n;
    if (i >= this.max) return i;
    this.seg(i, a, b, ca, k, cb, kb);
    this.count = i + 1;
    return i;
  }

  /** Append a polyline through pts (array of [x,y,z]). */
  polyline(pts: ArrayLike<number>[], c: ArrayLike<number>, k = 1, closed = false) {
    for (let i = 0; i + 1 < pts.length; i++) this.push(pts[i], pts[i + 1], c, k);
    if (closed && pts.length > 2) this.push(pts[pts.length - 1], pts[0], c, k);
  }

  /** Append a circle of radius r around center c in the plane spanned by unit vectors u, v. */
  circle(c: ArrayLike<number>, r: number, u: ArrayLike<number>, v: ArrayLike<number>, col: ArrayLike<number>, k = 1, segs = 96, dash = 1) {
    for (let i = 0; i < segs; i++) {
      if (dash > 1 && i % dash !== 0) continue;
      const a0 = (i / segs) * Math.PI * 2, a1 = ((i + 1) / segs) * Math.PI * 2;
      const p = (a: number) => [
        c[0] + r * (Math.cos(a) * u[0] + Math.sin(a) * v[0]),
        c[1] + r * (Math.cos(a) * u[1] + Math.sin(a) * v[1]),
        c[2] + r * (Math.cos(a) * u[2] + Math.sin(a) * v[2]),
      ];
      this.push(p(a0), p(a1), col, k);
    }
  }

  clear() {
    this.count = 0;
  }
  commit() {
    if (this.fat) {
      (this.fat.attributes.instanceStart as any).data.needsUpdate = true;
      (this.fat.attributes.instanceColorStart as any).data.needsUpdate = true;
    } else {
      this.pA!.needsUpdate = true;
      this.cA!.needsUpdate = true;
    }
  }
}
