import * as THREE from 'three/webgpu';
import { uniform, vec3 } from 'three/tsl';
import type { RGB } from '../engine/palette';
import { screenWindow } from './window';

/**
 * A flat area of ink: a filled polygon in one color. Like every mark it adds to the frame's ink, so
 * two fills that overlap print the mixture of their inks, as two passes of a press would. The
 * emblem is three of these.
 *
 * `points` is a convex polygon (three or more corners, in order). `uInk` is the density: 0 prints
 * nothing, 1 the nominal color.
 */
export class InkFill {
  readonly object: THREE.Mesh;
  readonly uInk = uniform(1);
  readonly uColor = uniform(new THREE.Vector3(1, 1, 1));
  /** Screen-UV rectangle (x0, y0, x1, y1) the fill is confined to; see kit/window.ts. */
  readonly uWindow: any;

  constructor(points: ArrayLike<number>[], color: RGB, ink = 1) {
    const n = points.length;
    const pos = new Float32Array((n - 2) * 9);
    for (let i = 1, j = 0; i + 1 < n; i++)
      for (const p of [points[0], points[i], points[i + 1]]) { pos[j++] = p[0]; pos[j++] = p[1]; pos[j++] = p[2]; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const win = screenWindow();
    this.uWindow = win.uWindow;
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = vec3(this.uColor as any).mul(this.uInk).mul(win.factor);
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = false;
    m.side = THREE.DoubleSide;
    m.blending = THREE.AdditiveBlending;
    this.object = new THREE.Mesh(g, m);
    this.object.frustumCulled = false;
    this.setColor(color);
    this.uInk.value = ink;
  }

  setColor(c: RGB) {
    (this.uColor.value as THREE.Vector3).set(c[0], c[1], c[2]);
  }
}
