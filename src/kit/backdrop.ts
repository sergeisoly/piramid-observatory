import * as THREE from 'three/webgpu';
import { vec3, positionLocal } from 'three/tsl';
import { THEME } from '../engine/palette';

/**
 * The backdrop. On paper there is nothing to draw (no ink = blank paper). In the dark theme it
 * is a plain deep-ink studio wall with a slight lift toward the horizon. No stars, dust or haze.
 */
export class Backdrop {
  readonly object = new THREE.Group();

  constructor() {
    if (THEME === 'paper') return;
    const dir: any = (positionLocal as any).normalize();
    const lift = dir.y.abs().oneMinus().pow(3.0);
    const col = vec3(0.004, 0.0085, 0.0125).add(vec3(0.002, 0.0035, 0.005).mul(lift));
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = col;
    m.side = THREE.BackSide;
    m.depthWrite = false;
    m.depthTest = false;
    const sky = new THREE.Mesh(new THREE.SphereGeometry(900, 32, 16), m);
    sky.renderOrder = -10;
    sky.frustumCulled = false;
    this.object.add(sky);
  }
}
