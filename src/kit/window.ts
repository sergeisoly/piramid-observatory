import * as THREE from 'three/webgpu';
import { uniform, screenUV, min, smoothstep } from 'three/tsl';

/**
 * Optional screen-space window for a batch of marks. `factor` is 1 inside the rectangle `uWindow`
 * (screen UV: x0, y0, x1, y1, origin top-left) and falls to 0 over `uFeather` at its edges. The
 * default rectangle is larger than the screen, so nothing is clipped until a scene sets it.
 * A scene uses this to keep 3-D content inside its own panel of a multi-panel figure.
 */
export function screenWindow() {
  const uWindow = uniform(new THREE.Vector4(-1, -1, 2, 2));
  const uFeather = uniform(0.02);
  const s: any = screenUV;
  const wx = min(s.x.sub(uWindow.x), uWindow.z.sub(s.x));
  const wy = min(s.y.sub(uWindow.y), uWindow.w.sub(s.y));
  const factor: any = smoothstep(0.0, uFeather, wx).mul(smoothstep(0.0, uFeather.mul(1.78), wy));
  return { uWindow, uFeather, factor };
}
