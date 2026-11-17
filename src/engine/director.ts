import * as THREE from 'three/webgpu';
import type { Shot } from './types';

export interface CameraState {
  pos: THREE.Vector3;
  look: THREE.Vector3;
  fov: number;
}

const ch = (s: Shot, k: number) => (k < 3 ? s.pos[k] : k < 6 ? s.look[k - 3] : s.fov ?? 40);

/**
 * Sample the camera path at local time t. All seven channels (position, look-at, fov) run
 * through one monotone cubic Hermite spline: velocity is continuous through keyframes that
 * keep moving, and a repeated keyframe is a true hold.
 */
export function sampleShots(shots: Shot[], t: number, out: CameraState) {
  const n = shots.length;
  if (n === 0) return out;
  if (n === 1 || t <= shots[0].t) return write(out, (k) => ch(shots[0], k));
  if (t >= shots[n - 1].t) return write(out, (k) => ch(shots[n - 1], k));
  let i = 0;
  while (i < n - 2 && t >= shots[i + 1].t) i++;
  const a = shots[i], b = shots[i + 1];
  const dt = b.t - a.t;
  const u = (t - a.t) / dt, u2 = u * u, u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
  const prev = shots[Math.max(0, i - 1)], next = shots[Math.min(n - 1, i + 2)];
  return write(out, (k) => {
    const p0 = ch(a, k), p1 = ch(b, k);
    // Monotone (PCHIP-style) tangents: zero wherever the path holds or turns, so the camera
    // never overshoots a keyframe or dips before a move that follows a hold.
    const d1 = (p1 - p0) / dt;
    const d0 = i === 0 ? 0 : (p0 - ch(prev, k)) / (a.t - prev.t);
    const d2 = i + 2 > n - 1 ? 0 : (ch(next, k) - p1) / (next.t - b.t);
    const m0 = d0 * d1 <= 0 ? 0 : (2 * d0 * d1) / (d0 + d1);
    const m1 = d1 * d2 <= 0 ? 0 : (2 * d1 * d2) / (d1 + d2);
    return h00 * p0 + h10 * dt * m0 + h01 * p1 + h11 * dt * m1;
  });
}

function write(out: CameraState, f: (k: number) => number) {
  out.pos.set(f(0), f(1), f(2));
  out.look.set(f(3), f(4), f(5));
  out.fov = f(6);
  return out;
}

const _off = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);

/** Chapters author shots for a ~40° lens; the film is shot on a longer one for flatter, more diagram-like perspective. */
const LENS = 0.6;

const WIDE = 16 / 9;

/** Apply the authored shot and the film's lens without pointer-driven camera offsets. */
export function applyCamera(cam: THREE.PerspectiveCamera, s: CameraState) {
  // Narrow the field of view and back the camera off so the subject at the look target keeps its size.
  const fov = s.fov * LENS;
  const back = Math.tan((s.fov * Math.PI) / 360) / Math.tan((fov * Math.PI) / 360);
  _off.copy(s.pos).sub(s.look).multiplyScalar(back);
  const dist = _off.length();
  // Preserve the authored camera's existing limit near a vertical view (89° of elevation).
  const elev = Math.asin(Math.max(-1, Math.min(1, _off.y / Math.max(dist, 1e-6))));
  const correction = Math.max(-1.553 - elev, Math.min(1.553 - elev, 0));
  _right.crossVectors(_up, _off).normalize();
  _off.applyAxisAngle(_right, -correction);
  cam.position.copy(s.look).add(_off);
  cam.up.set(0, 1, 0);
  cam.lookAt(s.look);
  // Figures are composed for 16:9. In a narrower window, widen the vertical field of view so the
  // horizontal one stays what it is at 16:9: the figure keeps its full width and gains margin above and below.
  const fit = cam.aspect < WIDE ? (2 * Math.atan(Math.tan((fov * Math.PI) / 360) * (WIDE / cam.aspect)) * 180) / Math.PI : fov;
  if (Math.abs(cam.fov - fit) > 1e-3) {
    cam.fov = fit;
    cam.updateProjectionMatrix();
  }
  return dist;
}
