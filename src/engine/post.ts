import * as THREE from 'three/webgpu';
import { pass, uniform, vec2, vec3, vec4, float, uv, mix, smoothstep, time, hash, screenCoordinate, renderOutput, exp, step } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { THEME, PAPER_LIN } from './palette';

/**
 * Output stage.
 *
 * paper theme: the scene pass holds accumulated ink (absorbance), so the picture is
 *   paper · exp(−ink). No bloom, no vignette. Chapter fades go to blank paper.
 * dark theme: the scene pass holds emitted light; a faint halo, a neutral tone map and a light
 *   vignette are added. Chapter fades go to the background.
 * Both get one gray level of dither so flat areas do not band.
 */
export function buildPost(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera) {
  const U = {
    bloom: uniform(0.2),
    vignette: uniform(0.28),
    dither: uniform(THEME === 'paper' ? 0.006 : 0.012),
    fade: uniform(1.0),
    view: uniform(new THREE.Vector3(1, 0, 0)), // scale, x/width, y/height (top-left origin)
    viewBounds: uniform(new THREE.Vector4(0, 0, 1, 1)),
    clipFigure: uniform(0), // active after either 2-D magnification or a local object transform
  };
  const scenePass = pass(scene, camera);
  const sceneTex: any = scenePass.getTextureNode('output');
  // Magnify the finished authored picture. Camera-authored flat panels and their clipping windows
  // follow the same map as world geometry, rather than cancelling a camera zoom when they redraw.
  // RenderPipeline UVs use the same top-left origin as frame pixels (including on WebGPU).
  const at: any = uv(), view = U.view, bounds = U.viewBounds;
  const source = vec2(at.x.sub(view.y).div(view.x), at.y.sub(view.z).div(view.x));
  const inside = step(bounds.x, at.x).mul(step(at.x, bounds.z)).mul(step(bounds.y, at.y)).mul(step(at.y, bounds.w));
  const picture: any = U.clipFigure.greaterThan(0.5).select(sceneTex.sample(source).mul(inside), sceneTex);
  const px: any = screenCoordinate;
  const seed = px.x.floor().add(px.y.floor().mul(3181.0)).add(time.mul(60.0).floor().mul(7919.0));
  const dither = hash(seed).sub(0.5).mul(U.dither);

  let out: any;
  let glow: any = null;
  if (THEME === 'paper') {
    const ink = picture.rgb.max(0.0).mul(U.fade);
    const lit = vec3(PAPER_LIN[0], PAPER_LIN[1], PAPER_LIN[2]).mul(exp(ink.negate()));
    out = (renderOutput(vec4(lit, 1.0)) as any).rgb.add(dither);
  } else {
    glow = bloom(picture, 1.0, 0.45, 0.4);
    const hdr = picture.rgb.add(glow.rgb.mul(U.bloom)).mul(U.fade);
    const d = (uv() as any).sub(0.5);
    const vig = mix(float(1.0), smoothstep(1.0, 0.35, d.length()), U.vignette);
    out = (renderOutput(vec4(hdr, 1.0)) as any).rgb.mul(vig).add(dither);
  }

  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false;
  pipeline.outputNode = vec4(out, 1.0);
  return { pipeline, U, glow };
}

export type Post = ReturnType<typeof buildPost>;
