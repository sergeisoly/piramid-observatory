import * as THREE from 'three/webgpu';
import { texture, uniform, vec3 } from 'three/tsl';
import type { FigureSource } from './types';
import { PAPER_LIN, THEME } from './palette';

/** Resolve bundled assets under Vite's base path, including project-site deployments. */
export function figureAssetURL(asset: string) {
  if (/^(?:[a-z]+:|\/\/)/i.test(asset) || asset.split('/').includes('..')) throw new Error('Source figures must use bundled local assets');
  return new URL(asset.replace(/^\.?\//, ''), new URL(import.meta.env.BASE_URL, document.baseURI)).href;
}

/** Load one original figure during chapter build and fit it into a world-space rectangle. */
export async function loadFigureImage(source: FigureSource, fit: { width: number; height?: number }) {
  if (!(fit.width > 0) || (fit.height !== undefined && !(fit.height > 0))) throw new Error('A source figure needs a positive fit rectangle');
  const image = await new THREE.TextureLoader().loadAsync(figureAssetURL(source.asset));
  image.colorSpace = THREE.SRGBColorSpace;
  const aspect = image.image.width / image.image.height;
  const width = Math.min(fit.width, fit.height === undefined ? fit.width : fit.height * aspect);
  const height = width / aspect;
  const sampled: any = texture(image);
  const opacity = uniform(1);
  const material = new THREE.MeshBasicNodeMaterial();
  // Paper scenes accumulate absorbance. Inverting that map keeps the original plot colors;
  // its white background becomes the same paper as the rest of the film.
  material.colorNode = THEME === 'paper' ? vec3(...PAPER_LIN).div(sampled.rgb.max(0.00001)).log().max(0) : sampled.rgb;
  material.opacityNode = sampled.a.mul(opacity);
  material.transparent = true;
  material.blending = THEME === 'paper' ? THREE.AdditiveBlending : THREE.NormalBlending;
  material.depthTest = false;
  material.depthWrite = false;
  material.side = THREE.DoubleSide;
  material.addEventListener('dispose', () => image.dispose());
  const geometry = new THREE.PlaneGeometry(width, height);
  const object = new THREE.Mesh(geometry, material);
  object.name = `source-figure:${source.asset}`;
  object.frustumCulled = false;
  return {
    object, width, height,
    setOpacity(alpha: number) { opacity.value = Math.max(0, Math.min(1, alpha)); object.visible = alpha > 0.001; },
    dispose() { geometry.dispose(); material.dispose(); },
  };
}
