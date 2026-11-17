import type { ChapterFactory } from '../engine/types';

/**
 * The film, in order. Every scene lives in ./<id>.ts and default-exports a ChapterFactory.
 *
 * Each scene uses a literal dynamic import so chapters can load on demand. `?only=<id>` selects
 * a single chapter at runtime.
 */
const SCENES = {
  prologue: () => import('./prologue'),
  percolation: () => import('./percolation'),
  generality: () => import('./generality'),
  meanfield: () => import('./meanfield'),
  frustration: () => import('./frustration'),
  entk: () => import('./entk'),
  elicitation: () => import('./elicitation'),
  parity: () => import('./parity'),
  scales: () => import('./scales'),
  finale: () => import('./finale'),
} satisfies Record<string, () => Promise<{ default: ChapterFactory }>>;

export type SceneId = keyof typeof SCENES;
export const ORDER = Object.keys(SCENES) as SceneId[];

export function manifest(only?: string[]) {
  return ORDER.filter((id) => !only || only.includes(id)).map((id) => ({
    id,
    load: () => SCENES[id]().then((m) => m.default),
  }));
}
