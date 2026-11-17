import PROLOGUE from './framing/prologue.json';
import PERCOLATION from './framing/percolation.json';
import GENERALITY from './framing/generality.json';
import MEANFIELD from './framing/meanfield.json';
import FRUSTRATION from './framing/frustration.json';
import ENTK from './framing/entk.json';
import ELICITATION from './framing/elicitation.json';
import PARITY from './framing/parity.json';
import SCALES from './framing/scales.json';
import FINALE from './framing/finale.json';
import { smooth } from '../lib/math';

/**
 * Re-framing.
 *
 * Every scene composes its figure for a fixed rectangle of the 1600 × 900 frame (the "base frame":
 * the scene's own camera, with the optical center 4.5 % above the middle). The layout around the
 * figure depends on the layout. On desktop, an open Notes page occupies part of the frame. The
 * engine fits the figure into the available area with a camera view offset. The base-frame bounds
 * come from tools/boxes.mjs, which writes one JSON file per chapter with one box per caption beat.
 */
export type Box = [number, number, number, number]; // x0, y0, x1, y1 in base-frame pixels

/** A chapter with no measured box is assumed to fill the area the scenes were first composed for. */
const DEFAULT_BOX: Box = [40, 190, 1160, 650];

const TABLE: Record<string, number[][]> = {
  prologue: PROLOGUE,
  percolation: PERCOLATION,
  generality: GENERALITY,
  meanfield: MEANFIELD,
  frustration: FRUSTRATION,
  entk: ENTK,
  elicitation: ELICITATION,
  parity: PARITY,
  scales: SCALES,
  finale: FINALE,
};

/**
 * Boxes set by hand, where the measured one does not give the best picture: [beat index, x0, y0, x1, y1].
 * They replace the measured box of that beat and survive a new run of tools/boxes.mjs.
 *  - frustration: one layout for the whole chapter, so one box for every slide.
 *  - scales: the figure keeps its place on all four slides, so that the picture does not re-zoom when the text
 *    changes.
 */
const BY_HAND: Record<string, number[][]> = {
  // Brill’s allocation schematic and source panels share a fixed camera during their crossfades.
  generality: [0, 1, 2].map((j) => [j, 230, 160, 1370, 750]),
  scales: [0, 1, 2, 3].map((j) => [j, 295, 176, 1411, 696]),
  // frustration keeps one layout for all four slides (the parts never move): every slide gets the box of the whole layout
  frustration: [0, 1, 2, 3].map((j) => [j, 28, 150, 1533, 687]),
};
for (const id of Object.keys(BY_HAND))
  for (const [j, ...box] of BY_HAND[id]) if (TABLE[id]?.[j]) TABLE[id][j] = [TABLE[id][j][0], ...box];

/** Frame-transition endpoints in seconds relative to the incoming beat's start. */
const DEFAULT_TRANSITION: [number, number] = [-0.5, 1.7];
const TRANSITIONS: Record<string, Record<number, [number, number]>> = {
  // The network moves left at 5.6–6.4, before the usual frame expansion would finish at 7.3.
  // Make room first, including a little lead for the camera's smoothing, so its labels stay in view.
  prologue: { 1: [-0.7, -0.1] },
};

/** The figure's box in the base frame at chapter time t (written into `out`). */
export function boxAt(id: string, t: number, out: Box): Box {
  const keys = TABLE[id];
  if (!keys?.length) { out[0] = DEFAULT_BOX[0]; out[1] = DEFAULT_BOX[1]; out[2] = DEFAULT_BOX[2]; out[3] = DEFAULT_BOX[3]; return out; }
  let j = 0;
  while (j + 1 < keys.length && t >= keys[j + 1][0] + (TRANSITIONS[id]?.[j + 1] ?? DEFAULT_TRANSITION)[0]) j++;
  const b = keys[j];
  // inside the window around this beat's start: blend from the previous beat's box
  const a = j > 0 ? keys[j - 1] : b;
  const [from, to] = TRANSITIONS[id]?.[j] ?? DEFAULT_TRANSITION;
  const w = j > 0 ? smooth(b[0] + from, b[0] + to, t) : 1;
  for (let k = 0; k < 4; k++) out[k] = a[k + 1] + (b[k + 1] - a[k + 1]) * w;
  return out;
}

export interface View {
  /** Magnification of the base picture. */
  s: number;
  /** The base-frame point … */
  px: number;
  py: number;
  /** … that lands on this point of the frame. */
  qx: number;
  qy: number;
}

/**
 * Room for labels that grow. In a window smaller than the 1600 × 900 frame a figure label keeps a smallest size on
 * screen (styles.css, .lbl), so it is wider than the label the box was measured with. The margin allows
 * for the growth of a label or legend in a scaled-down window. The HUD sets it from its scale to the
 * window (Hud.setFrame); at 1600 × 900 and larger it is 0.
 */
let labelRoom = 0;
export function setLabelRoom(uiScale: number, compact: boolean) {
  labelRoom = Math.max(0, 11.5 / Math.max(0.2, uiScale) - 13.5) * (compact ? 16 : 20);
}

/**
 * Fit `box` into `safe` (both in frame pixels), with an optional preferred vertical center and left alignment.
 * `pad` keeps the figure's outermost labels off the edge of the safe area; `labelRoom` adds to it at the sides.
 */
export function fit(box: Box, safe: Box, out: View, {
  min = 0.55,
  centerY = (safe[1] + safe[3]) / 2,
  alignLeft = false,
} = {}): View {
  const pad = 14;
  const bw = Math.max(40, box[2] - box[0]) + 2 * (pad + labelRoom), bh = Math.max(40, box[3] - box[1]) + 2 * (pad + labelRoom / 4);
  const sw = Math.max(0, safe[2] - safe[0]), sh = Math.max(0, safe[3] - safe[1]);
  out.s = Math.max(min, Math.min(1.5, sw / bw, sh / bh));
  out.px = (box[0] + box[2]) / 2;
  out.py = (box[1] + box[3]) / 2;
  out.qx = alignLeft ? safe[0] + Math.min(sw, bw * out.s) / 2 : (safe[0] + safe[2]) / 2;
  const half = Math.min(sh, bh * out.s) / 2;
  out.qy = Math.max(safe[1] + half, Math.min(centerY, safe[3] - half));
  return out;
}
