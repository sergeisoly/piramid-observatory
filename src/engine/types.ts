import type * as THREE from 'three/webgpu';

/** Which part of the PIRAMID program a chapter belongs to (drives the accent color). */
export type Pillar = 'frame' | 'theory' | 'tools' | 'data';

/** A caption (and optional equation) that becomes active at local time `at` (seconds). */
export interface Beat {
  at: number;
  caption: string; // may contain <em>/<b>
  eq?: string; // LaTeX, rendered with KaTeX in the formula corner. Give one only when the caption uses it; a formula carries on to later beats until a beat sets `eq: ''`.
  eqNote?: string; // one plain line under the formula that says what it is ("the scaling keeps the spread of y the same at every depth")
  hold?: number; // slides mode holds here (chapter time). Default: just before the next beat. Set it when the figure fades out earlier.
}

/** Camera keyframe. Positions are interpolated with a C1 Hermite spline through all shots. */
export interface Shot {
  t: number;
  pos: [number, number, number];
  look: [number, number, number];
  fov?: number;
}

/**
 * A row of a panel. Like every visible string, `label`, `value` and `note` may contain inline
 * maths between dollar signs (`$\lVert\Delta\rVert_2$`), which the HUD typesets with KaTeX.
 */
export interface Readout {
  label: string;
  value: string;
  note?: string; // optional second line under the row, set small and muted
  url?: string; // makes the value a link
  detail?: boolean; // a secondary row: the film hides it, Explore mode shows it (keep at most four rows without this flag at any time)
  mark?: string; // a CSS color: the row gets a small square of it before its label (the summary uses it for the chapter's team)
  head?: string; // HTML: a heading printed above this row; it opens a group of rows (the summary groups its rows by team)
  tag?: string; // the kind of evidence the row is ("null control", "illustration"), printed as a small tag before its note
}

/** A phrase of `Chapter.source` (or of a caption) that should be a link. arXiv ids and princint.ai/piramid are linked automatically. */
export interface Link {
  text: string;
  url: string;
}

/**
 * Explore mode: a place in the figure the viewer can ask about. The HUD draws a numbered marker
 * at `pos`; clicking it opens a card with the explanation.
 */
export interface Hotspot {
  id: string; // stable within the chapter
  pos: [number, number, number]; // world position of the marker (usually a panel's title corner)
  /** Optional offset of the marker from `pos`, in frame pixels: lets it sit beside a label anchored at the same point. */
  dx?: number;
  dy?: number;
  title: string; // plain text with $maths$, e.g. "Panel c: eigenvalues of $\bar K$"
  /** HTML with $maths$. Two or three short paragraphs (<p>): what is drawn, how to read it, what to notice in this run. */
  body: string;
  rows?: Readout[]; // optional live values that belong to the explanation
}

/** What a chapter's read-only hover gets to find the data element under the pointer. */
export interface ProbeInput {
  /** Pointer position in frame pixels (the 1600 × 900 design frame; the same pixels as LabelOpts.dx). */
  x: number;
  y: number;
  /** Where the pointer ray meets the world plane z = 0, or null if it misses. Enough for flat figures seen head-on. */
  plane: [number, number] | null;
  /** World position → frame pixels [x, y, depth]; depth outside (−1, 1) means off-screen. Use it to pick among 3-D points. */
  project(x: number, y: number, z: number): [number, number, number];
  /** World units per frame pixel at the camera's look target: multiply a pixel tolerance by it. */
  unit: number;
}

/** One data element, described in a read-only card next to it. */
export interface Probe {
  pos: [number, number, number]; // world position of the element (the HUD rings it)
  title: string; // plain text with $maths$, e.g. "Neuron 37" or "Kernel entry $\bar K(x, x')$"
  rows: Readout[]; // its values
  note?: string; // plain text with $maths$: one sentence on how to read them
}

/** Optional viewer transforms for individual figures; never changes the chapter's computation. */
export interface FigureInteraction {
  /** Pick a region in the current picture. A named region owns its drag; wheel falls back to the shared viewport when no wheel handler is supplied. */
  pick(t: number, p: ProbeInput): string | null;
  /** Whether the selected region supports a local drag; controls the grab cursor too. */
  draggable?(region: string): boolean;
  /** Frame-pixel deltas; the region picked at pointerdown stays captured through the gesture. */
  drag?(t: number, region: string, dx: number, dy: number): void;
  /** Wheel deltaY normalized to pixels. Omit to keep the shared figure zoom over this region. */
  wheel?(t: number, region: string, delta: number): void;
  reset(): void;
  changed(): boolean;
}

/**
 * One headline quantity a chapter's miniature measured, next to the value theory or the paper
 * gives for it. The finale collects these from every chapter into a reproduction table.
 */
export interface Result {
  label: string; // e.g. "CCDF tail exponent"
  measured: string; // from this run, e.g. "−1.49"
  reference: string; // e.g. "theory −3/2" or "paper 0.94 (at their scale)"
  headline?: boolean; // the row the closing summary shows for this chapter (default: the first)
  claim?: string; // what the number says, in one short plain line; the closing summary prints it under the number
  kind?: string; // the kind of evidence it is, in one to three words ("null control", "planted structure"); the summary tags the row with it
  board?: boolean; // the closing summary shows this row too, besides the headline
  /**
   * What kind of statement the row is. 'check': the rebuild agrees with theory or returns what the film
   * planted, which shows only that the rebuild works. 'finding': the run showed something. 'picture': nothing
   * was measured. The closing board groups its rows by this.
   */
  group?: 'check' | 'finding' | 'picture';
  /** The result as one plain sentence for the closing board ("An untrained network earns the same probe score as a trained one."). */
  say?: string;
}

export interface LabelOpts {
  color?: string; // CSS color
  align?: 'left' | 'center' | 'right' | 'below' | 'middle'; // center = above the anchor, below = under it, middle = on it
  alpha?: number; // 0..1
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'title';
  dx?: number; // pixel offset
  dy?: number;
  plate?: boolean; // a paper-colored backing, for a label that has to sit on top of lines
  /** Optional reading instructions (HTML with $maths$), revealed by a small accessible ? beside the label. */
  help?: string;
  /** An original source figure, opened independently of Notes from an explicit label action. */
  figure?: FigureSource;
}

/** Local original figure asset and the visible attribution that accompanies it. */
export interface FigureSource {
  asset: string;
  title: string;
  alt: string;
  credit: string;
  sourceURL: string;
  actionLabel?: string;
  /** Optional separate, larger source panels shown as a scrollable document rather than a gallery. */
  images?: { asset: string; alt: string; caption?: string }[];
}

/** Passed to Chapter.update every frame. */
export interface FrameApi {
  /** Declare a screen-space text label anchored to a world position. Must be re-declared every frame it should be visible. */
  label(key: string, pos: THREE.Vector3 | [number, number, number], text: string, opts?: LabelOpts): void;
  camera: THREE.PerspectiveCamera;
  /** Wall-clock seconds; use only for cosmetic shimmer, never for explanatory state. */
  clock: number;
}

/** Optional prose that adds context to one slide's figure and numbered notes. */
export interface NoteDetails {
  title: string;
  body: string;
}

export interface Chapter {
  id: string;
  pillar: Pillar;
  kicker: string; // small line above the title, e.g. "LEARNING THEORY"
  title: string; // 1–3 words, rendered huge
  source: string; // citation line
  /** Phrases of `source` that should be links (arXiv ids are linked automatically). */
  links?: Link[];
  duration: number; // seconds
  beats: Beat[];
  shots: Shot[];
  group: THREE.Group;
  /**
   * Set every visual to its state at local time t (seconds, 0..duration).
   * Timeline and scientific state MUST be a pure function of t so the timeline can be scrubbed in
   * both directions. Optional interaction transforms affect the view only and reset on slide changes.
   */
  update(t: number, api: FrameApi): void;
  /** Authored context for the slide at local time t; omitted when the figure and its notes suffice. */
  details?(t: number): NoteDetails | undefined;
  /** Source links on the end card, separate from the slide's Notes. */
  endSources?: Readout[];
  /**
   * The Measured rows are this chapter's subject, not a side note: the HUD sets them as a board beside
   * the figure, with every row's second line, and the figure keeps to the left of it.
   */
  board?: boolean;
  /**
   * The limit a viewer must have in view while reading this chapter's numbers, as a few words. It
   * stands in the heading of the Measured rows for the whole chapter.
   */
  caveat?: string;
  /** Live numbers for the MEASURED panel. Must be real values from the computation. */
  readouts?(t: number): Readout[];
  /** 2–4 headline quantities computed in build(), each with its reference value. */
  results?: Result[];
  /** Apply a `live` control immediately. `isDefault` restores automatic film behavior where applicable. */
  set?(key: string, value: ParamValue, isDefault?: boolean): void;
  /**
   * Hotspots visible at local time t. The inspector samples the current frame to place markers
   * and the slide's held frame to establish its Notes pages.
   */
  inspect?(t: number): Hotspot[];
  /**
   * The data element under the pointer at local time t, or null. Available with Notes open or closed;
   * called every frame while the pointer is over the figure. Must not change any state.
   */
  probe?(t: number, p: ProbeInput): Probe | null;
  /** Optional additional informational hover; its non-null results take precedence over `probe`. */
  hover?(t: number, p: ProbeInput): Probe | null;
  /** Opt-in local object controls. Unclaimed regions use the shared 2-D magnification viewport. */
  interaction?: FigureInteraction;
}

export type ParamValue = number | string | boolean;
export type Params = Record<string, ParamValue>;

/**
 * One chapter parameter, also settable through a URL override. When the parameter panel is enabled,
 * it calls `Chapter.set` for live changes and re-runs `build()` for other changes.
 */
export interface Control {
  key: string;
  label: string; // short, symbol first: "β  inverse temperature"
  kind: 'range' | 'select' | 'toggle';
  value: ParamValue; // default
  min?: number;
  max?: number;
  step?: number;
  options?: { value: ParamValue; label: string }[];
  live?: boolean;
  hint?: string; // one line: what changing it shows
  format?: (v: number) => string; // how a range value reads in the panel, when plain decimals would mislead
}

export interface BuildCtx {
  renderer: THREE.WebGPURenderer;
  /** Append a line to the boot log and yield to the browser (await it between heavy steps). */
  log(msg: string): Promise<void>;
  /** The film's chapters (live array; later chapters may read earlier chapters' `results`). */
  film?: { chapters: Chapter[]; notesOpen: boolean };
}

export interface ChapterFactory {
  id: string;
  label: string; // shown in the boot log while building
  /** Chapter parameter definitions. Every value in range must build without NaNs or exceptions. */
  controls?: Control[];
  /**
   * Optional head start. The boot calls it for every chapter before the first build, with the params
   * that `build` will get, so that a chapter can begin a long computation off the main thread (in a
   * worker) while the chapters before it build. It must return at once and must not throw.
   */
  prepare?(params?: Params): void;
  /** `params` holds control values keyed by Control.key; missing keys mean "use the default". */
  build(ctx: BuildCtx, params?: Params): Promise<Chapter>;
}
