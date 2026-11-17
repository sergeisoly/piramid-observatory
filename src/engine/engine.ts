import * as THREE from 'three/webgpu';
import { buildPost, type Post } from './post';
import { Hud } from './hud';
import { sampleShots, applyCamera, type CameraState } from './director';
import { Backdrop } from '../kit/backdrop';
import { FX } from '../kit/points';
import { THEME } from './palette';
import { smooth } from '../lib/math';
import type { Chapter, ChapterFactory, FigureInteraction, FrameApi, Params, ProbeInput } from './types';
import { setStrokeScale } from '../kit/lines';
import { boxAt, fit, type Box, type View } from './framing';
import { CARDS, CARD_HOLD } from './cards';
import { FigureViewport } from './viewport';

/** The frame the HUD and every chapter's figure are laid out for. */
const DESIGN_W = 1600, DESIGN_H = 900;

const FADE = 1.1; // seconds to fade out or in at a chapter boundary
const END_OFFSET = 0.01; // stay within the last chapter while the closing card is up
/**
 * How fast film time runs against the clock at 1× speed. The chapters' own times (beats, shots, durations)
 * were composed first and the captions were then measured against a reader: at 0.87 every caption stays
 * up 15 % longer, which brings the film's reading load to about 2.4 words a second. Times shown to the
 * viewer are clock times (film time divided by this).
 */
export const TEMPO = 0.87;

const RATES = [1.5, 1.25, 1, 0.75, 0.5]; // playback speeds, from fastest to slowest

/**
 * Reading time. The film plays the same slides a reader steps through, and a slide's text is a short
 * paragraph: longer than its animation. When film playback reaches the end of a slide's animation it
 * waits there until the text has been up long enough to read at READ_WPS words a second, plus a moment
 * to look at the figure.
 */
const READ_WPS = 2.5, READ_LOOK = 2; // a paragraph read without a moving figure beside it: 150 words a minute, and two seconds for the figure
/** Words a reader has to take in: tags do not count, a formula between dollar signs counts as one word. */
export function readingWords(html: string) {
  const text = html.replace(/<[^>]*>/g, ' ').replace(/\$[^$]*\$/g, ' x ').replace(/&[a-z]+;/g, ' ');
  return text.split(/\s+/).filter((w) => /[A-Za-z0-9]/.test(w)).length;
}

// The layout around the figure, in frame pixels (see the HUD section of styles.css).
const MARGIN = 40;
const HEAD = 100; // the running head and the definition end above this line
const FOOT = 246; // room at the foot for five lines of text, a clear band and the transport
const NOTE_SIDE_GUTTER = 44;
const MIN_NOTE_SIDE_GUTTER = 34; // CSS pixels, since legends stop shrinking in small windows
const NOTE_BOTTOM_GUTTER = 16;
const NOTE_COLUMN_WIDTH = 420; // fallback before Notes have laid out
const BOARD = 620 + 40; // a chapter's board of results (Chapter.board) and the gutter beside it

/**
 * Owns the renderer, the lens, the clock and the chapter timeline. The film is a pure
 * function of the global time T: every frame the active chapter is asked to pose itself at
 * its local time, which is what makes the whole thing scrubbable.
 */
export class Engine {
  renderer!: THREE.WebGPURenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(40, 1, 0.05, 3000);
  post!: Post;
  backdrop!: Backdrop;
  readonly hud: Hud;
  chapters: Chapter[] = [];
  /** Parallel to `chapters`: the factory that built each one and the parameters it was built with. */
  factories: (ChapterFactory | undefined)[] = [];
  params: Params[] = [];
  /** The reader's Notes session; slides without note pages temporarily hide its column. */
  notesOpen = false;
  /** Slides mode: playback stops at the end of every beat, and the viewer steps on (see stepSlide). */
  slides = false;
  /** Playback speed. The captions of a scientific film are dense, and a viewer may want more time for each. */
  rate = 1;
  private slideCache: { i: number; a: number; h: number; x: number }[] | null = null;
  /** Clock seconds the film still waits at the end of the current slide, so that its text can be read (film mode only). */
  private dwell: number | null = null;
  /** Parallel to `chapters`: the seconds of opening card before each chapter's own time starts. */
  intros: number[] = [];
  /** ?frame=base draws every figure where its scene put it, with no re-framing (tools/boxes.mjs measures that). */
  private baseFrame = new URLSearchParams(location.search).get('frame') === 'base';
  private box: Box = [0, 0, 0, 0];
  private safe: Box = [0, 0, 0, 0];
  private belowNote: Box = [0, 0, 0, 0];
  private heldBox: Box = [0, 0, 0, 0];
  private view: View = { s: 1, px: 0, py: 0, qx: 0, qy: 0 };
  /** The view offset in force (magnification and shift, in stage pixels), eased towards its target. */
  private vo = { s: 1, x: 0, y: 0, live: false };
  readonly viewport = new FigureViewport();
  private figureClipped = false;
  /** Horizontal lens shift (fraction of width). 0 by default; available to a host that needs to move the subject sideways. */
  lensTarget = 0;
  /** HUD scale and logical size (see resize): the frame, in the pixels labels and probes use. */
  private ui = 1;
  lw = DESIGN_W;
  lh = DESIGN_H;
  /** Pointer over the figure, in frame pixels. */
  private ptr = { x: 0, y: 0, on: false };
  private _v = new THREE.Vector3();
  private _d = new THREE.Vector3();
  private lens = 0;
  starts: number[] = [];
  total = 0;
  /** Film time per second of clock time at 1× speed (tools read it to turn film times into clock times). */
  readonly tempo = TEMPO;
  T = 0;
  playing = false;
  /** A lost graphics device is final: no frame or control can restart the film. */
  lost = false;
  active = -1;
  backend = '';
  frames = 0;
  /** Fired when the active chapter changes (used by the score). */
  onChapter: (c: Chapter, i: number) => void = () => {};
  /** Fired when the active chapter changes or is rebuilt (used by the Explore panel). */
  onActive: (i: number) => void = () => {};
  /** Fired when a new caption beat begins. */
  onBeat: (c: Chapter, beat: number) => void = () => {};
  /** Fired every frame after the chapter has posed itself (used by the Explore inspector). */
  onInspect: (c: Chapter, t: number) => void = () => {};
  /** Fired when the figure is clicked without being dragged. */
  onStageClick: () => void = () => {};

  private cam: CameraState = { pos: new THREE.Vector3(0, 0, 20), look: new THREE.Vector3(), fov: 40 };
  private last = 0;
  private clock = 0;
  private gesture: { pointer: number; interaction: FigureInteraction | null; region: string | null; pan: boolean; draggable: boolean; travel: number; lx: number; ly: number } | null = null;
  private cameraSlide = -1;
  fps = 60;
  private w = 1;
  private h = 1;
  private dpr = 1;
  private beat = -1;
  private api: FrameApi;

  constructor(private stage: HTMLElement, hudRoot: HTMLElement) {
    this.hud = new Hud(hudRoot, stage);
    this.hud.onSeek = (i, f) => this.goto(i, f);
    this.hud.onToggle = () => (this.slides ? this.stepSlide(1) : this.toggle());
    this.hud.onPrev = () => this.stepSlide(-1);
    this.hud.onSlides = () => this.setSlides(!this.slides);
    this.hud.onRate = () => this.stepRate(-1, true);
    this.hud.onResetView = () => this.resetView();
    this.hud.onSourceOpen = () => this.pause();
    this.api = {
      label: (key, pos, text, opts) => this.hud.label(this.camera, this.lw, this.lh, key, pos, text, opts),
      camera: this.camera,
      clock: 0,
    };
  }

  async init() {
    const r = (this.renderer = new THREE.WebGPURenderer({ antialias: false, powerPreference: 'high-performance' } as any));
    await r.init();
    // paper: the output stage does its own ink-on-paper mapping, so no tone curve.
    r.toneMapping = THEME === 'paper' ? THREE.NoToneMapping : THREE.NeutralToneMapping;
    r.toneMappingExposure = THEME === 'paper' ? 1.0 : 1.1;
    r.setClearColor(0x000000, 1);
    document.documentElement.dataset.theme = THEME;
    this.backend = (r.backend as any).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
    // Stop the frame loop and disable the controls after device loss.
    (r as any).onDeviceLost = (info: any) => {
      if (this.lost) return;
      console.error('graphics device lost', info);
      this.lost = true;
      this.playing = false;
      r.setAnimationLoop(null);
      this.hud.element.inert = true;
      const el = document.createElement('div');
      el.className = 'fatal';
      el.setAttribute('role', 'alert');
      el.textContent = 'The browser lost the graphics device, and the figure has stopped. Reloading starts this presentation again.';
      document.body.appendChild(el);
    };
    this.stage.appendChild(r.domElement);
    this.backdrop = new Backdrop();
    this.scene.add(this.backdrop.object);
    this.post = buildPost(r, this.scene, this.camera);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.compactFrame.addEventListener('change', () => this.resize());
    // Local object controls take precedence. Other figures magnify and pan in their flat viewport.
    const el = this.stage;
    const down = { x: 0, y: 0, t: 0 };
    const point = (e: { clientX: number; clientY: number }) => {
      const box = this.stage.getBoundingClientRect();
      this.ptr.x = (e.clientX - box.left) / this.ui; this.ptr.y = (e.clientY - box.top) / this.ui; this.ptr.on = true;
    };
    const pick = () => {
      if (!this.inFigure(this.ptr.x, this.ptr.y)) return { interaction: null, region: null };
      const ch = this.chapters[this.active], interaction = ch?.interaction ?? null;
      const region = interaction?.pick(this.T - this.starts[this.active], this.probeInput(this.ptr.x, this.ptr.y)) ?? null;
      return { interaction: region === null ? null : interaction, region };
    };
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || this.gesture) return;
      point(e);
      const target = pick();
      const pan = !target.interaction && this.viewport.changed && this.inFigure(this.ptr.x, this.ptr.y);
      const draggable = target.interaction && target.region !== null ? !!target.interaction.draggable?.(target.region) : pan;
      this.gesture = { pointer: e.pointerId, ...target, pan, draggable, travel: 0, lx: e.clientX, ly: e.clientY };
      down.x = e.clientX; down.y = e.clientY; down.t = performance.now();
      el.setPointerCapture(e.pointerId);
    });
    const end = () => { this.gesture = null; el.classList.remove('is-dragging'); };
    el.addEventListener('pointerup', (e) => {
      const gesture = this.gesture;
      if (!gesture || gesture.pointer !== e.pointerId) return;
      end();
      point(e); // a touch tap can arrive without a preceding pointermove
      // A press that did not travel is a click; a drag returning to its origin still is not.
      if (Math.max(gesture.travel, Math.hypot(e.clientX - down.x, e.clientY - down.y)) < 5 && performance.now() - down.t < 600) this.onStageClick();
      if (e.pointerType !== 'mouse') this.ptr.on = false;
    });
    el.addEventListener('pointercancel', (e) => { if (this.gesture?.pointer === e.pointerId) { end(); this.ptr.on = false; } });
    el.addEventListener('lostpointercapture', (e) => { if (this.gesture?.pointer === e.pointerId) end(); });
    el.addEventListener('pointerleave', () => { this.ptr.on = false; });
    el.addEventListener('pointermove', (e) => {
      point(e);
      const gesture = this.gesture;
      if (!gesture || gesture.pointer !== e.pointerId) return;
      gesture.travel = Math.max(gesture.travel, Math.hypot(e.clientX - down.x, e.clientY - down.y));
      if (gesture.interaction && gesture.region !== null) {
        if (gesture.draggable) gesture.interaction.drag?.(this.T - this.starts[this.active], gesture.region, (e.clientX - gesture.lx) / this.ui / this.viewport.scale, (e.clientY - gesture.ly) / this.ui / this.viewport.scale);
      } else if (gesture.pan) this.viewport.pan((e.clientX - gesture.lx) / this.ui, (e.clientY - gesture.ly) / this.ui);
      el.classList.toggle('is-dragging', gesture.draggable && gesture.travel >= 3);
      gesture.lx = e.clientX; gesture.ly = e.clientY;
    });
    el.addEventListener('wheel', (e) => {
      if (this.hud.compact && !e.altKey) return;
      e.preventDefault();
      point(e);
      const { interaction, region } = pick();
      const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.h : 1);
      if (interaction?.wheel && region !== null) interaction.wheel(this.T - this.starts[this.active], region, delta);
      else if (this.inFigure(this.ptr.x, this.ptr.y)) this.viewport.zoom(this.ptr.x, this.ptr.y, delta);
    }, { passive: false });
    el.addEventListener('dblclick', () => this.resetView());
  }

  /** Restore the active figure's viewer transform; the authored timeline camera continues. */
  resetView() {
    this.hud.hideFigureHelp();
    this.chapters[this.active]?.interaction?.reset();
    this.viewport.reset();
    this.gesture = null;
    this.stage.classList.remove('is-dragging');
    this.hud.setViewChanged(false);
  }

  private inFigure(x: number, y: number) {
    return this.active >= 0 && this.T >= this.starts[this.active] && !this.hud.endShown && this.viewport.contains(x, y);
  }

  /** Keep rendering, DOM labels and picking on one shared affine map. */
  private updateViewport() {
    const view = this.viewport;
    view.setBounds(this.safe);
    this.post.U.view.value.set(view.scale, view.x / this.lw, view.y / this.lh);
    this.post.U.viewBounds.value.set(view.bounds[0] / this.lw, view.bounds[1] / this.lh, view.bounds[2] / this.lw, view.bounds[3] / this.lh);
    this.figureClipped = view.changed || !!this.chapters[this.active]?.interaction?.changed();
    this.post.U.clipFigure.value = this.figureClipped ? 1 : 0;
    // Rotating or zooming a single object clips the ink, but the other panels' labels
    // and Notes markers still belong to the unzoomed figure.
    this.hud.setViewport(view, this.lw, this.lh);
  }

  private updateCursor() {
    let draggable = false;
    if (this.ptr.on && this.inFigure(this.ptr.x, this.ptr.y)) {
      const ch = this.chapters[this.active], target = ch.interaction?.pick(this.T - this.starts[this.active], this.probeInput(this.ptr.x, this.ptr.y));
      draggable = target != null ? !!ch.interaction?.draggable?.(target) : this.viewport.changed;
    }
    this.stage.classList.toggle('is-pannable', draggable);
  }

  private readonly compactFrame = matchMedia('(max-width: 900px), (max-height: 426px), (max-width: 1080px) and (max-aspect-ratio: 5/4)');
  private resize() {
    this.hud.setCompact(this.compactFrame.matches);
    const w = Math.max(2, this.stage.clientWidth), h = Math.max(2, this.stage.clientHeight);
    // Spend at most ~3.4 megapixels: full retina in a window, slightly less when fullscreen.
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, Math.sqrt(3.4e6 / (w * h))));
    this.w = w;
    this.h = h;
    // The HUD is designed for a 1600 × 900 frame and scaled as one layer: its logical size is the
    // window divided by the scale, so text, panels and labels keep their proportion to the figure.
    this.ui = Math.min(w / DESIGN_W, h / DESIGN_H);
    this.lw = w / this.ui;
    this.lh = h / this.ui;
    this.hud.setFrame(this.lw, this.lh, this.ui);
    setStrokeScale(dpr * this.ui);
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.vo.live = false; // the next frame sets the view afresh
    this.setView(1, -w * this.lens, h * 0.045);
  }

  /** Magnify the base picture by `s` and shift it by (x, y) stage pixels. s = 1, y = 4.5 % of the height is the base frame. */
  private setView(s: number, x: number, y: number) {
    this.camera.setViewOffset(this.w * s, this.h * s, x, y, this.w, this.h);
    this.camera.updateProjectionMatrix();
  }

  /** The part of the frame the layout leaves to the figure, in frame pixels. */
  private safeArea(out: Box) {
    out[0] = MARGIN;
    if (this.hud.compact) {
      out[1] = MARGIN;
      out[2] = this.lw - MARGIN;
      out[3] = this.lh - MARGIN;
      return out;
    }
    out[1] = Math.max(HEAD, this.hud.headEnd() + 14);
    // The closing board or the Notes column use the right of the frame.
    const col = this.hud.notesColumnVisible;
    const note = this.hud.noteColumnBounds;
    const right = this.lw - MARGIN;
    const gutter = Math.max(NOTE_SIDE_GUTTER, MIN_NOTE_SIDE_GUTTER / this.ui);
    out[2] = col ? (note ? note[0] : right - NOTE_COLUMN_WIDTH) - gutter : right - (this.hud.reservesBoard ? BOARD : 0);
    // Leave a clear line of space between the lowest label and the caption or measurements.
    out[3] = Math.min(this.lh - FOOT, this.hud.captionTop(this.lh) - 24, this.hud.readoutsTop(this.lh) - 16);
    return out;
  }

  /**
   * Re-frame: fit the figure's box (measured in the base frame, see framing.ts) into the safe area.
   * The scene's own camera frames the figure for the base frame; this is a zoom and a shift of that
   * finished picture, so labels, probes and hotspots follow through the camera's projection.
   */
  private reframe(id: string, t: number, hold: number, dt: number, snap: boolean) {
    const w = this.w, h = this.h, ui = this.ui, vo = this.vo;
    const previousTop = this.safe[1], previousRight = this.safe[2], previousBottom = this.safe[3];
    const safe = this.safeArea(this.safe);
    let s = 1, x = -w * this.lens, y = h * 0.045;
    if (!this.baseFrame) {
      const box = boxAt(id, t, this.box);
      const col = !this.hud.compact && this.hud.notesColumnVisible;
      const note = this.hud.noteColumnBounds;
      const centerY = (safe[1] + safe[3]) / 2;
      let under = false;
      if (col && note) {
        const below = this.belowNote;
        below[0] = safe[0];
        below[1] = Math.max(safe[1], note[1] + NOTE_BOTTOM_GUTTER);
        below[2] = this.lw - MARGIN;
        below[3] = safe[3];
        // Use the held picture for this choice so an animated change in aspect ratio
        // cannot move the figure back and forth between the two regions.
        const held = boxAt(id, hold, this.heldBox);
        const sideScale = fit(held, safe, this.view, { min: 0 }).s;
        const lower = fit(held, below, this.view, { min: 0, centerY });
        // A lower position needs to buy useful size. A short note can give the figure
        // the full width without moving it far from the center of the reading area.
        const drop = (lower.qy - centerY) / Math.max(1, safe[3] - centerY);
        under = lower.s > sideScale * (1.05 + 0.4 * drop);
        if (under) {
          safe[1] = below[1];
          safe[2] = below[2];
        }
      }
      // An incoming page can occupy more room immediately. Ease into newly freed space,
      // but keep the previous picture from crossing an occupied column or caption.
      if (col && (safe[1] > previousTop || safe[2] < previousRight || safe[3] < previousBottom)) snap = true;
      const v = fit(box, safe, this.view, { min: col ? 0 : 0.55, centerY, alignLeft: under });
      // Never show what the scene's own camera kept out of its frame (a neighboring panel during a pan),
      // unless the figure's box itself reaches past the frame's edge.
      if (v.s >= 1 && !col) {
        const lim0 = Math.min(0, box[0] - 14), lim1 = Math.max(DESIGN_W, box[2] + 14);
        const lo = lim0 + v.qx / v.s, hi = lim1 - (this.lw - v.qx) / v.s;
        if (lo <= hi) v.px = Math.max(lo, Math.min(hi, v.px));
      }
      // The authored 16:9 picture sits inside the stage. Its optical center follows the
      // stage's midpoint horizontally and sits at 45.5% of its height vertically.
      const px = v.px + (this.lw - DESIGN_W) / 2;
      const py = v.py + 0.455 * (this.lh - DESIGN_H);
      s = v.s;
      x = (s * px - v.qx) * ui - s * w * this.lens;
      y = s * (py * ui + 0.045 * h) - v.qy * ui;
    }
    if (snap || !vo.live) { vo.s = s; vo.x = x; vo.y = y; vo.live = true; }
    else {
      const k = 1 - Math.exp(-dt * 5);
      vo.s += (s - vo.s) * k; vo.x += (x - vo.x) * k; vo.y += (y - vo.y) * k;
    }
    this.setView(vo.s, vo.x, vo.y);
    this.hud.setFigureScale(vo.s);
  }

  /** What a chapter's `probe` needs for the frame-pixel position (x, y). */
  probeInput(x: number, y: number): ProbeInput {
    const cam = this.camera, w = this.lw, h = this.lh, v = this._v, d = this._d;
    const [baseX, baseY] = this.viewport.unproject(x, y);
    v.set((baseX / w) * 2 - 1, -((baseY / h) * 2 - 1), 0.5).unproject(cam);
    d.copy(v).sub(cam.position);
    const s = Math.abs(d.z) > 1e-9 ? -cam.position.z / d.z : -1;
    const plane: [number, number] | null = s > 0 ? [cam.position.x + d.x * s, cam.position.y + d.y * s] : null;
    const dist = cam.position.distanceTo(this.cam.look);
    return {
      x, y, plane,
      unit: (2 * dist * Math.tan((cam.fov * Math.PI) / 360)) / h / this.vo.s / this.viewport.scale,
      project: (px, py, pz) => {
        v.set(px, py, pz).project(cam);
        const [sx, sy] = this.viewport.project((v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h);
        return [sx, sy, v.z];
      },
    };
  }
  /** The same for the live pointer; null while it is off the figure or a gesture is in progress. */
  pointerInput(): ProbeInput | null {
    return this.ptr.on && !this.gesture && (!this.figureClipped || this.inFigure(this.ptr.x, this.ptr.y)) ? this.probeInput(this.ptr.x, this.ptr.y) : null;
  }

  addChapter(c: Chapter, factory?: ChapterFactory, params: Params = {}) {
    c.group.visible = true;
    const intro = this.cardOf(c);
    this.intros.push(intro);
    this.starts.push(this.total + intro);
    this.chapters.push(c);
    this.factories.push(factory);
    this.params.push(params);
    this.total += intro + c.duration;
    this.slideCache = null;
  }

  /** Seconds of opening card before chapter `c`. `starts[i]` is where the chapter's own time begins, after its card. */
  private cardOf(c: Chapter) {
    return CARDS[c.id]?.hold ?? CARD_HOLD;
  }
  /** Global time at which chapter i's opening card begins (the chapter's own time 0 is `starts[i]`). */
  cardStart(i: number) {
    return this.starts[i] - this.intros[i];
  }

  /**
   * Re-run chapter i's computation with its current parameters and swap the result in place.
   * The chapter is built from a snapshot of `params[i]`. That object itself stays in place: the Explore
   * panel keeps writing to it while the build runs, and a change made meanwhile must not be lost.
   */
  async rebuild(i: number, status: (msg: string) => void = () => {}) {
    const f = this.factories[i];
    if (!f) throw new Error('chapter has no factory');
    const params: Params = { ...this.params[i] };
    status('computing…');
    const ch = await f.build(
      { renderer: this.renderer, film: this, log: async (m) => { status(m); await new Promise<void>((r) => setTimeout(r, 0)); } },
      params,
    );
    const old = this.chapters[i];
    const local = Math.min(this.T - this.starts[i], ch.duration - 0.01);
    this.chapters[i] = ch;
    this.total = 0;
    this.chapters.forEach((c, k) => { this.intros[k] = this.cardOf(c); this.starts[k] = this.total + this.intros[k]; this.total += this.intros[k] + c.duration; });
    this.slideCache = null;
    if (this.active === i) {
      this.scene.remove(old.group);
      this.scene.add(ch.group);
      this.T = this.starts[i] + Math.max(0, local);
      this.beat = -1;
      this.hud.setChapters(this.chapters);
      this.hud.setChapter(ch, i, this.chapters.length);
    } else this.hud.setChapters(this.chapters);
    old.group.traverse((obj: any) => {
      obj.geometry?.dispose?.();
      const m = obj.material;
      if (Array.isArray(m)) m.forEach((x) => x.dispose?.());
      else m?.dispose?.();
    });
    if (this.active === i) this.onActive(i);
  }

  /** Compile a chapter's shaders ahead of time so cuts never hitch. */
  async warm(c: Chapter) {
    this.scene.add(c.group);
    try {
      sampleShots(c.shots, 0, this.cam);
      applyCamera(this.camera, this.cam);
      await (this.renderer as any).compileAsync(this.scene, this.camera);
    } catch (e) {
      console.warn('warm failed for', c.id, e);
    }
    this.scene.remove(c.group);
  }

  ready() {
    this.hud.setChapters(this.chapters);
    this.last = performance.now() / 1000;
    if (!this.lost) this.renderer.setAnimationLoop(this.frame);
  }

  playEpoch = 0;
  play() {
    this.playEpoch++;
    this.playing = true;
  }
  /** An explicit reader action supersedes any playback Notes might otherwise resume. */
  pause() {
    this.playEpoch++;
    this.playing = false;
  }
  setRate(r: number) {
    this.rate = RATES.includes(r) ? r : 1;
    this.hud.setRate(this.rate);
  }
  /** One step slower (dir < 0) or faster; `wrap` cycles through speed options. */
  stepRate(dir: number, wrap = false) {
    const k = RATES.indexOf(this.rate) - Math.sign(dir); // RATES runs from fast to slow
    this.setRate(RATES[wrap ? (k + RATES.length) % RATES.length : Math.max(0, Math.min(RATES.length - 1, k))]);
  }
  toggle() {
    // Play at the end of the film starts it again
    if (this.playing) { this.pause(); return; }
    if (this.ended) this.seek(0);
    this.play();
  }
  private get endTime() {
    return this.total - END_OFFSET;
  }
  /** The closing card is the stable endpoint of the presentation. */
  get ended() {
    return this.chapters.length > 0 && this.T >= this.endTime;
  }
  seek(T: number) {
    if (!Number.isFinite(T)) return;
    this.dwell = null;
    this.T = Math.max(0, Math.min(this.endTime, T));
  }
  goto(i: number, frac = 0) {
    const k = Math.max(0, Math.min(this.chapters.length - 1, i));
    if (this.slides && frac === 0 && this.intros[k] > 0) {
      this.pause();
      this.seek(this.cardHold(k));
      return;
    }
    this.seek(this.cardStart(k) + (this.intros[k] + this.chapters[k].duration) * frac + (frac === 0 ? 0.01 : 0));
  }
  /** The held frame before the divider fades into the first figure. */
  private cardHold(i: number) {
    return this.starts[i] - 0.5;
  }
  /**
   * The film as slides: one per beat. A slide starts at its caption (global time `a`) and is held at
   * `h`, just before the next caption, or before the fade at the end of its chapter.
   */
  private slideList() {
    if (!this.slideCache) {
      const L: { i: number; a: number; h: number; x: number }[] = [];
      // x: the clock seconds the film waits at h, on top of the time the slide's own animation gives its text
      const wait = (a: number, h: number, words: number) => Math.max(0, words / READ_WPS + READ_LOOK - (h - a) / TEMPO);
      this.chapters.forEach((c, i) => {
        // the opening card is a slide of its own
        if (this.intros[i] > 0) {
          const a = this.cardStart(i), h = this.cardHold(i);
          L.push({ i, a, h, x: wait(a, h, readingWords(`${c.title} ${CARDS[c.id]?.line ?? ''}`)) });
        }
        c.beats.forEach((b, j) => {
          const end = j + 1 < c.beats.length ? c.beats[j + 1].at - 0.05 : c.duration - FADE - 0.15;
          const a = this.starts[i] + b.at, h = this.starts[i] + Math.max(b.at + 0.1, Math.min(end, b.hold ?? end));
          L.push({ i, a, h, x: wait(a, h, readingWords(b.caption)) });
        });
      });
      // the last slide is the end card
      if (L.length) L.push({ i: this.chapters.length - 1, a: this.endTime, h: this.endTime, x: 0 });
      this.slideCache = L;
    }
    return this.slideCache;
  }
  /** The slide that holds time T: the last one that has started, or the first one of T's chapter. */
  private slideAt(T: number) {
    const L = this.slideList();
    let k = 0;
    for (let n = 0; n < L.length && L[n].a <= T + 1e-6; n++) k = n;
    let c = this.chapters.length - 1;
    while (c > 0 && T < this.cardStart(c)) c--;
    if (L[k] && L[k].i < c) k = Math.min(L.length - 1, k + 1); // before the first caption of a chapter
    return k;
  }
  slideInfo() {
    const slides = this.slideList(), index = this.slideAt(this.T), slide = slides[index];
    return { index, count: slides.length, start: slide.a, hold: slide.h };
  }
  /** The current slide and the action a forward step will take. */
  private slideState() {
    const L = this.slideList(), k = this.slideAt(this.T);
    return { index: k, count: L.length, beforeHold: !!L[k] && this.T < L[k].h - 0.02 };
  }
  /** Clock seconds of film playback up to film time T at 1× speed: film time at the film's tempo plus the reading waits behind T. */
  clockAt(T: number) {
    let w = 0;
    for (const s of this.slideList()) {
      if (s.h < T) w += s.x;
      else if (s.h === T && T === this.T && this.dwell !== null) w += s.x - this.dwell;
    }
    return T / TEMPO + w;
  }
  /** Clock length of the whole film at 1× speed, reading waits included. */
  get clockTotal() {
    return this.endTime / TEMPO + this.slideList().reduce((n, s) => n + s.x, 0);
  }
  setSlides(on: boolean) {
    if (on !== this.slides) {
      this.playEpoch++;
      this.dwell = null;
      if (on && this.playing && !this.slideState().beforeHold) this.playing = false;
    }
    this.slides = on;
    this.hud.setSlidesMode(on);
  }
  /**
   * Forward: play the current slide to its end and stop there; at the end, start the next slide.
   * A second press while a slide is still moving jumps to its end.
   * Chapter cards and Back land directly on the held picture.
   */
  stepSlide(dir: number) {
    const L = this.slideList();
    if (!L.length) return;
    const k = this.slideAt(this.T), cur = L[k];
    if ((dir < 0 && k === 0) || (dir > 0 && k === L.length - 1)) return;
    this.dwell = null;
    this.playEpoch++;
    if (dir > 0) {
      if (this.T < cur.h - 0.02) {
        if (this.playing) { this.T = cur.h; this.playing = false; }
        else this.playing = true;
      } else {
        const next = L[k + 1];
        this.playing = next.a < next.h && next.a >= this.starts[next.i];
        this.T = this.playing ? next.a : next.h;
      }
    } else {
      this.T = L[k - 1].h;
      this.playing = false;
    }
  }
  step(dir: number) {
    const a = Math.max(0, this.active), local = this.T - this.cardStart(a);
    this.goto(dir < 0 && local > 3 + this.intros[a] ? this.active : this.active + dir);
  }
  indexOf(id: string) {
    return this.chapters.findIndex((c) => c.id === id);
  }

  private advance(elapsed: number) {
    const T0 = this.T;
    const slide = this.slideList()[this.slideAt(T0)];
    const finishedHold = this.dwell !== null;
    this.dwell = null;
    this.T += elapsed * TEMPO;
    if (this.slides) {
      // A mode change or a seek can start playback at a slide's held endpoint.
      if (slide && this.T >= slide.h && T0 <= slide.h + 0.02) {
        this.T = Math.max(T0, slide.h);
        this.playing = false;
      }
    } else if (!finishedHold && slide && slide.x > 0 && T0 <= slide.h && this.T >= slide.h) {
      // Use the leftover fraction of this frame as the first part of the reading wait.
      const excess = (this.T - slide.h) / TEMPO;
      if (excess < slide.x) {
        this.T = slide.h;
        this.dwell = slide.x - excess;
      } else this.T -= slide.x * TEMPO;
    }
    if (this.T >= this.endTime) {
      this.T = this.endTime;
      this.playing = false;
    }
  }

  private frame = () => {
    const now = performance.now() / 1000;
    const dt = Math.min(0.1, Math.max(0, now - this.last));
    this.last = now;
    this.clock += dt;
    this.fps += (1 / Math.max(dt, 1e-3) - this.fps) * 0.05;
    if (!this.chapters.length) return;

    if (this.playing) {
      let elapsed = dt * (this.slides ? 1 : this.rate);
      if (this.dwell !== null) {
        const spent = Math.min(this.dwell, elapsed);
        this.dwell -= spent;
        elapsed -= spent;
      }
      if (elapsed > 0) this.advance(elapsed);
    }
    let i = this.chapters.length - 1;
    while (i > 0 && this.T < this.cardStart(i)) i--;
    const ch = this.chapters[i];
    const cameraSlide = this.slideAt(this.T);
    if (this.cameraSlide >= 0 && cameraSlide !== this.cameraSlide) this.resetView();
    this.cameraSlide = cameraSlide;
    const tRaw = this.T - this.starts[i]; // negative while the chapter's opening card is up
    const t = Math.max(0, tRaw);
    const card = tRaw < -0.3;

    const cut = i !== this.active;
    if (cut) {
      if (this.active >= 0) this.scene.remove(this.chapters[this.active].group);
      this.scene.add(ch.group);
      this.active = i;
      this.beat = -1;
      this.hud.setChapter(ch, i, this.chapters.length);
      this.onChapter(ch, i);
      this.onActive(i);
    }

    // Measure the incoming caption before fitting the current frame.
    let b = -1;
    let eq: string | undefined, eqNote: string | undefined;
    for (let j = 0; j < ch.beats.length && this.starts[i] + ch.beats[j].at <= this.T; j++) {
      b = j;
      if (ch.beats[j].eq !== undefined) { eq = ch.beats[j].eq; eqNote = ch.beats[j].eqNote; }
    }
    const edge = t > ch.duration - FADE * 0.8;
    this.hud.setCaption(b >= 0 && !edge ? ch.beats[b].caption : '');
    this.hud.setEquation(edge ? undefined : eq || undefined, eqNote);

    // camera
    const lt = this.w > 900 ? this.lensTarget : 0;
    if (Math.abs(lt - this.lens) > 1e-4) this.lens += (lt - this.lens) * Math.min(1, dt * 4);
    this.hud.setViewChanged(this.viewport.changed || !!ch.interaction?.changed());
    sampleShots(ch.shots, t, this.cam);
    const dist = applyCamera(this.camera, this.cam);
    FX.focus.value = dist;
    this.reframe(ch.id, t, this.slideList()[cameraSlide].h - this.starts[i], dt, cut || card);
    this.camera.updateMatrixWorld();
    this.updateViewport();

    // Fade the picture at chapter boundaries.
    this.post.U.fade.value = tRaw < 0 ? 0 : smooth(0, FADE, t) * (1 - smooth(ch.duration - FADE, ch.duration, t));
    this.hud.setCard(card);

    // chapter content + labels
    this.api.clock = this.clock;
    this.hud.beginLabels();
    ch.update(t, this.api);
    this.hud.endLabels();
    this.updateCursor();
    // Inspector must read the current frame's end state, including on a chapter cut.
    this.hud.setEnd(this.ended);
    this.onInspect(ch, t);

    // Beat notifications, telemetry and transport
    if (b !== this.beat) {
      this.beat = b;
      if (b >= 0) this.onBeat(ch, b);
    }
    // The slide shows the figure and its explanation; the numbers behind it are a page of the notes (inspect.ts).
    // Only the closing board, whose rows are the figure, shows them on the slide.
    this.hud.setReadouts(ch.board ? ch.readouts?.(t) : undefined, now);
    this.hud.setProgress(i, (this.T - this.cardStart(i)) / (this.intros[i] + ch.duration), this.clockAt(this.T), this.clockTotal, this.playing, this.slides ? this.slideState() : undefined);
    this.hud.setStat(`${this.backend} · ${Math.round(this.fps)} FPS · ${Math.round(this.w * this.dpr)}×${Math.round(this.h * this.dpr)}`);

    this.post.pipeline.render();
    this.frames++;
  };
}
