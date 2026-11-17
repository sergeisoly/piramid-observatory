import type { Engine } from './engine';
import type { Chapter, Hotspot, NoteDetails, Probe, Readout } from './types';
import { FROZEN } from './explore';

/** What the notes column shows: a numbered note (a hotspot on the figure), an authored details page, or a probe card. */
type Pin = { kind: 'hot'; id: string } | { kind: 'numbers' } | { kind: 'details' } | { kind: 'probe'; probe: Probe };

/**
 * Owns the selected page of a slide's Notes. Its held picture defines a stable stack of numbered
 * notes, measurements when present, and optional authored prose. Marker visibility stays live,
 * while changing the page requires navigation or explicit selection. A new slide starts at page one.
 */
export class Inspector {
  private pin: Pin | null = null;
  private hover: Probe | null = null;
  private slide = -1;
  private chapter: Chapter | null = null;
  private endShown = false;
  private pages: string[] = []; // the held slide's notes, independent of markers fading in/out
  private hotPages = new Map<string, Hotspot>();
  private numbers: Readout[] = [];
  private details: NoteDetails | undefined;
  open = false;
  /** Opens the notes layer when a viewer clicks a marker shown while the notes are closed (set by Explore). */
  onPeekOpen: () => void = () => {};
  /** Closes the notes layer (the note's × button; set by Explore). */
  onClose: () => void = () => {};
  /** A slide may have no Notes; the surrounding interface keeps that column out of the figure. */
  onAvailability: () => void = () => {};

  constructor(private engine: Engine) {
    engine.onInspect = (ch, t) => this.frame(ch, t);
    engine.onStageClick = () => {
      // When frozen, clicks on the figure leave the selected page in place.
      if (!this.open || FROZEN) return;
      // Refresh after pointerup instead of using hover from before the gesture.
      const e = this.engine, ch = e.chapters[e.active], input = e.pointerInput();
      const probe = input && ch?.probe ? ch.probe(e.T - e.starts[e.active], input) : null;
      this.pin = probe ? { kind: 'probe', probe } : null;
    };
    engine.hud.onHot = (id) => {
      // a marker clicked while the notes are closed opens them on that marker's note
      if (!this.open) this.onPeekOpen();
      this.pin = { kind: 'hot', id };
    };
    engine.hud.onNoteStep = (dir) => this.step(dir);
    engine.hud.onInspectClose = () => this.onClose();
  }

  setOpen(on: boolean) {
    this.open = on;
    if (on) return;
    this.clear();
    const e = this.engine;
    e.hud.setHotspots(e.camera, e.lw, e.lh, null, null);
    e.hud.setRing(e.camera, e.lw, e.lh, null);
    e.hud.hideInspect();
    e.hud.hideNote();
  }

  /** Forget what was pinned (chapter changed or was rebuilt). Returns whether a probe card was pinned. */
  clear() {
    const had = this.pin?.kind === 'probe';
    this.pin = null;
    this.hover = null;
    return had;
  }

  /** Pin a hotspot by id (used by deep links and the frame grabber). */
  pinHot(id: string) {
    this.pin = { kind: 'hot', id };
  }
  /** Show an authored details page (the D key, deep links, the frame grabber). */
  pinDetails() {
    if (!this.details) return false;
    this.pin = { kind: 'details' };
    return true;
  }
  /** Stable page identity for restoring a reading session after a development reload. */
  get selectedPage(): string | null {
    return this.pin?.kind === 'hot' ? this.pin.id : this.pin?.kind === 'numbers' || this.pin?.kind === 'details' ? this.pin.kind : null;
  }
  /** Restore only a page owned by the current slide, after its first frame has established the stack. */
  selectPage(id: string) {
    const index = this.pages.indexOf(id);
    if (index >= 0) this.pin = this.pageAt(index);
  }
  /** Whether this slide has authored context beyond its numbered notes. */
  get hasDetails() {
    return this.details !== undefined;
  }
  /** Before the first frame, defer availability until the slide has established its pages. */
  get hasNotes() {
    return this.chapter === null || this.pages.length > 0;
  }
  /** Whether the authored details page is on show. */
  get onDetails() {
    return this.open && this.pin?.kind === 'details';
  }
  /** Probe at a frame-pixel position and pin the result (used by deep links and the frame grabber). */
  pinAt(x: number, y: number) {
    const e = this.engine, ch = e.chapters[e.active];
    if (!ch?.probe) return;
    const p = ch.probe(e.T - e.starts[e.active], e.probeInput(x, y));
    if (p) this.pin = { kind: 'probe', probe: p };
  }

  /** One page back (dir < 0) or on through the slide's stack; stops at either end. */
  step(dir: number) {
    if (!this.open || !this.pages.length) return;
    const cur = this.pin?.kind === 'hot' ? this.pin.id : this.pin?.kind === 'details' || this.pin?.kind === 'numbers' ? this.pin.kind : this.pages[0];
    const k = Math.max(0, Math.min(this.pages.length - 1, this.pages.indexOf(cur) + Math.sign(dir)));
    this.pin = this.pageAt(k);
  }

  private pageAt(k: number): Pin {
    const id = this.pages[k];
    return id === 'details' ? { kind: 'details' } : id === 'numbers' ? { kind: 'numbers' } : { kind: 'hot', id };
  }

  private frame(ch: Chapter, t: number) {
    const e = this.engine, hud = e.hud, cam = e.camera, w = e.lw, h = e.lh;
    const context = e.slideInfo(), slide = context.index;
    // A slide owns its page stack. Sampling its held picture makes the first note available
    // immediately, even while the previous picture fades out or the new markers have not arrived.
    // Rebuilt chapters also invalidate the stack; ordinary animation never does.
    if (slide !== this.slide || ch !== this.chapter || hud.endShown !== this.endShown) {
      // (the first frame is not a change of slide: a page asked for by a deep link before it stays)
      if (this.slide >= 0) this.pin = null;
      this.slide = slide;
      this.chapter = ch;
      this.endShown = hud.endShown;
      const held = context.hold - e.starts[e.active];
      // A title card has no Notes pages. The reading session can resume on the first figure.
      const hasContent = held >= 0 && !hud.endShown;
      this.hotPages = new Map((hasContent ? ch.inspect?.(held) ?? [] : []).map((hot) => [hot.id, hot]));
      this.numbers = hasContent ? ch.readouts?.(held) ?? [] : [];
      this.details = hasContent ? ch.details?.(held) : undefined;
      this.pages = [...this.hotPages.keys(), ...(this.numbers.length ? ['numbers'] : []), ...(this.details ? ['details'] : [])];
      this.onAvailability();
    }
    // Sample the actual frame last: some chapters refresh their marker transforms in inspect().
    // Only this slide's markers are offered during a crossfade, never the previous slide's notes.
    const showFigure = !hud.endShown && e.T >= e.starts[e.active];
    const hots = (showFigure ? ch.inspect?.(t) ?? [] : []).filter((hot) => this.hotPages.has(hot.id));
    for (const hot of hots) this.hotPages.set(hot.id, hot);
    const input = showFigure ? e.pointerInput() : null;
    // Read-only data lookup leaves the selected Notes page in place.
    this.hover = input ? ch.hover?.(t, input) ?? ch.probe?.(t, input) ?? null : null;
    const showHover = () => {
      hud.setRing(cam, w, h, this.hover?.pos ?? null);
      if (this.hover) hud.showInspect(cam, w, h, this.hover.pos, this.hover, false);
      else hud.hideInspect();
    };
    if (!this.open) {
      // The second layer of the film: while a slide holds or the film is paused, the figure shows its
      // numbered markers faintly, so a viewer can see that more is there and click a marker to read it.
      // While the film plays nothing extra is on screen.
      const peek = !e.playing && t >= 0 && hots.length > 0;
      hud.setPeek(peek);
      hud.setHotspots(cam, w, h, peek ? hots : null, null, this.pages);
      showHover();
      return;
    }
    hud.setPeek(false);

    // Explicit deep links can name a page outside this slide. Resolve once to its first page.
    if (this.pin?.kind === 'hot' && !this.pages.includes(this.pin.id)) this.pin = null;
    if ((this.pin?.kind === 'details' || this.pin?.kind === 'numbers') && !this.pages.includes(this.pin.kind)) this.pin = null;
    if (!this.pin && this.pages.length) {
      this.pin = this.pageAt(0);
    }

    const N = this.pages.length;
    if (this.pin?.kind === 'hot') {
      const id = this.pin.id, hs = this.hotPages.get(id)!;
      // a panel's explanation goes to the right column, so it never covers the panel it explains
      hud.setHotspots(cam, w, h, hots, id, this.pages);
      showHover();
      hud.showNote(hs, this.pages.indexOf(id) + 1, `${slide}:hot:${id}`, this.pages.indexOf(id) + 1, N);
      return;
    }
    if (this.pin?.kind === 'numbers') {
      hud.setHotspots(cam, w, h, hots, null, this.pages);
      showHover();
      hud.showNumbersPage(this.numbers, this.pages.indexOf('numbers') + 1, N, `${slide}:numbers`);
      return;
    }
    if (this.pin?.kind === 'details') {
      hud.setHotspots(cam, w, h, hots, null, this.pages);
      showHover();
      hud.showDetailsPage(this.details!, N, N, `${slide}:details`);
      return;
    }
    hud.hideNote();
    hud.setHotspots(cam, w, h, hots, null, this.pages);
    const shown = this.pin?.kind === 'probe' ? this.pin.probe : this.hover;
    if (shown) {
      hud.setRing(cam, w, h, shown.pos);
      hud.showInspect(cam, w, h, shown.pos, shown, this.pin !== null);
    } else {
      hud.setRing(cam, w, h, null);
      hud.hideInspect();
    }
  }
}
