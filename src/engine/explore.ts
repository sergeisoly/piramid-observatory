import type { Engine } from './engine';
import type { Control, ParamValue, Params } from './types';
import { coerce } from './params';
import { Inspector } from './inspect';
import { texPlain } from './tex';

const fmtVal = (c: Control, v: ParamValue) => {
  if (c.kind === 'toggle') return v ? 'on' : 'off';
  if (c.kind === 'select') return String(c.options?.find((o) => o.value === v)?.label ?? v);
  const n = v as number;
  if (c.format) return c.format(n);
  // as many decimals as the step is written with: 0.25 → 2, 0.002 → 3, 5 → 0
  const st = String(c.step ?? 1), dot = st.indexOf('.');
  const digits = st.includes('e-') ? Math.min(4, +st.split('e-')[1]) : dot < 0 ? 0 : Math.min(4, st.length - dot - 1);
  // Large counts get a thousands separator; a seed such as 2026 stays plain.
  return digits === 0 && Math.abs(n) >= 10000 ? n.toLocaleString('en-US') : n.toFixed(digits);
};

/**
 * Hide parameter controls and disable click-to-pin on the figure. Read-only hover, figure transforms,
 * and the `?p.key=value` and `?hover=x,y` URL options remain available.
 */
export const FROZEN = true;

const DEFAULT_HINT = FROZEN ? 'A numbered marker on the figure opens a note on that part.' : 'Numbered markers explain the panels. Pointing at the data reads a value, and a click pins it.';

/**
 * Open a Notes session, pause playback, and let the inspector select pages from the figure.
 * When enabled, parameter controls appear in the same column and apply changes live or rebuild
 * the chapter with the new values.
 */
export class ExplorePanel {
  private el: HTMLElement;
  private body: HTMLElement;
  private status: HTMLElement;
  private hintEl: HTMLElement;
  private root: HTMLElement;
  private shown = -1;
  private timer = 0;
  private busy = false;
  private dirty = new Set<number>(); // chapters whose parameters changed since their last build
  private wasPlaying = false;
  private epoch = 0;
  readonly inspector: Inspector;
  open = false;

  constructor(parent: HTMLElement, private engine: Engine) {
    this.root = parent;
    this.inspector = new Inspector(engine);
    this.inspector.onPeekOpen = () => this.toggle(true);
    this.inspector.onClose = () => this.toggle(false);
    this.inspector.onAvailability = () => this.syncNotes();
    const el = (this.el = document.createElement('section'));
    el.className = 'xp';
    el.innerHTML = `
      <div class="xp__head"><span>Explore</span><span class="xp__status"></span><button class="xp__reset" title="Restore this chapter's default parameters">Defaults</button><button class="xp__link" title="Copy a URL that reopens this chapter at this moment with these parameters">Copy link</button><button class="xp__close" aria-label="Close">×</button></div>
      <div class="xp__body"></div>
      <div class="xp__hintbar"></div>`;
    parent.querySelector('.hud__side')!.prepend(el);
    this.body = el.querySelector('.xp__body')!;
    this.status = el.querySelector('.xp__status')!;
    this.hintEl = el.querySelector('.xp__hintbar')!;
    this.hintEl.textContent = DEFAULT_HINT;
    el.querySelector('.xp__close')!.addEventListener('click', () => this.toggle(false));
    el.querySelector('.xp__reset')!.addEventListener('click', () => this.reset());
    el.querySelector('.xp__link')!.addEventListener('click', () => this.copyLink());
    // keep pointer and wheel events on the panel from orbiting the camera
    for (const ev of ['pointerdown', 'wheel', 'dblclick']) el.addEventListener(ev, (e) => e.stopPropagation());
    // A slider keeps the keyboard after a click and would take the arrow keys from the film. Once the
    // pointer leaves the panel, hand the keys back. (Reaching a control with Tab is unaffected.)
    el.addEventListener('pointerleave', () => {
      const f = document.activeElement as HTMLElement | null;
      if (f && el.contains(f)) f.blur();
    });
  }

  /** Hide parameter chrome when this chapter has no interactive controls. */
  private setBare(bare: boolean) {
    this.el.classList.toggle('is-bare', bare);
  }

  toggle(force?: boolean) {
    if (!this.open && force !== false && this.engine.ended) return; // the end card has no notes
    const open = force ?? !this.open;
    if (open && !this.open && !this.inspector.hasNotes && force !== true) return;
    if (open === this.open) return; // repeated open requests must preserve the reading and playback state
    const focused = document.activeElement;
    const restoreFocus = !open && (focused?.closest('.hud__note, .xp, .insp')
      || (focused?.matches('.xbtn') && !this.inspector.hasNotes));
    this.open = open;
    if (!this.open) this.setBare(false);
    // the notes need the chrome: opening them brings back what H hid
    if (this.open && this.root.classList.contains('is-minimal')) this.engine.hud.toggleChrome(false);
    this.syncNotes();
    this.engine.notesOpen = this.open;
    this.inspector.setOpen(this.open);
    // Explore is "stop and look": pause on the way in, pick the film up again on the way out
    if (this.open) {
      this.wasPlaying = this.engine.playing;
      this.epoch = this.engine.playEpoch;
      this.engine.playing = false;
      this.render(this.engine.active);
    } else {
      if (this.wasPlaying && this.engine.playEpoch === this.epoch && !this.engine.ended) this.engine.playing = true;
      if (restoreFocus) this.engine.hud.focusNotes();
    }
  }

  /** An open reading session survives an empty slide without showing an empty column. */
  private syncNotes() {
    const available = this.inspector.hasNotes, session = this.open && !this.engine.ended, visible = session && available;
    const btn = this.root.querySelector<HTMLButtonElement>('.xbtn');
    if (btn) {
      btn.disabled = !available && !session;
      btn.textContent = session ? 'Close notes' : 'Notes';
      btn.title = session ? 'Close notes (E)' : available ? 'Read the notes for this slide (E)' : 'No notes on this slide';
      btn.classList.toggle('is-on', visible);
      btn.setAttribute('aria-expanded', String(visible));
    }
    this.el.classList.toggle('is-open', visible);
    this.root.classList.toggle('is-explore', visible);
    this.engine.hud.syncCol();
    document.documentElement.classList.toggle('is-explore', visible);
  }

  /** Called by the engine whenever the active chapter changes or is rebuilt. */
  onChapter(i: number) {
    this.inspector.clear();
    // A rebuild of the chapter on show leaves the controls alone: they already hold the latest values,
    // and the viewer may be dragging one of them.
    if (this.open && !(this.busy && i === this.shown)) this.render(i);
  }

  /** Esc: let go of a pinned probe card first, close the notes second. */
  escape() {
    if (!this.open) return;
    if (!this.inspector.clear()) this.toggle(false);
  }

  /** D: open this slide's authored context when present; press again to close Notes. */
  showDetails() {
    if (!this.inspector.hasDetails) return;
    if (this.inspector.onDetails) { this.toggle(false); return; }
    if (!this.open) this.toggle(true);
    this.inspector.pinDetails();
  }

  private controls(i: number): Control[] {
    return FROZEN ? [] : this.engine.factories[i]?.controls ?? [];
  }

  private render(i: number) {
    if (i < 0) return;
    this.shown = i;
    this.status.textContent = '';
    this.hintEl.textContent = DEFAULT_HINT;
    const controls = this.controls(i);
    this.body.innerHTML = '';
    // A chapter without parameters leaves the column to its authored Notes.
    this.setBare(!controls.length);
    if (!controls.length) return;
    controls.forEach((c, k) => {
      const v = coerce(c, this.engine.params[i][c.key]);
      const row = document.createElement('div');
      row.className = 'xp__row';
      const id = `xp-${k}`;

      // labels are written "symbol  description": the symbol is set in ink, the description muted
      const name = document.createElement('label');
      name.htmlFor = id;
      const [sym, ...rest] = c.label.split(/\s{2,}/);
      // (a one-word label such as "seed" is a name; a longer label with no symbol part is all description)
      const symEl = document.createElement(rest.length || !/\s/.test(sym) ? 'b' : 'span');
      symEl.innerHTML = texPlain(sym);
      const restEl = document.createElement('span');
      restEl.innerHTML = rest.length ? ' ' + texPlain(rest.join(' ')) : '';
      name.append(symEl, restEl);
      name.title = c.label.replace(/\s{2,}/, ': ').replace(/\$/g, '') + (c.live ? ' (applies at once)' : ' (re-runs the computation)');

      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = fmtVal(c, v);

      let input: HTMLInputElement | HTMLSelectElement;
      if (c.kind === 'range') {
        const r = (input = document.createElement('input'));
        r.type = 'range';
        r.min = String(c.min ?? 0);
        r.max = String(c.max ?? 1);
        r.step = String(c.step ?? 'any');
        r.value = String(v);
      } else if (c.kind === 'select') {
        const s = (input = document.createElement('select'));
        for (const o of c.options ?? []) {
          const opt = document.createElement('option');
          opt.value = String(o.value);
          opt.textContent = o.label;
          opt.selected = o.value === v;
          s.appendChild(opt);
        }
      } else {
        const t = (input = document.createElement('input'));
        t.type = 'checkbox';
        t.checked = !!v;
      }
      input.id = id;
      const read = (): ParamValue => coerce(c, c.kind === 'toggle' ? (input as HTMLInputElement).checked : input.value);
      const apply = (final: boolean) => {
        const nv = read();
        val.textContent = fmtVal(c, nv);
        this.engine.params[i][c.key] = nv; // always the engine's current object, never a stale copy
        if (c.live) this.engine.chapters[i].set?.(c.key, nv);
        else if (final) this.schedule(i);
      };
      input.addEventListener('input', () => apply(c.kind !== 'range'));
      input.addEventListener('change', () => apply(true));

      // a select already shows its value, so it takes the value column as well
      row.append(name, input);
      if (c.kind === 'select') row.classList.add('xp__row--select');
      else row.appendChild(val);

      // the hint of the control under the pointer (or with focus) shows in the shared line below
      const hint = texPlain(c.hint ?? DEFAULT_HINT);
      row.addEventListener('pointerenter', () => (this.hintEl.innerHTML = hint));
      row.addEventListener('focusin', () => (this.hintEl.innerHTML = hint));
      row.addEventListener('pointerleave', () => (this.hintEl.textContent = DEFAULT_HINT));
      this.body.appendChild(row);
    });
  }

  /** A URL that reopens this chapter at this moment with these parameters. */
  private copyLink() {
    const e = this.engine, i = e.active;
    if (i < 0) return;
    const u = new URL(location.href);
    u.search = '';
    u.searchParams.set('ch', e.chapters[i].id);
    // rounded down and kept inside the chapter: a link copied in its last instant must not open the next one
    const local = Math.min(e.chapters[i].duration - 0.1, Math.max(0, e.T - e.starts[i]));
    u.searchParams.set('t', (Math.floor(local * 10) / 10).toFixed(1));
    u.searchParams.set('explore', '1');
    for (const [k, v] of Object.entries(e.params[i])) u.searchParams.set('p.' + k, String(v));
    const done = () => (this.status.textContent = 'link copied');
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(u.toString()).then(done, () => (this.hintEl.textContent = u.toString()));
    else this.hintEl.textContent = u.toString();
  }

  private reset() {
    const i = this.shown;
    if (i < 0) return;
    const params: Params = {};
    this.engine.params[i] = params;
    for (const c of this.controls(i)) if (c.live) this.engine.chapters[i].set?.(c.key, c.value, true);
    this.render(i);
    this.schedule(i);
  }

  private schedule(i: number) {
    this.dirty.add(i);
    clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.run(), 250);
  }

  /**
   * Rebuild every chapter whose parameters changed. One build runs at a time. A change that arrives
   * during a build marks its chapter dirty again, and the loop builds it once more with the latest values.
   */
  private async run() {
    if (this.busy) return;
    this.busy = true;
    this.el.classList.add('is-busy');
    while (this.dirty.size) {
      const i = this.dirty.values().next().value as number;
      this.dirty.delete(i);
      const say = (m: string) => { if (i === this.shown) this.status.textContent = m; };
      const t0 = performance.now();
      try {
        await this.engine.rebuild(i, say);
        say(`re-ran in ${Math.round(performance.now() - t0)} ms`);
        // live values persist across a rebuild
        const P = this.engine.params[i];
        for (const c of this.controls(i)) if (c.live && P[c.key] !== undefined) this.engine.chapters[i].set?.(c.key, P[c.key]);
      } catch (e: any) {
        console.error(e);
        say(`failed: ${e?.message ?? e}`);
      }
    }
    this.busy = false;
    this.el.classList.remove('is-busy');
  }
}
