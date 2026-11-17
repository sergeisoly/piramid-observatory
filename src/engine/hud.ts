import * as THREE from 'three/webgpu';
import katex from 'katex';
import type { Chapter, FigureSource, Hotspot, LabelOpts, Link, NoteDetails, Readout } from './types';
import { figureAssetURL } from './figure-image';
import { CSS, MARK, PILLAR_HUE, PILLAR_NAME, printed } from './palette';
import { esc, linkify, tex, texPlain } from './tex';
import { markSvg, cornerOf, CORNERS, VERT, CENTER } from './mark';
import { CARDS, TEAM_LABEL, pipsFor } from './cards';
import { CREDIT_HTML, REACH_HTML } from './credits';
import type { FigureViewport } from './viewport';
import { setLabelRoom } from './framing';

const pad2 = (n: number) => String(n).padStart(2, '0');
const ACRONYMS = new Set(['NTK', 'ENTK', 'PIRAMID', 'CPE', 'SOGI', 'MELBO', 'SAE', 'MLP', 'LLM', 'AI']);
/** Chapters give titles in capitals; the HUD sets them like the blog's headings ("Empirical NTK"). */
export const pretty = (s: string) =>
  s !== s.toUpperCase()
    ? s
    : s.toLowerCase().replace(/[a-z][a-z'’]*/g, (w, i) => (ACRONYMS.has(w.toUpperCase()) ? (w === 'entk' ? 'eNTK' : w.toUpperCase()) : i === 0 ? w[0].toUpperCase() + w.slice(1) : w));
const tc = (s: number) => `${pad2(Math.floor(s / 60))}:${pad2(Math.floor(s % 60))}`;

/**
 * "paper:" and "post:" mark a number that is quoted from the cited work and was not computed here.
 * The film sets the mark as a small stamp, so that it cannot be mistaken for an emphasized word.
 */
const prov = (html: string) =>
  html
    .replace(/<b>(paper|post):?<\/b>:?\s?/g, '<span class="src">$1</span>\u00a0')
    .replace(/(^|[\s(>])(paper|post):\s/g, (_, pre: string, w: string) => (pre === '(' ? `<span class="nb">(<span class="src">${w}</span></span>\u00a0` : `${pre}<span class="src">${w}</span>\u00a0`));

/**
 * Shared evidence grammar: "Computed for this presentation", "Reported by the authors · <named source>, <table/figure>",
 * or "Illustration · <optional qualifier>". Runtime location belongs in Notes only when it matters to the result.
 * A chapter writes the words as an ordinary label; the status becomes one small stamp, the same in every chapter
 * (styles.css, `.ev`). A named source or qualifier follows it in plain small text. The same stamp can be
 * written inside any label, caption or note as <span class="ev">…</span>; a panel row has `Readout.tag` for it.
 */
const EVIDENCE_WORDS = String.raw`computed\s+for\s+this\s+presentation|reported\s+by\s+the\s+authors|illustration`;
const EVIDENCE_TAG = new RegExp(String.raw`^\s*(${EVIDENCE_WORDS})\b(.*)$`, 'is');
const evidenceTag = (text: string) => {
  const m = EVIDENCE_TAG.exec(text);
  if (!m) return null;
  const rest = m[2].replace(/^\s*[·:,]?\s*/, '');
  return `<span class="ev">${m[1]}</span>${rest ? `<span class="ev__rest">${tex(rest)}</span>` : ''}`;
};

/**
 * A status stamp inside a label that does not start with one. The label keeps its own size and ink, and only the stamp
 * words become the stamp: after " · " ("large encoder weights · Computed for this presentation", the " · " goes),
 * or at the start of a later line (a panel title with "<small>Reported by the authors · Mack et al., Table 3</small>";
 * the " · " after the stamp goes, the qualifier stays as plain text). Maths is left alone.
 */
const STAMP_WORDS = `(${EVIDENCE_WORDS})\\b`;
const STAMP_AFTER = new RegExp(`\\s·\\s${STAMP_WORDS}`, 'gi');
const STAMP_LINE = new RegExp(`(<small[^>]*>|<br\\s*/?>)\\s*${STAMP_WORDS}(?:\\s*·\\s*)?`, 'gi');
const stampInside = (html: string) =>
  html.split(/(\$[^$]*\$)/).map((part, i) => (i % 2 ? part : part.replace(STAMP_LINE, '$1<span class="ev">$2</span> ').replace(STAMP_AFTER, ' <span class="ev">$1</span>'))).join('');

/** A hyphenated word ("2-hot", "password-locked") is kept on one line: its hyphen becomes a non-breaking one. Maths and tags are left alone. */
const noBreak = (html: string) =>
  html.split(/(\$[^$]*\$|<[^>]+>)/).map((part, i) => (i % 2 ? part : part.replace(/(\w)-(?=\w)/g, '$1\u2011'))).join('');

interface LabelEl {
  el: HTMLDivElement;
  text: string;
  used: boolean;
  /** The label states the status of the evidence (see evidenceTag). */
  tag: boolean;
  help: string;
  helpButton: HTMLButtonElement | null;
  figure: FigureSource | null;
  figureButton: HTMLButtonElement | null;
}

/** What the inspect card shows: a hotspot's explanation or a probed data element. */
export interface InspectContent {
  title: string; // plain text with $maths$
  rows?: Readout[];
  body?: string; // HTML with $maths$
  note?: string; // plain text with $maths$
}

/** A row that starts with "paper:" or "post:" is quoted from the cited work, not measured here. */
const QUOTED = /^\s*(?:<b>)?(paper|post)\b:?/;
const chip = (r: Readout) => (r.mark ? `<i class="mk" style="background:${esc(r.mark)}"></i>` : '');
/** A value that is a sentence rather than a number gets its own line, set flush left (it would wrap raggedly on the right). */
const prose = (r: Readout) => r.value.replace(/\$[^$]*\$/g, 'x').length > 44 || r.value.length > 62;
/**
 * A row's tag: the status stamp, then its qualifier in plain text ("Reported by the authors · Mack et al., Table 3"),
 * the same grammar as a tag on a figure (evidenceTag). A tag in other words is set whole as the stamp.
 */
const tagHtml = (tag: string) => {
  const m = EVIDENCE_TAG.exec(tag);
  if (!m) return `<b class="kind">${esc(tag)}</b>`;
  const rest = m[2].replace(/^\s*[·:,]?\s*/, '');
  return `<b class="kind">${esc(m[1])}</b>${rest ? `<span class="kind__rest">${esc(rest)}</span>` : ''}`;
};
/** A row's second line: the kind of evidence as a tag, then the note. */
const noteHtml = (r: Readout) => (r.tag ? tagHtml(r.tag) : '') + (r.note ? `<span class="nt">${prov(texPlain(r.note))}</span>` : '');
const rowHtml = (r: Readout) =>
  `<div class="row${prose(r) ? ' is-prose' : ''}">${r.head ? `<span class="h">${r.head}</span>` : ''}<span class="l">${chip(r)}${prov(texPlain(r.label))}</span><span class="v">${
    r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${texPlain(r.value)}</a>` : linkify(prov(texPlain(r.value)))
  }</span>${r.note || r.tag ? `<span class="n">${noteHtml(r)}</span>` : ''}</div>`;

/** The heads-up display: all crisp text lives in the DOM, layered over the WebGPU canvas. */
export class Hud {
  private root: HTMLElement;
  private reading!: HTMLElement;
  private figure!: HTMLElement;
  private figureSheet!: HTMLElement;
  private figureOverlay!: HTMLElement;
  compact = false;
  private readingBeforeNotes = 0;
  private $ = <T extends HTMLElement>(sel: string) => this.root.querySelector(sel) as T;
  private kicker!: HTMLElement;
  private index!: HTMLElement;
  private title!: HTMLElement;
  private source!: HTMLElement;
  private question!: HTMLElement;
  private card!: HTMLElement;
  private cardOn = false;
  private teleNote = 'Computed for this presentation';
  private teleHead = '';
  private order: string[] = []; // chapter ids in film order
  private capBox!: HTMLElement;
  private endEl!: HTMLElement;
  private endOn = false;
  private labelScale = 1;
  private figScale = 1;
  private viewport: FigureViewport | null = null;
  private teleSig = '';
  private teleH = 0; // height of the frame's foot that the Measured block takes in the film layout (0 = none)
  private headBottom = 80;
  private boardRows = false;
  private capTop = 0; // height of the frame's foot that the caption takes, in frame pixels (0 = no caption)
  /** Named works linked from the chapter's source line, captions, and notes. */
  private links: Link[] = [];
  private endSources: Readout[] = [];
  private caption!: HTMLElement;
  private eqBox!: HTMLElement;
  private eq!: HTMLElement;
  private tele!: HTMLElement;
  private teleRows: { h: HTMLElement; l: HTMLElement; v: HTMLElement; n: HTMLElement; row: HTMLElement; sh: string; sl: string; sv: string; sn: string }[] = [];
  private segs: HTMLElement[] = [];
  private currentChapter = -1;
  private fills: HTMLElement[] = [];
  private timecode!: HTMLElement;
  private stat!: HTMLElement;
  private playBtn!: HTMLElement;
  private labelLayer!: HTMLElement;
  private labels = new Map<string, LabelEl>();
  private hotLayer!: HTMLElement;
  private hots: HTMLButtonElement[] = [];
  private ring!: HTMLElement;
  private insp!: HTMLElement;
  private inspSig = '';
  private noteBox!: HTMLElement;
  private noteSig = '';
  private notePageKey = '';
  private noteBounds: readonly [left: number, bottom: number] | null = null;
  private inspW = 0;
  private inspH = 0;
  private curCaption = '';
  /** What the play button does now: "Play or pause", "Next slide" or "Finish animation" (its label and tooltip). */
  private nextLabel = '';
  private curEq: string | undefined = '';
  private lastTele = 0;
  private v = new THREE.Vector3();
  private helpTip!: HTMLElement;
  private helpAnchor: HTMLButtonElement | null = null;
  private helpPinned = false;
  private helpTimer: number | undefined;
  private frameW = 1600;
  private frameH = 900;
  private uiScale = 1;
  private sourceView!: HTMLElement;
  private sourceTrigger: HTMLButtonElement | null = null;
  private sourceInert = new Map<HTMLElement, boolean>();
  onSeek: (chapter: number, frac: number) => void = () => {};
  onToggle: () => void = () => {};
  onPrev: () => void = () => {};
  onSlides: () => void = () => {};
  onRate: () => void = () => {};
  onResetView: () => void = () => {};
  onHome: () => void = () => {};
  onSourceOpen: () => void = () => {};
  onExplore: () => void = () => {};
  onSound: () => void = () => {};
  /** The end card's buttons: start the film again (as slides or not), or open Explore on a chapter. */
  onReplay: (slides: boolean) => void = () => {};
  onEndExplore: () => void = () => {};
  /** A hotspot marker was clicked. */
  onHot: (id: string) => void = () => {};
  /** The inspect card's close button was clicked. */
  onInspectClose: () => void = () => {};
  /** The note's ‹ or › was clicked (dir −1 or +1). */
  onNoteStep: (dir: number) => void = () => {};
  /** The HUD root, for panels that live alongside it. */
  get element() {
    return this.root;
  }
  setViewChanged(changed: boolean) {
    (this.$('.view-reset') as HTMLButtonElement).hidden = !changed;
  }
  /** Keep the figure's projection independent of the scrolling text in compact windows. */
  setCompact(on: boolean) {
    if (on === this.compact) return;
    this.compact = on;
    this.root.classList.toggle('is-compact', on);
    if (on) this.figureSheet.prepend(this.stage);
    else this.root.before(this.stage);
    this.hideFigureHelp();
    this.setKeys(this.root.classList.contains('is-slides'));
  }
  /** Lay the HUD out at `lw` × `lh` logical pixels and scale it by `s` to fill the window. */
  setFrame(lw: number, lh: number, s: number) {
    this.frameW = lw; this.frameH = lh; this.uiScale = s;
    const st = (this.compact ? this.figureOverlay : this.root).style;
    const other = (this.compact ? this.root : this.figureOverlay).style;
    other.removeProperty('width');
    other.removeProperty('height');
    other.removeProperty('transform');
    st.width = `${lw.toFixed(2)}px`;
    st.height = `${lh.toFixed(2)}px`;
    st.transform = Math.abs(s - 1) < 1e-4 ? '' : `scale(${s.toFixed(5)})`;
    this.root.style.setProperty('--lh', `${lh.toFixed(1)}px`);
    // The scale itself: a label on the figure keeps a smallest size on screen (styles.css, .lbl), so that a
    // window smaller than the 1600 × 900 frame does not shrink it below what can be read.
    this.root.style.setProperty('--ui', s.toFixed(4));
    setLabelRoom(s, this.compact); // and the figure leaves room at its sides for labels that keep that size
    this.figure.tabIndex = this.compact && this.figure.scrollWidth > this.figure.clientWidth ? 0 : -1;
    if (!this.compact) {
      const head = this.$('.hud__head');
      this.headBottom = head.offsetTop + head.offsetHeight;
      this.capTop = this.curCaption ? this.root.offsetHeight - this.capBox.offsetTop : 0;
      this.teleH = this.teleSig ? this.root.offsetHeight - this.tele.offsetTop : 0;
    }
    this.measureNoteColumn();
  }

  constructor(root: HTMLElement, private stage: HTMLElement) {
    this.root = root;
    root.className = 'hud is-hidden';
    root.innerHTML = `
      <div class="hud__scrim"></div>
      <div class="figure-help-tip" id="figure-help-tip" role="tooltip" hidden></div>
      <div class="figure-source-view" hidden><section class="figure-source-dialog" role="dialog" aria-modal="true" aria-labelledby="figure-source-title"><header><h2 id="figure-source-title"></h2><button class="figure-source-close" aria-label="Close source figure">×</button></header><div class="figure-source-body" tabindex="0" role="region" aria-label="Source figures"></div><footer class="figure-source-credit"></footer></section></div>
      <main class="hud__reading">
        <header class="hud__head">
          <div class="head__text">
            <div class="head__line"><span class="head__n index"></span><h1 class="head__title hud__title"></h1><span class="head__q"></span></div>
            <div class="head__sub"><span class="head__kicker kicker-text"></span><span class="head__source hud__source"></span></div>
          </div>
        </header>
        <div class="hud__caption"><p></p></div>
        <div class="hud__plate">
          <p class="figure-scroll-hint">Scroll sideways to see the full figure.</p>
          <div class="hud__figure" role="region" aria-label="Research figure">
            <div class="hud__figure-sheet">
              <div class="hud__figure-overlay">
                <div class="hud__labels"></div>
                <div class="ring"></div>
                <div class="hud__hots"></div>
              </div>
            </div>
          </div>
        </div>
        <section class="hud__eq"><div class="eq"></div></section>
        <div class="hud__card" aria-hidden="true">
          <div class="card__top"></div>
          <div class="card__n"></div>
          <div class="card__body">
            <h2 class="card__title"></h2>
            <p class="card__line"></p>
            <div class="card__source"></div>
          </div>
          <div class="card__mark"></div>
        </div>
        <aside class="hud__side">
        <section class="panel hud__note" id="slide-notes" aria-labelledby="slide-note-title"><div class="note__head"><span class="note__n"></span><span class="note__title" id="slide-note-title"></span><span class="note__nav"><button class="note__prev" aria-label="Previous note" title="Previous note ([)">‹</button><span class="note__count"></span><button class="note__next" aria-label="Next note" title="Next note (])">›</button></span><button class="note__close" aria-label="Close notes" title="Close the notes (Esc)">×</button></div><div class="insp__body note__body" role="region" aria-label="Note text"></div><div class="rows note__rows" role="region" aria-label="Note measurements"></div></section>
          <section class="panel hud__tele"><div class="panel__tag"><span class="tele__word">Measured</span> <span class="note">in this browser</span></div><div class="rows"></div></section>
        </aside>
        <div class="hud__end" aria-hidden="true">
          <div class="end__mark"></div>
          <div class="end__body">
            <h2 class="end__title">The goal of PIRAMID</h2>
            <p class="end__text">PIRAMID aims to make AI safer by understanding what neural networks learn and how they use it to make decisions. The team has developed ways to test explanations in simple networks and to change specific behaviors in larger models.</p>
            <p class="end__text">The open question is whether we can understand a large model’s decisions well enough to anticipate when it will fail, even in situations we haven’t tested.</p>
            <div class="end__actions"><button class="end__slides">Read again as slides</button><button class="end__replay">Play as a film</button><button class="end__explore">Open chapter 1 with its notes</button></div>
            <p class="end__links">PIRAMID in its own words: <a href="https://princint.ai/piramid/" target="_blank" rel="noopener noreferrer">princint.ai/piramid</a> · <a href="https://www.lesswrong.com/posts/nbSJhbLERTZFeNxY7/introducing-piramid-physics-informed-research-for-ambitious" target="_blank" rel="noopener noreferrer">Introducing PIRAMID</a> · <a href="https://www.lesswrong.com/posts/T2REsZneix3bmAKtL/piramid-progress-and-plans" target="_blank" rel="noopener noreferrer">PIRAMID: Progress and Plans</a></p>
            <p class="end__by">${CREDIT_HTML}</p>${REACH_HTML ? `<p class="end__by end__reach">${REACH_HTML}</p>` : ''}
          </div>
          <div class="end__sources"><div class="panel__tag">Sources</div><div class="rows"></div></div>
        </div>
      </main>
      <footer class="hud__bar">
        <button class="overview" title="Return to the opening overview (Home)">Overview</button>
        <button class="prev" aria-label="Previous slide" title="Previous slide (←)"><svg viewBox="0 0 12 12"><path d="M10 1L1 6l9 5z"/></svg></button>
        <button class="play" aria-label="Play or pause"><svg viewBox="0 0 12 12" aria-hidden="true"><path class="i-pause" d="M2 1h3v10H2zM7 1h3v10H7z"/><path class="i-play" d="M2 1l9 5-9 5z"/><path class="i-finish" d="M1 1l7 5-7 5zM9 1h2v10H9z"/></svg></button>
        <div class="timecode">00:00 / 00:00</div>
        <div class="chapname"></div>
        <div class="track"></div>
        <button class="view-reset" hidden aria-label="Reset view" title="Reset view">Reset view</button>
        <button class="mode" title="Step through one slide at a time (S)">Slides</button>
        <button class="rate" title="Playback speed of the film, from 0.5× to 1.5× (keys < and >)" aria-label="Playback speed">1×</button>
        <button class="snd" title="Sound on or off (M)" aria-label="Sound" hidden><svg viewBox="0 0 14 12"><path d="M1 4h2.6L7 1v10L3.6 8H1z"/><path class="i-wave" d="M9 3.2c1.6 1.4 1.6 4.2 0 5.6M10.8 1.6c2.6 2.4 2.6 6.4 0 8.8" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><path class="i-off" d="M9.2 4l3.6 4M12.8 4l-3.6 4" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg></button>
        <button class="xbtn" title="Read the notes for this slide (E)" aria-controls="slide-notes" aria-expanded="false">Notes</button>
        <button class="kbtn" aria-label="Controls and keyboard shortcuts" aria-controls="keyboard-shortcuts" aria-expanded="false">?</button>
        <div class="hud__brand">Independent presentation · not affiliated with <a href="https://princint.ai/piramid/" target="_blank" rel="noopener noreferrer">PIRAMID</a></div>
      </footer>
      <div class="hud__keys" id="keyboard-shortcuts" role="note" aria-label="Controls and keyboard shortcuts"></div>
      <div class="insp"><button class="insp__close" aria-label="Close">×</button><div class="insp__title"></div><div class="insp__body"></div><div class="rows insp__rows"></div><div class="insp__body insp__note"></div></div>`;
    this.reading = this.$('.hud__reading');
    this.figure = this.$('.hud__figure');
    this.figureSheet = this.$('.hud__figure-sheet');
    this.figureOverlay = this.$('.hud__figure-overlay');
    this.figure.addEventListener('keydown', (event) => {
      if (this.figure.tabIndex < 0 || event.target !== this.figure || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      this.figure.scrollLeft += (event.key === 'ArrowLeft' ? -1 : 1) * this.figure.clientWidth * 0.8;
      event.preventDefault();
      event.stopPropagation();
    });
    this.kicker = this.$('.kicker-text');
    this.index = this.$('.index');
    this.title = this.$('.hud__title');
    this.source = this.$('.hud__source');
    this.caption = this.$('.hud__caption p');
    this.capBox = this.$('.hud__caption');
    this.question = this.$('.head__q');
    this.card = this.$('.hud__card');
    this.card.inert = true;
    this.$('.snd').addEventListener('click', () => this.onSound());
    this.stat = document.createElement('div');
    this.stat.className = 'stat';
    this.setKeys(false);
    const kb = this.$('.kbtn');
    const keys = this.$('.hud__keys');
    const setKeysOpen = (open: boolean) => {
      this.root.classList.toggle('is-keys', open);
      kb.setAttribute('aria-expanded', String(open));
    };
    kb.addEventListener('click', () => setKeysOpen(!this.root.classList.contains('is-keys')));
    const dismissKeysOutside = (event: Event) => {
      if (!this.root.classList.contains('is-keys')) return;
      const target = event.target;
      if (target instanceof Node && !kb.contains(target) && !keys.contains(target)) setKeysOpen(false);
    };
    // Capture also covers canvas gestures and controls that stop event propagation.
    document.addEventListener('pointerdown', dismissKeysOutside, true);
    document.addEventListener('click', dismissKeysOutside, true);
    this.endEl = this.$('.hud__end');
    this.endEl.inert = true;
    {
      const S = 320, PAD = 8;
      const lab = CORNERS.map((k) => {
        const [x, y] = VERT[k], below = (y * S + PAD + 16).toFixed(0);
        const st = k === 'data' ? `left:${(x * S + PAD).toFixed(0)}px;top:-8px;translate:-50% -100%;text-align:center` : k === 'theory' ? `left:${(x * S + PAD).toFixed(0)}px;top:${below}px;translate:-100% 0;text-align:right` : `left:${PAD}px;top:${below}px`;
        return `<span class="card__corner is-on" style="${st};--c:${CSS[PILLAR_HUE[k]]}">${TEAM_LABEL[k].replace(' and ', '<br>and ').replace('Interpretability ', 'Interpretability<br>')}</span>`;
      }).join('');
      this.$('.end__mark').innerHTML = markSvg({ size: S }) + lab;
    }
    this.$('.end__replay').addEventListener('click', () => this.onReplay(false));
    this.$('.end__slides').addEventListener('click', () => this.onReplay(true));
    this.$('.end__explore').addEventListener('click', () => this.onEndExplore());
    this.eqBox = this.$('.hud__eq');
    this.eq = this.$('.hud__eq .eq');
    this.tele = this.$('.hud__tele');
    this.timecode = this.$('.timecode');
    this.playBtn = this.$('.play');
    this.labelLayer = this.$('.hud__labels');
    this.hotLayer = this.$('.hud__hots');
    this.ring = this.$('.ring');
    this.insp = this.$('.insp');
    this.playBtn.addEventListener('click', () => this.onToggle());
    this.$('.prev').addEventListener('click', () => this.onPrev());
    this.$('.mode').addEventListener('click', () => this.onSlides());
    this.$('.rate').addEventListener('click', () => this.onRate());
    this.$('.view-reset').addEventListener('click', () => this.onResetView());
    this.$('.overview').addEventListener('click', () => this.onHome());
    this.$('.xbtn').addEventListener('click', () => this.onExplore());
    this.$('.insp__close').addEventListener('click', () => this.onInspectClose());
    this.$('.note__close').addEventListener('click', () => this.onInspectClose());
    this.$('.note__prev').addEventListener('click', () => this.onNoteStep(-1));
    this.$('.note__next').addEventListener('click', () => this.onNoteStep(1));
    this.noteBox = this.$('.hud__note');
    new ResizeObserver(() => this.measureNoteColumn()).observe(this.noteBox);
    this.helpTip = this.$('.figure-help-tip');
    this.sourceView = this.$('.figure-source-view');
    this.$('.figure-source-close').addEventListener('click', () => this.hideSourceFigure());
    this.sourceView.addEventListener('pointerdown', (e) => { if (e.target === this.sourceView) this.hideSourceFigure(); e.stopPropagation(); });
    for (const ev of ['pointerup', 'wheel', 'dblclick']) this.sourceView.addEventListener(ev, (e) => e.stopPropagation());
    this.helpTip.addEventListener('pointerenter', () => this.cancelHelpTimer());
    this.helpTip.addEventListener('pointerleave', () => this.deferHideHelp());
    this.helpTip.addEventListener('focusin', () => this.cancelHelpTimer());
    this.helpTip.addEventListener('focusout', () => this.deferHideHelp());
    for (const ev of ['pointerdown', 'pointerup', 'wheel', 'dblclick']) this.helpTip.addEventListener(ev, (e) => e.stopPropagation());
    window.addEventListener('pointerdown', (e) => {
      if (!this.sourceView.hidden && !this.$('.figure-source-dialog').contains(e.target as Node)) this.hideSourceFigure();
      if (this.helpAnchor && !this.helpAnchor.contains(e.target as Node) && !this.helpTip.contains(e.target as Node)) this.hideFigureHelp();
    }, true);
    window.addEventListener('keydown', (e) => {
      if (!this.sourceView.hidden) {
        if (e.key === 'Home' && !e.metaKey && !e.ctrlKey && !e.altKey) {
          this.onHome(); e.preventDefault(); e.stopImmediatePropagation(); return;
        }
        if (e.key === 'Escape') this.hideSourceFigure();
        if (e.key === 'Tab') {
          const focusable = [...this.sourceView.querySelectorAll<HTMLElement>('button, a[href], [tabindex="0"]')];
          const first = focusable[0], last = focusable.at(-1);
          if (e.shiftKey && document.activeElement === first) { last?.focus(); e.preventDefault(); }
          else if (!e.shiftKey && document.activeElement === last) { first?.focus(); e.preventDefault(); }
        }
        // Native up/down and Page keys scroll the focused document region; no key reaches the film.
        if (['Escape', 'ArrowLeft', 'ArrowRight'].includes(e.key)) e.preventDefault();
        e.stopImmediatePropagation();
        return;
      }
      if (e.key === 'Escape' && this.helpAnchor) {
        this.hideFigureHelp(); e.preventDefault(); e.stopImmediatePropagation();
      }
    }, true);
    for (const ev of ['pointerdown', 'wheel', 'dblclick']) this.noteBox.addEventListener(ev, (e) => e.stopPropagation());
    // a pinned card is interactive (links, text selection); keep its events from orbiting the camera
    for (const ev of ['pointerdown', 'wheel', 'dblclick']) this.insp.addEventListener(ev, (e) => e.stopPropagation());
  }

  show() {
    this.root.classList.remove('is-hidden');
    if (this.compact) this.reading.scrollTop = 0;
  }
  /** Hide the presentation layer and dismiss its overlays before showing the opening overview. */
  hide() {
    this.hideSourceFigure();
    this.hideFigureHelp();
    this.root.classList.remove('is-keys');
    this.$('.kbtn').setAttribute('aria-expanded', 'false');
    this.toggleChrome(false);
    this.root.classList.add('is-hidden');
  }
  focusTransport() {
    this.playBtn.focus({ preventScroll: true });
  }
  focusNotes() {
    const button = this.$<HTMLButtonElement>('.xbtn');
    if (button.disabled) this.focusTransport();
    else button.focus({ preventScroll: true });
  }
  toggleChrome(force?: boolean) {
    const minimal = this.root.classList.toggle('is-minimal', force);
    this.syncContentFocus();
    if (minimal) this.hideFigureHelp();
  }

  /** Only the content visible in the current presentation state can receive focus. */
  private syncContentFocus() {
    if (!this.sourceView.hidden) return;
    const minimal = this.root.classList.contains('is-minimal');
    this.$('.hud__bar').inert = minimal;
    this.card.inert = !this.cardOn || this.endOn;
    this.endEl.inert = !this.endOn;
    const hidden = minimal || this.cardOn || this.endOn;
    this.root.querySelectorAll<HTMLElement>('.hud__side, .hud__head, .hud__eq, .hud__hots, .hud__labels, .insp').forEach((el) => (el.inert = hidden));
  }

  /** Build the chapter track at the bottom. */
  setChapters(chapters: Chapter[]) {
    const track = this.$('.track');
    track.innerHTML = '';
    this.segs = [];
    this.fills = [];
    this.order = chapters.map((c) => c.id);
    chapters.forEach((c, i) => {
      const seg = document.createElement('button');
      seg.className = 'seg';
      seg.style.flexGrow = String(c.duration);
      seg.style.setProperty('--c', c.pillar === 'frame' ? 'var(--ink)' : MARK[PILLAR_HUE[c.pillar]]);
      seg.title = pretty(c.title);
      seg.setAttribute('aria-label', seg.title);
      seg.innerHTML = `<span class="seg__fill"></span><span class="seg__n">${pad2(i)}</span><span class="seg__name"><b>${pad2(i)}</b>${pretty(c.title)}</span>`;
      seg.addEventListener('click', (e) => {
        // Enter another chapter at its divider; click within the current chapter to scrub.
        if (i !== this.currentChapter || e.detail === 0) { this.onSeek(i, 0); return; }
        const r = seg.getBoundingClientRect();
        this.onSeek(i, Math.min(0.98, Math.max(0, (e.clientX - r.left) / r.width)));
      });
      track.appendChild(seg);
      this.segs.push(seg);
      this.fills.push(seg.querySelector('.seg__fill') as HTMLElement);
    });
    this.markActiveChapter(this.currentChapter);
  }

  private markActiveChapter(i: number) {
    this.segs.forEach((seg, k) => {
      seg.classList.toggle('is-active', k === i);
      if (k === i) seg.setAttribute('aria-current', 'true');
      else seg.removeAttribute('aria-current');
    });
  }

  setChapter(c: Chapter, i: number, n: number) {
    this.currentChapter = i;
    const hue = PILLAR_HUE[c.pillar], framed = c.pillar === 'frame';
    this.root.style.setProperty('--accent', CSS[hue]);
    this.root.style.setProperty('--accent-mark', framed ? 'var(--ink)' : MARK[hue]);
    // a chapter of one team is headed by the team's name alone; a framing chapter by its own word ("Prologue", "Summary")
    const kicker = framed ? c.kicker || PILLAR_NAME[c.pillar] : PILLAR_NAME[c.pillar], title = pretty(c.title), line = CARDS[c.id]?.line ?? '';
    const source = linkify(texPlain(c.source), c.links).replace(/ · /g, '\u00a0· ');
    this.links = c.links ?? [];

    // running head
    this.kicker.textContent = kicker;
    this.index.textContent = pad2(i);
    this.title.textContent = title;
    this.question.textContent = line;
    this.source.innerHTML = source;
    if (this.compact) this.figure.scrollLeft = 0;
    // opening card
    const card = this.card, q = <T extends HTMLElement>(sel: string) => card.querySelector(sel) as T;
    q('.card__n').textContent = pad2(i);
    q('.card__top').innerHTML = `Chapter ${i} of ${n - 1}<b>${esc(kicker)}</b>`;
    q('.card__title').textContent = title;
    q('.card__line').textContent = line;
    q('.card__source').innerHTML = source;
    const S = 400, PAD = 8; // the emblem's side and the margin markSvg leaves around it
    // corner labels sit outside the triangle, each on the outer side of its corner
    const lab = CORNERS.map((k) => {
      const [x, y] = VERT[k], on = !framed && k === c.pillar, below = (y * S + PAD + 18).toFixed(0);
      const st = k === 'data' ? `left:${(x * S + PAD).toFixed(0)}px;top:-8px;translate:-50% -100%;text-align:center` : k === 'theory' ? `left:${(x * S + PAD).toFixed(0)}px;top:${below}px;translate:-100% 0;text-align:right` : `left:${PAD}px;top:${below}px`;
      return `<span class="card__corner${on ? ' is-on' : ''}" style="${st};--c:${CSS[PILLAR_HUE[k]]}">${TEAM_LABEL[k]}</span>`;
    }).join('');
    // the card lights the chapter's team and nothing else; the teams and the map were introduced by the poster and the prologue
    // a chapter that stands across the teams (coarse-graining) gets its badge in the center, drawn as a team
    // chapter's pip on show: filled ink, its number in paper, a ring around it
    let mid = '';
    if (framed && c.id !== 'prologue' && c.id !== 'finale') {
      const x = (PAD + CENTER[0] * S).toFixed(1), y = (PAD + CENTER[1] * S).toFixed(1), r = Math.max(9.5, S * 0.028), sw = Math.max(1, S / 150);
      mid = `<circle cx="${x}" cy="${y}" r="${(r + 1).toFixed(1)}" fill="${CSS.white}" stroke="${CSS.white}" stroke-width="${sw.toFixed(2)}"/>` +
        `<text x="${x}" y="${(+y + r * 0.36).toFixed(1)}" text-anchor="middle" font-family="Inter Variable, Inter, sans-serif" font-size="${(r * 1.02).toFixed(1)}" font-weight="700" fill="${printed([])}">${pad2(i)}</text>` +
        `<circle cx="${x}" cy="${y}" r="${(r + 5).toFixed(1)}" fill="none" stroke="${CSS.white}" stroke-width="${(sw * 0.8).toFixed(2)}"/>`;
    }
    q('.card__mark').innerHTML = markSvg({ size: S, focus: cornerOf(c.pillar), solo: true, pips: pipsFor(c.id, this.order) }).replace('</svg>', `${mid}</svg>`) + lab;
    this.root.classList.toggle('is-board', !!c.board);
    // The closing chapter's board is its map of the evidence, whose rows each carry their own status.
    this.teleNote = c.caveat ?? (c.board ? 'by chapter' : 'Computed for this presentation');
    this.teleHead = '';
    this.notePageKey = '';
    this.endSources = c.endSources ?? [];
    this.markActiveChapter(i);
    this.$('.chapname').textContent = title;
    this.curCaption = '\u0000';
    this.curEq = '\u0000';
    const head = this.$('.hud__head');
    head.classList.remove('is-in');
    void head.offsetWidth;
    head.classList.add('is-in');
    // the head may take two or three lines (a long source wraps): the figure starts below it
    this.headBottom = head.offsetTop + head.offsetHeight;
  }

  /** Reserve the right column while Notes are open. */
  syncCol() {
    const open = this.root.classList.contains('is-explore');
    const changed = open !== this.notesColumnVisible;
    this.root.classList.toggle('is-col', open);
    if (this.compact && changed) {
      if (open) this.readingBeforeNotes = this.reading.scrollTop;
      else this.reading.scrollTop = this.cardOn || this.endOn ? 0 : this.readingBeforeNotes;
    }
  }
  /** Actual authored Notes content, distinct from a reading session retained across empty slides. */
  get notesColumnVisible() {
    return this.root.classList.contains('is-col');
  }
  /** Frame-pixel y where the running head ends. */
  headEnd() {
    return this.headBottom;
  }
  private measureNoteColumn() {
    if (!this.notesColumnVisible || !this.noteBox.classList.contains('is-in')) {
      this.noteBounds = null;
      return;
    }
    const side = this.noteBox.parentElement!;
    this.noteBounds = this.compact ? null : [side.offsetLeft, side.offsetTop + side.offsetHeight];
    for (const region of this.noteBox.querySelectorAll<HTMLElement>('.note__body, .note__rows')) {
      if (region.scrollHeight > region.clientHeight + 1) region.tabIndex = 0;
      else region.removeAttribute('tabindex');
    }
  }
  /** The occupied part of the desktop Notes column, in frame pixels. */
  get noteColumnBounds() {
    return this.noteBounds;
  }
  /** Reserve the chapter's board column even while its rows are hidden between slides. */
  get reservesBoard() {
    return this.root.classList.contains('is-board');
  }

  /**
   * Show or hide the chapter's opening card. While it is up, the rest of the HUD (running head,
   * caption, readouts) stays out of the way; the figure itself is blank paper at that moment.
   */
  setCard(on: boolean) {
    if (on === this.cardOn) return;
    this.cardOn = on;
    this.card.setAttribute('aria-hidden', String(!on));
    this.root.classList.toggle('is-card', on);
    this.syncContentFocus();
    if (on) {
      if (this.compact) this.reading.scrollTop = 0;
      // restart the card's entrance
      this.card.classList.remove('is-in');
      void this.card.offsetWidth;
      this.card.classList.add('is-in');
    }
  }

  /** The film has stopped at its end: show the end card over the blank paper. */
  setEnd(on: boolean) {
    if (on === this.endOn) return;
    this.endOn = on;
    this.root.classList.toggle('is-end', on);
    this.endEl.setAttribute('aria-hidden', String(!on));
    this.syncContentFocus();
    if (on) {
      if (this.compact) this.reading.scrollTop = 0;
      const sources = this.endSources;
      (this.endEl.querySelector('.end__sources .rows') as HTMLElement).innerHTML = sources.map(rowHtml).join('');
      (this.endEl.querySelector('.end__sources') as HTMLElement).hidden = sources.length === 0;
    }
  }

  /**
   * Tell the labels how large the figure is drawn relative to how it was composed. A figure that is shown
   * smaller (Explore gives part of the frame to its column) gets labels a little smaller, so that
   * neighboring labels do not run into each other.
   */
  setFigureScale(s: number) {
    this.figScale = s;
    const ls = (s >= 1 ? 1 : Math.max(0.74, s)) * (this.viewport?.scale ?? 1);
    if (Math.abs(ls - this.labelScale) < 0.004) return;
    this.labelScale = ls;
    this.labelLayer.style.setProperty('--ls', ls.toFixed(3));
    this.hotLayer.style.setProperty('--ls', ls.toFixed(3));
  }

  setViewport(view: FigureViewport, w: number, h: number) {
    this.viewport = view;
    const b = view.bounds;
    const clip = view.changed ? `inset(${b[1]}px ${w - b[2]}px ${h - b[3]}px ${b[0]}px)` : '';
    this.labelLayer.style.clipPath = clip;
    this.hotLayer.style.clipPath = clip;
    this.labelLayer.style.setProperty('--vs', String(view.scale));
    this.setFigureScale(this.figScale);
  }

  /** Top edge of the caption block in frame pixels, or the frame's height while there is no caption. */
  captionTop(lh: number) {
    return this.capTop > 0 ? lh - this.capTop : lh;
  }
  /** Top edge of the Measured block in the film layout, or the frame's height while it is not there. */
  readoutsTop(lh: number) {
    return this.teleH > 0 ? lh - this.teleH : lh;
  }

  setCaption(html: string) {
    if (html === this.curCaption) return;
    this.curCaption = html;
    const p = this.caption;
    p.classList.remove('is-in');
    void p.offsetWidth;
    p.innerHTML = linkify(tex(noBreak(html)), this.links); // captions carry no quoted numbers, so "the post: …" keeps its colon
    if (html) {
      p.classList.add('is-in');
      if (this.compact && !this.notesColumnVisible) this.reading.scrollTop = 0;
    }
    // how much of the frame's foot the caption takes: the figure is kept clear of it (see Engine.safeArea)
    this.capTop = html ? this.root.offsetHeight - this.capBox.offsetTop : 0;
  }

  setEquation(src: string | undefined, note?: string) {
    const key = src ? `${src}\u0001${note ?? ''}` : undefined;
    if (key === this.curEq) return;
    this.curEq = key;
    if (!src) {
      this.eqBox.classList.remove('is-in');
      return;
    }
    this.eq.innerHTML = katex.renderToString(src, { displayMode: true, throwOnError: false, strict: false }) + (note ? `<p class="eq__note">${texPlain(note)}</p>` : '');
    this.eqBox.classList.remove('is-in');
    void this.eqBox.offsetWidth;
    this.eqBox.classList.add('is-in');
  }

  setReadouts(all: Readout[] | undefined, now: number) {
    if (now - this.lastTele < 0.08) return;
    this.lastTele = now;
    const rows = this.$('.hud__tele .rows');
    // the film shows the main rows only; Explore mode adds the ones a chapter marks as detail
    const list = this.root.classList.contains('is-col') ? all : all?.filter((r) => !r.detail);
    const n = list?.length ?? 0;
    this.tele.classList.toggle('is-in', n > 0);
    // The heading says where the numbers come from. A slide that shows only numbers quoted from the paper
    // or the post must not stand under "Measured in this browser".
    const quoted = n > 0 && list!.every((r) => QUOTED.test(r.label) || QUOTED.test(r.value));
    const src = quoted ? (list!.some((r) => /^\s*(?:<b>)?post\b/.test(r.label) || /^\s*(?:<b>)?post\b/.test(r.value)) ? 'post' : 'paper') : '';
    // the closing board is a map of the evidence (its numbers are in the notes): each row tags its own status
    const head = quoted ? `Quoted\u0001from the ${src}` : this.root.classList.contains('is-board') ? `Map of the evidence\u0001${this.teleNote}` : `Measured\u0001${this.teleNote}`;
    if (head !== this.teleHead) {
      this.teleHead = head;
      const [word, note] = head.split('\u0001');
      (this.tele.querySelector('.tele__word') as HTMLElement).textContent = word;
      (this.tele.querySelector('.note') as HTMLElement).textContent = note;
    }
    const board = n > 0 && this.root.classList.contains('is-board');
    if (board !== this.boardRows) { this.boardRows = board; this.root.classList.toggle('has-board', board); }
    // In the film layout the block stands at the foot of the frame, right of the caption. It is as tall as its
    // rows and their second lines, and the figure is kept clear of it (see Engine.safeArea). Measured only
    // when the rows change.
    const sig = this.root.classList.contains('is-col') || board ? '' : (list ?? []).map((r) => `${r.label}\u0001${r.note ?? ''}\u0001${r.tag ?? ''}`).join('\u0002');
    const resize = sig !== this.teleSig;
    this.teleSig = sig;
    while (this.teleRows.length < n) {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = '<span class="h"></span><span class="l"></span><span class="v"></span><span class="n"></span>';
      rows.appendChild(row);
      this.teleRows.push({ row, h: row.querySelector('.h')!, l: row.querySelector('.l')!, v: row.querySelector('.v')!, n: row.querySelector('.n')!, sh: '', sl: '', sv: '', sn: '' });
    }
    this.teleRows.forEach((r, i) => {
      const it = list?.[i];
      r.row.style.display = it ? '' : 'none';
      if (!it) return;
      // compare the source strings, so the HTML (and any KaTeX) is rebuilt only when a value changes
      const sl = it.label + '\u0001' + (it.mark ?? '');
      if (r.sl !== sl) { r.sl = sl; r.l.innerHTML = chip(it) + prov(texPlain(it.label)); r.row.classList.toggle('is-quoted', QUOTED.test(it.label) || QUOTED.test(it.value)); }
      if (r.sv !== it.value) { r.sv = it.value; r.v.innerHTML = prov(texPlain(it.value)); r.row.classList.toggle('is-prose', prose(it)); }
      const note = (it.tag ?? '') + '\u0001' + (it.note ?? '');
      if (r.sn !== note) { r.sn = note; r.n.innerHTML = noteHtml(it); }
      const head = it.head ?? '';
      if (r.sh !== head) { r.sh = head; r.h.innerHTML = head; }
    });
    if (resize) this.teleH = sig ? this.root.offsetHeight - this.tele.offsetTop : 0;
  }

  /** The speed button highlights any rate other than 1×. */
  setRate(r: number) {
    const b = this.$('.rate');
    b.textContent = `${r}×`;
    b.classList.toggle('is-on', r !== 1);
  }
  /** The list of keys behind the "?" button; slides mode steps where the film plays. */
  private setKeys(slides: boolean) {
    const keys: [string, string][] = [
      ...(slides ? [['→ or space', 'finish the animation, then next slide'], ['←', 'previous slide']] : [['space', 'play or pause'], ['← →', 'previous or next chapter']]) as [string, string][],
      ...(slides ? [] : [['< >', 'slower or faster']]) as [string, string][],
      ['E', 'open or close the notes'],
      ['[ ]', 'previous or next note'],
      ['D', 'open additional context, when available'],
      ['S', 'switch between slides and film'],
      ['Home', 'return to the opening overview'],
      ['F', 'enter or leave fullscreen'],
      ['H', 'hide or show the controls'],
      ['M', 'sound on or off'],
      ['Esc', 'close help or source figure, then notes'],
    ];
    const box = this.$('.hud__keys');
    const viewKeys = this.compact
      ? 'Swipe sideways across a wide figure. With a mouse, hold Alt while scrolling over a chart to zoom. Use Reset view to restore it.'
      : 'Scroll over a chart to zoom. Drag a 3D object to rotate it, or a magnified chart to pan. Double-click or use Reset view to restore the view.';
    box.innerHTML = keys.map(([k, v]) => `<kbd>${esc(k)}</kbd><span>${esc(v)}</span>`).join('') +
      `<p class="keys__note">${viewKeys}</p>` +
      '<p class="keys__note">Point at data to read values. A small ? explains graphical marks. A numbered marker opens the note on that part of the figure.</p>';
    box.appendChild(this.stat);
  }

  /** The sound button appears once the score exists, and shows whether the score is muted. */
  setSound(muted: boolean) {
    const b = this.$('.snd');
    b.hidden = false;
    b.classList.toggle('is-off', muted);
  }
  /** Slides mode on or off: the bar gets a back button, the play button steps, the key hints change. */
  setSlidesMode(on: boolean) {
    this.root.classList.toggle('is-slides', on);
    const m = this.$('.mode');
    m.textContent = on ? 'Film' : 'Slides';
    m.title = on ? 'Play continuously (S)' : 'Step through one slide at a time (S)';
    this.timecode.setAttribute('aria-live', on ? 'polite' : 'off');
    this.setKeys(on);
  }

  setProgress(active: number, frac: number, T: number, total: number, playing: boolean, slide?: { index: number; count: number; beforeHold: boolean }) {
    this.fills.forEach((f, i) => {
      f.style.transform = `scaleX(${i < active ? 1 : i === active ? frac : 0})`;
    });
    const s = slide ? `slide ${slide.index + 1} / ${slide.count}` : `${tc(T)} / ${tc(total)}`;
    if (this.timecode.textContent !== s) this.timecode.textContent = s;
    this.playBtn.classList.toggle('is-playing', playing);
    const firstSlide = slide?.index === 0, lastSlide = !!slide && slide.index === slide.count - 1;
    const prev = this.$('.prev'), disabled = String(firstSlide);
    if (prev.getAttribute('aria-disabled') !== disabled) prev.setAttribute('aria-disabled', disabled);
    const next = lastSlide ? 'Last slide' : slide?.beforeHold ? (playing ? 'Finish animation' : 'Play animation') : slide ? 'Next slide' : playing ? 'Pause' : this.endOn ? 'Replay' : 'Play';
    if (next !== this.nextLabel) {
      this.nextLabel = next;
      this.playBtn.setAttribute('aria-label', next);
      this.playBtn.setAttribute('aria-disabled', String(lastSlide));
      this.playBtn.title = lastSlide ? next : `${next} (${slide ? '→' : 'Space'})`;
    }
  }

  setStat(s: string) {
    if (this.stat.textContent !== s) this.stat.textContent = s;
  }

  // ───────────── world-anchored labels ─────────────
  private showSourceFigure(source: FigureSource, trigger: HTMLButtonElement) {
    this.hideFigureHelp();
    this.sourceTrigger = trigger;
    const title = this.$('#figure-source-title');
    title.textContent = source.title;
    const body = this.$('.figure-source-body');
    body.replaceChildren();
    const images = source.images?.length ? source.images : [{ asset: source.asset, alt: source.alt }];
    for (const image of images) {
      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = figureAssetURL(image.asset); img.alt = image.alt; img.decoding = 'async';
      const caption = document.createElement('figcaption');
      if (image.caption) { const text = document.createElement('span'); text.textContent = image.caption; caption.appendChild(text); }
      const full = document.createElement('a');
      full.href = img.src; full.target = '_blank'; full.rel = 'noopener'; full.textContent = 'Open full-resolution image';
      caption.appendChild(full); figure.append(img, caption); body.appendChild(figure);
    }
    const credit = this.$('.figure-source-credit');
    credit.replaceChildren();
    const text = document.createElement('span'); text.textContent = source.credit;
    const link = document.createElement('a'); link.href = source.sourceURL; link.target = '_blank'; link.rel = 'noopener'; link.textContent = 'Source';
    credit.append(text, link);
    for (const child of this.root.children) {
      if (child === this.sourceView || !(child instanceof HTMLElement)) continue;
      this.sourceInert.set(child, child.inert); child.inert = true;
    }
    this.sourceView.hidden = false;
    body.scrollTop = 0;
    this.onSourceOpen();
    this.$('.figure-source-close').focus();
  }
  hideSourceFigure() {
    if (this.sourceView.hidden) return;
    this.sourceView.hidden = true;
    for (const [element, inert] of this.sourceInert) element.inert = inert;
    this.sourceInert.clear();
    this.syncContentFocus();
    const trigger = this.sourceTrigger;
    this.sourceTrigger = null;
    if (trigger?.isConnected) trigger.focus();
  }
  private cancelHelpTimer() {
    if (this.helpTimer !== undefined) window.clearTimeout(this.helpTimer);
    this.helpTimer = undefined;
  }
  private deferHideHelp() {
    this.cancelHelpTimer();
    this.helpTimer = window.setTimeout(() => {
      this.helpTimer = undefined;
      const active = document.activeElement;
      if (!this.helpPinned && active !== this.helpAnchor && !this.helpTip.contains(active) && !this.helpTip.matches(':hover')) this.hideFigureHelp();
    }, 120);
  }
  /** Figure reading help has its own state and never selects a Notes page or a data probe. */
  hideFigureHelp() {
    this.cancelHelpTimer();
    this.helpAnchor?.setAttribute('aria-expanded', 'false');
    this.helpAnchor?.removeAttribute('aria-describedby');
    this.helpAnchor = null;
    this.helpPinned = false;
    this.helpTip.hidden = true;
  }
  private showFigureHelp(button: HTMLButtonElement, pinned = false) {
    this.cancelHelpTimer();
    if (this.helpAnchor !== button) this.hideFigureHelp();
    this.helpAnchor = button;
    this.helpPinned = pinned;
    button.setAttribute('aria-expanded', 'true');
    button.setAttribute('aria-describedby', 'figure-help-tip');
    this.helpTip.innerHTML = linkify(tex(button.dataset.help ?? ''), this.links);
    this.helpTip.hidden = false;
    this.placeFigureHelp();
  }
  private placeFigureHelp() {
    const button = this.helpAnchor;
    if (!button) return;
    const label = button.closest<HTMLElement>('.lbl');
    if (!label || label.style.display === 'none' || !button.isConnected || this.cardOn || this.endOn) { this.hideFigureHelp(); return; }
    if (this.compact) {
      this.helpTip.style.removeProperty('transform');
      this.helpTip.style.removeProperty('max-width');
      this.helpTip.style.removeProperty('max-height');
      return;
    }
    const root = this.root.getBoundingClientRect(), box = button.getBoundingClientRect(), s = this.uiScale;
    const ax = (box.left - root.left) / s, ay = (box.top - root.top) / s;
    const right = this.root.classList.contains('is-col') ? Math.min(this.frameW - 12, this.viewport?.bounds[2] ?? this.frameW - 480) : this.frameW - 12;
    // A panned or clipped heading cannot leave its help floating over an unrelated figure.
    if (box.right < root.left || ax > right || ay < 0 || ay > this.frameH - 96 || (this.viewport?.changed && !this.viewport.contains(ax + box.width / s / 2, ay + box.height / s / 2))) { this.hideFigureHelp(); return; }
    const tip = this.helpTip;
    tip.style.maxWidth = `${Math.max(180, Math.min(340, right - 24))}px`;
    tip.style.maxHeight = `${Math.max(100, this.frameH - 130)}px`;
    const w = tip.offsetWidth, h = tip.offsetHeight, gap = 10;
    let x = ax + box.width / s + gap, y = ay + box.height / s + gap;
    if (x + w > right) x = ax - gap - w;
    if (y + h > this.frameH - 96) y = ay - gap - h;
    x = Math.max(12, Math.min(right - w, x));
    y = Math.max(12, Math.min(this.frameH - 96 - h, y));
    x = this.keepLeftOfNote(x, y, w);
    tip.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  }

  private keepLeftOfNote(x: number, y: number, width: number) {
    const note = this.noteBounds, gap = 10;
    return note && y < note[1] + gap ? Math.min(x, note[0] - gap - width) : x;
  }

  /** World position → frame pixels; false if it is behind the camera. Leaves the result in this.v. */
  private toScreen(cam: THREE.PerspectiveCamera, w: number, h: number, pos: THREE.Vector3 | ArrayLike<number>) {
    const v = this.v;
    if ((pos as THREE.Vector3).isVector3) v.copy(pos as THREE.Vector3);
    else v.set((pos as ArrayLike<number>)[0], (pos as ArrayLike<number>)[1], (pos as ArrayLike<number>)[2]);
    v.project(cam);
    if (v.z > 1 || v.z < -1) return false;
    v.x = (v.x * 0.5 + 0.5) * w;
    v.y = (-v.y * 0.5 + 0.5) * h;
    if (this.viewport) [v.x, v.y] = this.viewport.project(v.x, v.y);
    return true;
  }
  beginLabels() {
    for (const l of this.labels.values()) l.used = false;
  }
  label(cam: THREE.PerspectiveCamera, w: number, h: number, key: string, pos: THREE.Vector3 | [number, number, number], text: string, o: LabelOpts = {}) {
    const alpha = o.alpha ?? 1;
    if (alpha <= 0.01) return;
    if (!this.toScreen(cam, w, h, pos)) return;
    // A label's pixel offset was chosen against the figure at its composed size. It follows the figure's
    // scale, so that two labels a scene lined up with offsets stay in line when the figure is enlarged.
    const scale = this.figScale * (this.viewport?.scale ?? 1);
    const x = this.v.x + (o.dx ?? 0) * scale, y = this.v.y + (o.dy ?? 0) * scale;
    if (x < -200 || x > w + 200 || y < -100 || y > h + 100) return;
    let l = this.labels.get(key);
    if (!l) {
      const el = document.createElement('div');
      el.className = 'lbl';
      this.labelLayer.appendChild(el);
      l = { el, text: '', used: true, tag: false, help: '', helpButton: null, figure: null, figureButton: null };
      this.labels.set(key, l);
    }
    l.used = true;
    // a label that states the status of the evidence is set as the film's one stamp for it, whatever size and color
    // the chapter asked for
    if (l.text !== text) {
      l.text = text;
      const tag = evidenceTag(text);
      l.tag = tag !== null;
      l.el.innerHTML = `<span class="lbl__text">${tag ?? tex(stampInside(text))}</span>`;
      if (l.helpButton) l.el.appendChild(l.helpButton);
      if (l.figureButton) l.el.appendChild(l.figureButton);
    }
    const help = o.help ?? '';
    if (help && !l.helpButton) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'figure-help'; button.textContent = '?';
      button.setAttribute('aria-controls', 'figure-help-tip'); button.setAttribute('aria-expanded', 'false');
      button.addEventListener('pointerenter', () => this.showFigureHelp(button, this.helpAnchor === button && this.helpPinned));
      button.addEventListener('pointerleave', () => this.deferHideHelp());
      button.addEventListener('focus', () => this.showFigureHelp(button, this.helpAnchor === button && this.helpPinned));
      button.addEventListener('blur', () => this.deferHideHelp());
      button.addEventListener('click', (e) => { e.stopPropagation(); if (this.helpAnchor === button && this.helpPinned) this.hideFigureHelp(); else this.showFigureHelp(button, true); });
      for (const ev of ['pointerdown', 'pointerup', 'wheel', 'dblclick']) button.addEventListener(ev, (e) => e.stopPropagation());
      l.helpButton = button; l.el.appendChild(button);
    } else if (!help && l.helpButton) {
      if (this.helpAnchor === l.helpButton) this.hideFigureHelp();
      l.helpButton.remove(); l.helpButton = null;
    }
    if (l.helpButton) {
      l.helpButton.dataset.help = help;
      l.helpButton.setAttribute('aria-label', `How to read ${text.replace(/<[^>]*>/g, ' ').replace(/\$/g, '').trim()}`);
      if (help !== l.help && this.helpAnchor === l.helpButton) this.showFigureHelp(l.helpButton, this.helpPinned);
    }
    l.help = help;
    l.figure = o.figure ?? null;
    if (l.figure && !l.figureButton) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'figure-source-action';
      button.setAttribute('aria-haspopup', 'dialog');
      button.addEventListener('click', (e) => { e.stopPropagation(); if (l!.figure) this.showSourceFigure(l!.figure, button); });
      for (const ev of ['pointerdown', 'pointerup', 'wheel', 'dblclick']) button.addEventListener(ev, (e) => e.stopPropagation());
      l.figureButton = button; l.el.appendChild(button);
    } else if (!l.figure && l.figureButton) { l.figureButton.remove(); l.figureButton = null; }
    if (l.figureButton) l.figureButton.textContent = l.figure?.actionLabel ?? 'View source figure';
    const el = l.el;
    el.className = (l.tag ? `lbl lbl--tag lbl--${o.align ?? 'left'}` : `lbl lbl--${o.size ?? 'sm'} lbl--${o.align ?? 'left'}${o.plate ? ' lbl--plate' : ''}`) + (help ? ' lbl--help' : '');
    el.style.color = l.tag ? '' : o.color ?? '';
    el.style.opacity = String(alpha);
    el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
    el.style.display = '';
  }
  endLabels() {
    for (const l of this.labels.values()) if (!l.used) l.el.style.display = 'none';
    this.placeFigureHelp();
  }

  // ───────────── Explore mode: hotspots, probe ring, inspect card ─────────────
  /** Draw one numbered marker per hotspot (null or [] removes them). `active` is the pinned one. */
  setHotspots(cam: THREE.PerspectiveCamera, w: number, h: number, list: Hotspot[] | null, active: string | null, order?: readonly string[]) {
    const n = list?.length ?? 0;
    while (this.hots.length < n) {
      const b = document.createElement('button');
      b.className = 'hot';
      b.addEventListener('click', (e) => { e.stopPropagation(); this.onHot(b.dataset.id ?? ''); });
      b.addEventListener('pointerdown', (e) => e.stopPropagation());
      this.hotLayer.appendChild(b);
      this.hots.push(b);
    }
    this.hots.forEach((b, i) => {
      const hs = list?.[i];
      if (!hs || !this.toScreen(cam, w, h, hs.pos)) { b.style.display = 'none'; return; }
      b.style.display = '';
      const scale = this.figScale * (this.viewport?.scale ?? 1);
      b.style.transform = `translate3d(${(this.v.x + (hs.dx ?? 0) * scale).toFixed(1)}px, ${(this.v.y + (hs.dy ?? 0) * scale).toFixed(1)}px, 0)`;
      // the tooltip is plain text: drop the dollars and the commonest TeX wrappers
      if (b.dataset.id !== hs.id) { b.dataset.id = hs.id; b.title = hs.title.replace(/\$/g, '').replace(/\{,\}/g, ',').replace(/\\(text|mathrm|bar|hat)\s*/g, '').replace(/[{}\\]/g, ''); b.setAttribute('aria-label', b.title); }
      const label = String((order ? order.indexOf(hs.id) : i) + 1);
      if (b.textContent !== label) b.textContent = label;
      b.classList.toggle('is-on', hs.id === active);
    });
  }
  /** Markers shown while the notes are closed are drawn faintly (see Inspector.frame). */
  setPeek(on: boolean) {
    if (this.hotLayer.classList.contains('is-peek') !== on) this.hotLayer.classList.toggle('is-peek', on);
  }
  /** Ring the probed element, or hide the ring with null. */
  setRing(cam: THREE.PerspectiveCamera, w: number, h: number, pos: ArrayLike<number> | null) {
    if (!pos || !this.toScreen(cam, w, h, pos)) { this.ring.style.display = 'none'; return; }
    this.ring.style.display = 'block';
    this.ring.style.transform = `translate3d(${this.v.x.toFixed(1)}px, ${this.v.y.toFixed(1)}px, 0)`;
  }
  /**
   * Show the inspect card next to a world position. A pinned card takes pointer events (links,
   * selection, close button); a hover card does not, so it never gets between the pointer and the figure.
   */
  showInspect(cam: THREE.PerspectiveCamera, w: number, h: number, pos: ArrayLike<number>, c: InspectContent, pinned: boolean, dx = 0, dy = 0) {
    if (!this.toScreen(cam, w, h, pos)) { this.hideInspect(); return; }
    const ax = this.v.x + dx, ay = this.v.y + dy;
    const sig = `${c.title}\u0001${c.body ?? ''}\u0001${c.note ?? ''}\u0001${(c.rows ?? []).map((r) => `${r.label}\u0002${r.value}\u0002${r.note ?? ''}`).join('\u0003')}`;
    const el = this.insp;
    if (sig !== this.inspSig) {
      this.inspSig = sig;
      // reading order: title, the explanation, the values, a closing note
      (el.children[1] as HTMLElement).innerHTML = texPlain(c.title);
      (el.children[2] as HTMLElement).innerHTML = c.body ? linkify(tex(c.body), this.links) : '';
      (el.children[3] as HTMLElement).innerHTML = (c.rows ?? []).map(rowHtml).join('');
      (el.children[4] as HTMLElement).innerHTML = c.note ? `<p>${texPlain(c.note)}</p>` : '';
      el.style.display = 'block';
      this.inspW = el.offsetWidth;
      this.inspH = el.offsetHeight;
    }
    el.style.display = 'block';
    el.classList.toggle('is-pinned', pinned);
    // to the right of and below the anchor by default; flipped where that would leave the frame
    const gap = 16, bw = this.inspW, bh = this.inspH;
    let x = ax + gap, y = ay + gap;
    if (x + bw > w - 12) x = ax - gap - bw;
    if (y + bh > h - 96) y = ay - gap - bh;
    x = Math.max(12, Math.min(w - bw - 12, x));
    y = Math.max(12, Math.min(h - bh - 12, y));
    x = this.keepLeftOfNote(x, y, bw);
    el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  }
  /**
   * Show a hotspot's explanation in the right column, in place of the Measured panel. It is long
   * text about a whole panel, so it must not sit on the figure the way a probe card does.
   * `n` is the number on the hotspot's marker.
   */
  showNote(c: InspectContent, n: number | null, pageKey: string, k = 1, N = 1) {
    const sig = `${n}\u0001${k}/${N}\u0001${c.title}\u0001${c.body ?? ''}\u0001${(c.rows ?? []).map((r) => `${r.label}\u0002${r.value}\u0002${r.note ?? ''}`).join('\u0003')}`;
    // The inspector supplies slide + page identity. Live text, values and marker counts can
    // change on the same page without taking away the reader's place.
    const newPage = pageKey !== this.notePageKey;
    this.notePageKey = pageKey;
    const body = this.noteBox.querySelector('.note__body') as HTMLElement, rows = this.noteBox.querySelector('.note__rows') as HTMLElement;
    const bodyScroll = body.scrollTop, rowsScroll = rows.scrollTop;
    const changed = sig !== this.noteSig;
    if (changed) {
      this.noteSig = sig;
      const box = this.noteBox;
      (box.querySelector('.note__n') as HTMLElement).textContent = n === null ? '' : String(n);
      box.classList.toggle('is-unnumbered', n === null);
      (box.querySelector('.note__count') as HTMLElement).textContent = `${k} / ${N}`;
      (box.querySelector('.note__nav') as HTMLElement).hidden = N < 2;
      box.querySelector('.note__prev')?.setAttribute('aria-disabled', String(k <= 1));
      box.querySelector('.note__next')?.setAttribute('aria-disabled', String(k >= N));
      (box.querySelector('.note__title') as HTMLElement).innerHTML = texPlain(c.title);
      (box.querySelector('.note__body') as HTMLElement).innerHTML = c.body ? linkify(tex(c.body), this.links) : '';
      (box.querySelector('.note__rows') as HTMLElement).innerHTML = (c.rows ?? []).map(rowHtml).join('');
    }
    const opening = !this.noteBox.classList.contains('is-in');
    if (opening) {
      this.noteBox.classList.add('is-in');
      this.root.classList.add('is-noting');
    }
    if (changed || opening) this.measureNoteColumn();
    // Replacing a paragraph can reset its native scroll position even when it is the same page.
    // Restore after fitting, which may itself change the available scroll range.
    body.scrollTop = newPage ? 0 : bodyScroll;
    rows.scrollTop = newPage ? 0 : rowsScroll;
    if (this.compact && opening) this.noteBox.scrollIntoView({ block: 'start' });
  }
  /** Optional authored prose uses its own title and has no imposed introduction or table. */
  showDetailsPage(details: NoteDetails, k: number, N: number, pageKey: string) {
    this.showNote(details, null, pageKey, k, N);
  }
  /** A page of the notes: the numbers behind the slide, measured here or quoted from the source. */
  showNumbersPage(rows: Readout[], k: number, N: number, pageKey: string) {
    // the closing board's page is its whole map of the evidence, every question at once
    if (this.root.classList.contains('is-board')) {
      this.showNote({ title: 'This presentation’s evidence map', body: '', rows } as InspectContent, null, pageKey, k, N);
      return;
    }
    const quoted = rows.length > 0 && rows.every((r) => QUOTED.test(r.label) || QUOTED.test(r.value));
    const title = quoted ? 'Reported measurements' : 'Measurements';
    this.showNote({ title, rows } as InspectContent, null, pageKey, k, N);
  }
  /** Whether the end card is up. */
  get endShown() {
    return this.endOn;
  }
  hideNote() {
    if (!this.noteSig) return;
    this.noteSig = '';
    this.notePageKey = '';
    this.noteBounds = null;
    this.noteBox.classList.remove('is-in');
    this.root.classList.remove('is-noting');
  }
  hideInspect() {
    if (this.insp.style.display !== 'none') this.insp.style.display = 'none';
    this.inspSig = '';
  }
}
