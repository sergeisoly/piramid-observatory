import * as THREE from 'three/webgpu';
import type { Chapter, ChapterFactory, Hotspot, Readout } from '../engine/types';
import { TEAM_LABEL, TEAM_CHAPTERS, TITLES, PILLAR_OF, LOOP_SIDE, EVIDENCE, BEFORE_TRAINING_KIND } from '../engine/cards';
import { C, CSS, PILLAR_HUE, type RGB } from '../engine/palette';
import { PointField } from '../kit/points';
import { LineField } from '../kit/lines';
import { InkFill } from '../kit/ink';
import { mulberry32, gaussian, smooth } from '../lib/math';
import { ORDER } from './index';

/**
 * FINALE: the loop, then the evidence one question at a time. The flat map of the three teams from the prologue
 * returns, with dots travelling along its sides in the direction the loop runs (theory to tools, tools to data, data
 * to theory). Slide 9.1 shows the loop alone. Slides 9.2 to 9.6 each bring one question onto the board beside
 * it, with only that question's rows, and light the chapters those rows come from on the triangle:
 *   9.2 ideal allocation and predictions for trained networks (generality, mean field, frustration);
 *   9.3 readable (percolation probes, the eNTK's cliffs);
 *   9.4 changed by training (the eNTK before training; percolation's comparison once chapter 1 reports it);
 *   9.5 causal control (elicitation) and computational dependence (the parity bottleneck);
 *   9.6 bounded (coarse-graining), and the return to the prologue's opening question.
 * Notes keep interpretation limits that the board alone does not explain. Chapter 1 enters as clauses chosen by
 * what it reports (ch1Control).
 */
/** When each slide's caption comes in. */
const AT = [0.8, 7.4, 11.4, 15.6, 19.8, 24.0];
const DUR = 29.6;
const R = 7.2, Y0 = -3.4;
const INK = 0.54; // density of each team's print, as in the emblem (engine/mark.ts)
// seen from above, Data is at the top, Learning theory bottom right, the tools bottom left
const A: RGB = [R * Math.cos(Math.PI / 6), Y0, R * Math.sin(Math.PI / 6)]; // Learning theory
const B: RGB = [-R * Math.cos(Math.PI / 6), Y0, R * Math.sin(Math.PI / 6)]; // Interpretability applications
const Cv: RGB = [0, Y0, -R]; // Data models and validation
const MID: RGB = [0, Y0, R * 0.12]; // the center, where coarse-graining stands
const lerp3 = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** The questions of slides 9.2 to 9.6, in order: the group key, the heading, and the comparison it takes. */
type Q = 'theory' | 'readable' | 'trained' | 'causal' | 'used' | 'bounded';
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const QUESTIONS: [Q, string, string][] = [
  ['theory', 'What theory predicts', 'distinguish an ideal capacity allocation from predictions tested on trained networks'],
  ['readable', cap(EVIDENCE.readable), 'compare what each method finds with known structure'],
  ['trained', cap(EVIDENCE.trained), 'compare patterns recovered from the same network before and after training'],
  ['causal', 'Causal control', 'test whether a selected weight change alters behavior on new problems'],
  ['used', 'Computational dependence', 'trace what reaches a block and test what changes when features are edited'],
  ['bounded', 'How large can the error be?', 'Specify the output, the input range and which details can vary.'],
];
/** The groups of rows each slide puts on the board (9.1, the loop, none). */
const SLIDE_Q: Q[][] = [[], ['theory'], ['readable'], ['trained'], ['causal', 'used'], ['bounded']];

/**
 * One row of the map: the chapter, the question it answers, the status tag (stamp · qualifier), the claim, and what
 * it was compared with or what follows. They summarize the chapters' last slides. `slide: false` keeps a row off
 * the board when its caption does not discuss it.
 */
interface MapRow { id: string; q: Q; tag: string; claim: string; note: string; slide?: boolean }
const MAP: MapRow[] = [
  {
    id: 'generality', q: 'theory', tag: 'Reported by the authors · Ari Brill, Fig. 2',
    claim: 'In Brill’s model, sharing circuits across tasks becomes optimal above a capacity threshold',
    note: 'The curves optimize the capacity budget for power-law-distributed tasks under assumed circuit costs.',
  },
  {
    id: 'meanfield', q: 'theory', tag: 'Computed for this presentation',
    claim: 'Neurons trained separately against a frozen error map form a similar four-lobed pattern',
    note: 'In the product task, their input-weight projections and combined error map resemble those of the jointly trained network.',
  },
  {
    id: 'frustration', q: 'theory', tag: 'Computed for this presentation',
    claim: 'Decoder weights paired with small encoder weights scatter more in the narrower denoiser',
    note: "In the comparison network these weights stay near zero. Both the architecture and the task change, so the source of the scatter remains uncertain.",
  },
  {
    id: 'percolation', q: 'readable', tag: 'Reported by the authors · Brill and Carlson, Fig. 4a',
    claim: 'Linear probes predict the hidden variables more accurately from neuron activity than from raw inputs',
    note: 'Both kinds of probe predict variables on examples excluded from probe training.',
  },
  {
    id: 'entk', q: 'readable', tag: 'Computed for this presentation',
    claim: 'Two leading kernel subspaces align with known Fourier features across all input pairs',
    note: 'With the default settings used in this presentation, the subspaces overlap the Fourier features more than random subspaces of the same size.',
  },
  {
    id: 'entk', q: 'trained', tag: 'Computed for this presentation',
    claim: 'In the default run shown here, the second kernel group aligns more closely with the sum features after training',
    note: 'The overlap with the features in $a + b$ starts near the random baseline.',
  },
  {
    id: 'elicitation', q: 'causal', tag: 'Reported by the authors · Mack et al., Table 3',
    claim: 'A selected weight edit recovers most of the BigCodeBench gap in a password-locked 70B model',
    note: 'Table 3 reports pass rates of 0.43 locked, 0.52 locked plus the edit, and 0.54 with the password.',
  },
  {
    id: 'parity', q: 'used', tag: 'Reported by the authors · Mack et al., Figs. 1–2',
    claim: 'The selected features and their signed scores set the direction of the feedforward input',
    note: 'On three tasks, the target token outranks its comparison token more often after bottleneck edits than after autoencoder edits.',
  },
  {
    id: 'entk', q: 'used', tag: 'Reported by the authors · PIRAMID: Progress and Plans', slide: false,
    claim: 'Removing weight directions associated with sum features lowers modular-addition accuracy more than a random ablation',
    note: 'PIRAMID reports this follow-up against an equal-sized random removal.',
  },
  {
    id: 'finale', q: 'bounded', tag: 'Illustration',
    claim: 'Within the band, moving the prediction toward one edge increases its error at the other.',
    note: 'At x = 1 the possible outputs span ½. Any prediction must be at least ¼ away from one edge. The midpoint attains that minimum worst-case error.',
  },
  {
    id: 'scales', q: 'bounded', slide: false, tag: 'Illustration · toy tree',
    claim: 'The same shift has different effects on the branch mean and the whole-tree mean',
    note: 'The global mean changes by the shift times the fraction of nodes in the branch.',
  },
];

/**
 * The chapters each slide discusses: their lines on the triangle are lit, the others dimmed (9.1 lights none).
 * Percolation is lit on the training-comparison slide once it reports its before-training control.
 */
const LIT: string[][] = [[], ['generality', 'meanfield', 'frustration'], ['percolation', 'entk'], ['entk', 'percolation'], ['elicitation', 'parity'], ['scales']];

interface Stream {
  a: RGB; b: RGB; ca: RGB; cb: RGB; start: number; count: number; speed: number;
}

const finale: ChapterFactory = {
  id: 'finale',
  label: 'finale · the loop',
  async build(ctx) {
    const rnd = mulberry32(99), g = gaussian(rnd);
    const group = new THREE.Group();

    // One stream per side, in the direction of the loop. The way back (a mismatch) is said in words.
    const PER = 1500;
    const defs: [RGB, RGB, RGB, RGB][] = [
      [A, B, C.ice, C.amber], // theory → tools: what a tool should find
      [B, Cv, C.amber, C.violet], // tools → data: scored on a known answer
      [Cv, A, C.violet, C.ice], // data → theory: checks the prediction
    ];
    // each stream runs on the paper just outside its side, so that the side itself stays one clean line
    const OUT = 0.2;
    const shift = (a: RGB, b: RGB): [RGB, RGB] => {
      const mx = (a[0] + b[0]) / 2, mz = (a[2] + b[2]) / 2, len = Math.hypot(mx, mz), ox = (mx / len) * OUT, oz = (mz / len) * OUT;
      return [[a[0] + ox, Y0, a[2] + oz], [b[0] + ox, Y0, b[2] + oz]];
    };
    const streams: Stream[] = defs.map(([a, b, ca, cb], i) => { const [sa, sb] = shift(a, b); return { a: sa, b: sb, ca, cb, speed: 0.05, start: i * PER, count: PER }; });
    const total = streams.length * PER;
    const pts = new PointField({ count: total, size: 0.04 });
    const phase = new Float32Array(total), jx = new Float32Array(total), jz = new Float32Array(total), lane = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      phase[i] = rnd();
      const tight = 0.028; // a thin line of moving dots: wider, and a still frame reads it as a ragged edge
      jx[i] = g() * tight; jz[i] = g() * tight;
      lane[i] = 0.7 + rnd() * 0.6;
      pts.sizes[i] = 0.55 + rnd() * 1.1;
    }

    // the three prints (each the triangle scaled by 3/4 about the team's corner), in the order data, theory, tools
    const base: RGB[] = [Cv, A, B];
    const TEAMS = ['data', 'theory', 'tools'] as const;
    const prints = base.map((v, k) => new InkFill(base.map((u) => lerp3(v, u, 0.75)), [C.violet, C.ice, C.amber][k], INK));

    const frame = new LineField(4, { width: 1.6 });
    for (const [a, b] of [[A, B], [B, Cv], [Cv, A]] as [RGB, RGB][]) frame.push(a, b, C.white, 0.8);
    frame.commit();

    // An analytic example, independent of the fitted slope in the opening simulation.
    // For x in [0,1], every allowed output abx is between x/4 and 3x/4.
    const response = (x: number, y: number): RGB => [-6 + 10 * x, Y0, 4 - 11 * y];
    const band = new InkFill([response(0, 0), response(1, 0.25), response(1, 0.75)], C.cyan, 0.26);
    const axes = new LineField(3, { width: 1.5 });
    axes.push(response(0, 0), response(1.04, 0), C.white);
    axes.push(response(0, 0), response(0, 0.84), C.white);
    axes.push(response(1, 0.015), response(1, -0.015), C.white);
    axes.commit();
    const slopes = new LineField(6, { width: 2 });
    for (const slope of [0.25, 0.5, 0.75])
      slopes.push(response(0, 0), response(1, slope), slope === 0.5 ? C.white : C.cyan);
    slopes.polyline([response(1.015, 0.5), response(1.06, 0.5), response(1.06, 0.75), response(1.015, 0.75)], C.white, 0.75);
    slopes.commit();
    const responsePlot = new THREE.Group();
    responsePlot.add(band.object, axes.object, slopes.object);
    const teamMap = new THREE.Group();
    teamMap.add(frame.object, pts.object, ...prints.map(p => p.object));
    group.add(responsePlot, teamMap);

    /** A point beside the middle of a side, `out` world units outside the triangle. */
    const beside = (a: RGB, b: RGB, out: number): RGB => {
      const mx = (a[0] + b[0]) / 2, mz = (a[2] + b[2]) / 2, len = Math.hypot(mx, mz);
      return [mx + (mx / len) * out, Y0, mz + (mz / len) * out];
    };
    const teamInk: Record<string, string> = { data: CSS[PILLAR_HUE.data], theory: CSS[PILLAR_HUE.theory], tools: CSS[PILLAR_HUE.tools] };

    // The shared manifest keeps full-film numbering even in a partial build.
    const num = (id: string) => String(ORDER.indexOf(id as (typeof ORDER)[number])).padStart(2, '0');
    const name = (id: string) => `${num(id)} ${TITLES[id] ?? id}`;

    /** Chapter 1's comparison with the same network before training, once chapter 1 reports one (see BEFORE_TRAINING_KIND). */
    const ch1Control = () => ctx.film?.chapters.find((c) => c.id === 'percolation')?.results?.find((r) => r.kind === BEFORE_TRAINING_KIND && !!r.say);
    const trainedCaption = () => ch1Control()
      ? 'In our modular-addition experiment, training strengthens the kernel’s alignment with sum features. In the percolation runs, probes can already read latent values from networks at initialization.'
      : 'In our modular-addition experiment, input features already align with the kernel at initialization. The sum features align more strongly after training.';

    const ch1Row = (): MapRow | null => {
      const result = ch1Control();
      if (!result) return null;
      return { id: 'percolation', q: 'trained', tag: 'Computed for this presentation', claim: result.claim ?? result.say!, note: '' };
    };
    /** The whole map, in the order of QUESTIONS. */
    const mapRows = (): MapRow[] => {
      const extra = ch1Row();
      const rows: MapRow[] = [];
      for (const [q] of QUESTIONS) {
        for (const m of MAP) {
          if (m.q !== q) continue;
          // When chapter 1 supplies a before-training result, use its computed probe comparison here.
          rows.push(m.id === 'percolation' && q === 'readable' && extra ? {
            ...m,
            tag: 'Computed for this presentation',
            claim: 'On points held out from probe fitting, linear probes beat a baseline that predicts the mean',
            note: 'On the example tree, hidden-layer probes make smaller errors than probes on either raw inputs or the residual stream.',
          } : m);
        }
        if (q === 'trained' && extra) rows.push(extra);
      }
      return rows;
    };

    /** Slide on show at time t: 0 for 9.1 (the loop) to 5 for 9.6. */
    const slideAt = (t: number) => { let k = 0; while (k + 1 < AT.length && t >= AT[k + 1]) k++; return k; };
    /** Whether chapter id is lit on slide k: 1 lit, 0 neither (9.1), −1 dimmed. Chapter 1 is lit on 9.4 only with its comparison. */
    const chLit = (id: string, k: number) => (k === 0 ? 0 : LIT[k].includes(id) && (id !== 'percolation' || k !== 3 || !!ch1Control()) ? 1 : -1);
    /** How strongly a team's print is inked on slide k: full where the slide discusses one of its chapters. */
    const teamLit = (team: string, k: number) => {
      if (k === 0) return 1;
      return TEAM_CHAPTERS[team as 'data' | 'theory' | 'tools'].some((id) => chLit(id, k) > 0) ? 1 : 0.34;
    };

    const chapter: Chapter = {
      id: 'finale',
      pillar: 'frame',
      kicker: 'Summary',
      title: 'RESEARCH LOOP',
      board: true,
      caveat: 'grouped for this presentation',
      source: 'Introducing PIRAMID · LessWrong, July 2026',
      links: [
        { text: 'Introducing PIRAMID', url: 'https://www.lesswrong.com/posts/nbSJhbLERTZFeNxY7/introducing-piramid-physics-informed-research-for-ambitious' },
      ],
      duration: DUR,
      group,
      beats: [
        { at: AT[0], caption: 'The percolation generator gives researchers a tree of hidden variables with known values. A theory can also predict structure inside a network, as mean field does for groups of neurons. Both give researchers something specific to look for.' },
        { at: AT[1], caption: 'Brill asks which circuits an ideal learner would share. Mean-field theory predicts a distribution of neuron weights in a toy model. Frustration offers a possible explanation for disorder that survives training. Each claim calls for a different test.' },
        { at: AT[2], caption: 'A probe predicts hidden variables on points left out of its fit. The modular-addition experiment measures how closely kernel subspaces align with known Fourier waves across the full input grid.' },
        // its last sentence follows what chapter 1 reports (the film's chapters exist by the time it is read)
        { at: AT[3], get caption() { return trainedCaption(); } },
        { at: AT[4], caption: 'A weight edit recovers most of the lost coding performance in a deliberately password-locked model. A parity bottleneck sets each feedforward input’s direction using a sparse code. Interventions test which features matter for an answer.' },
        { at: AT[5], caption: 'Suppose the two weights multiply to a number between ¼ and ¾. At input 1, the network’s output lies in that same range. Our estimate of ½ can be off by at most ¼. If we use only some of a trained network’s features to predict its output, how far off could we be?' },
      ],
      // one fixed view, straight down: the figure does not move while its labels are read
      shots: [
        { t: 0, pos: [0, 46, 1.1], look: [0, Y0, 0], fov: 25 },
        { t: DUR, pos: [0, 46, 1.1], look: [0, Y0, 0], fov: 25 },
      ],
      update(t, api) {
        const k = slideAt(t), was = Math.max(0, k - 1), f = smooth(AT[k], AT[k] + 0.6, t);
        /** A value that moves from slide k − 1's to slide k's as slide k comes in. */
        const blend = (v: (s: number) => number) => (k === 0 ? v(0) : v(was) + (v(k) - v(was)) * f);

        const showBound = k === 5;
        responsePlot.visible = showBound;
        teamMap.visible = !showBound;
        if (showBound) {
          api.label('bound-title', response(0.5, 1.0), 'Possible network outputs<small>Illustration</small>', {
            align: 'center', size: 'lg', help: '<p>The shaded band gives the outputs allowed by the weight constraint at each input. The middle line predicts halfway between the lowest and highest possible outputs. The largest distance to either edge occurs at $x = 1$.</p>',
          });
          api.label('bound-y', response(0, 0.84), 'network output', { align: 'center', size: 'md' });
          api.label('bound-x', response(0.55, 0), 'input x', { align: 'below', dy: 4, size: 'md' });
          api.label('bound-zero', response(0, 0), '0', { align: 'right', dx: -5, size: 'sm' });
          api.label('bound-one', response(1, 0), '1', { align: 'below', dy: 4, size: 'sm' });
          api.label('bound-high', response(1.06, 0.75), '¾ x', { align: 'left', dx: 4, size: 'md', color: CSS.cyan });
          api.label('bound-error', response(1.06, 0.625), '¼ max error', { align: 'left', dx: 4, size: 'md' });
          api.label('bound-mid', response(1.06, 0.5), '½ x  ·  our estimate', { align: 'left', dx: 4, size: 'md' });
          api.label('bound-low', response(1, 0.25), '¼ x', { align: 'left', dx: 8, size: 'md', color: CSS.cyan });
          return;
        }

        // the loop's streams: full on the loop slide, quieter while the board is read
        const P = pts.positions, Cc = pts.colors;
        const on = smooth(2.0, 4.0, t) * blend((s) => (s === 0 ? 1 : 0.55));
        for (const s of streams) {
          const a = s.a, b = s.b;
          for (let i = s.start; i < s.start + s.count; i++) {
            const u = (phase[i] + t * s.speed * lane[i]) % 1;
            const env = Math.sin(Math.PI * u);
            const j3 = i * 3;
            P[j3] = a[0] + (b[0] - a[0]) * u + jx[i];
            P[j3 + 1] = Y0;
            P[j3 + 2] = a[2] + (b[2] - a[2]) * u + jz[i];
            const kk = 0.42 * on * (0.25 + 0.75 * env);
            Cc[j3] = (s.ca[0] + (s.cb[0] - s.ca[0]) * u) * kk;
            Cc[j3 + 1] = (s.ca[1] + (s.cb[1] - s.ca[1]) * u) * kk;
            Cc[j3 + 2] = (s.ca[2] + (s.cb[2] - s.ca[2]) * u) * kk;
          }
        }
        pts.commit(true, true, false);
        // the teams whose chapters the slide discusses keep their print; the others step back
        prints.forEach((p, i) => { p.uInk.value = INK * blend((s) => teamLit(TEAMS[i], s)); });

        // corners: the team and its chapters, each chapter lit or dimmed by the slide
        const a = smooth(0.6, 1.8, t);
        const chLine = (id: string) => {
          const l = chLit(id, k);
          return `<span class="chl${l > 0 ? ' is-on' : l < 0 ? ' is-off' : ''}"><b>${num(id)}</b> ${TITLES[id] ?? id}</span>`;
        };
        const lab = (team: 'data' | 'theory' | 'tools', title: string) => `<span class="caps">${title}</span>${TEAM_CHAPTERS[team].map(chLine).join('')}`;
        api.label('fC', Cv, lab('data', TEAM_LABEL.data), { color: CSS.violet, alpha: a, size: 'md', align: 'center', dy: -6 });
        api.label('fA', A, lab('theory', TEAM_LABEL.theory), { color: CSS.ice, alpha: a, size: 'md', align: 'left', dx: 10, dy: 8 });
        api.label('fB', B, lab('tools', TEAM_LABEL.tools.replace('Interpretability ', 'Interpretability<br>')), { color: CSS.amber, alpha: a, size: 'md', align: 'right', dx: -10, dy: 8 });
        // coarse-graining belongs to no single team and stands in the middle
        const mid = chLit('scales', k);
        api.label('f7', MID, `<span class="chl${mid > 0 ? ' is-on' : mid < 0 ? ' is-off' : ''}"><b>${num('scales')}</b> ${TITLES.scales}</span><small>across the teams</small>`, { color: CSS.white, alpha: a, size: 'md', align: 'middle', plate: true });
        // what passes along each side of the loop, in the prologue's words, with an arrow in the direction it runs;
        // the shared loop stays muted while a specific evidence board is read
        const side = smooth(2.2, 4.2, t) * blend((s) => (s === 0 ? 1 : 0));
        api.label('e1', beside(A, B, 0.55), `&larr;&ensp;${LOOP_SIDE.theoryTools}`, { alpha: side, align: 'below', dy: 2, size: 'md' });
        api.label('e2', beside(B, Cv, 0.4), `${LOOP_SIDE.toolsData}&ensp;&nearr;`, { alpha: side, align: 'right', size: 'md' });
        api.label('e3', beside(Cv, A, 0.4), `&searr;&ensp;${LOOP_SIDE.dataTheory}`, { alpha: side, align: 'left', size: 'md' });
        const back = smooth(4.4, 6.0, t) * blend((s) => (s === 0 ? 1 : 0));
        api.label('e4', beside(A, B, 0.55), LOOP_SIDE.back, { alpha: back, align: 'below', dy: 26, size: 'md', color: CSS.red });
      },
      // The map of the evidence: only rows for the slide's question appear on the board.
      readouts(t) {
        const k = slideAt(t), qs = SLIDE_Q[k];
        // the board fills as its question comes in; the loop slide (the first) has none
        const shown = t >= AT[k] + 0.25;
        const rows: Readout[] = [];
        let last: Q | null = null;
        for (const m of mapRows()) {
          const qi = QUESTIONS.find(([key]) => key === m.q)!;
          const head = (m.q !== last ? `<span class="q"><b>${qi[1]}</b>${qi[2]}</span>` : '') + `<span class="chn" style="--c:${teamInk[PILLAR_OF[m.id]] ?? CSS.white}">${name(m.id)}</span>`;
          last = m.q;
          rows.push({ label: m.claim, value: '', tag: m.tag, note: m.note, head, detail: !shown || !qs.includes(m.q) || m.slide === false });
        }
        return rows;
      },
      // arXiv ids become links on their own; the LessWrong posts carry their address
      endSources: [
        { label: 'Introducing PIRAMID', value: 'LessWrong, Jul 2026', url: 'https://www.lesswrong.com/posts/nbSJhbLERTZFeNxY7/introducing-piramid-physics-informed-research-for-ambitious' },
        { label: 'PIRAMID: Progress and Plans', value: 'LessWrong, Jul 2026', url: 'https://www.lesswrong.com/posts/T2REsZneix3bmAKtL/piramid-progress-and-plans' },
        { label: `${num('percolation')} Critical percolation`, value: 'arXiv:2606.20347' },
        { label: `${num('generality')} A Model for Scaling Laws of General Intelligence`, value: 'OpenReview, 2025', url: 'https://openreview.net/notes/edits/attachment?id=kaw40vezr3&name=pdf' },
        { label: `${num('meanfield')} Mean field sequence`, value: 'LessWrong, Apr 2026', url: 'https://www.lesswrong.com/posts/rduzFkTKx5pGKWKcL/mean-field-sequence-an-introduction' },
        { label: `${num('meanfield')} Width-robust learnability`, value: 'arXiv:2607.05735' },
        { label: `${num('frustration')} A tale of three theories`, value: 'LessWrong, Jan 2026', url: 'https://www.lesswrong.com/posts/siu22scEfuKxpSgfK/a-tale-of-three-theories-sparsity-frustration-and' },
        { label: `${num('entk')} Empirical NTK features`, value: 'arXiv:2510.00468' },
        { label: `${num('elicitation')} Eliciting latent behaviors`, value: 'arXiv:2606.29604' },
        { label: `${num('parity')} Parity bottleneck layers`, value: 'arXiv:2607.20652' },
        { label: `${num('scales')} Scale-aware interpretability`, value: 'arXiv:2602.05184' },
      ],

      // ── Notes: context that changes how the current comparison can be interpreted ──
      inspect(t) {
        if (t < 2.4) return [];
        const k = slideAt(t);
        const inside = (p: RGB, s = 0.84): [number, number, number] => [p[0] * s, p[1], p[2] * s];
        const notes: Record<string, Hotspot> = {
          loop: {
            id: 'loop',
            pos: beside(A, B, 0.55),
            dy: 12,
            dx: -240,
            title: 'What researchers learn from failed tests',
            body:
              '<p>Introducing PIRAMID imagines a method that recovers only nonlinear or distributed patterns where researchers expected clean hierarchical features. Such a mismatch would give researchers a reason to reconsider both what the network learned and what their validation setup can detect.</p>',
          },
          allocation: {
            id: 'allocation',
            pos: inside(Cv, 0.8),
            title: 'What the allocation assumes',
            body: '<p>Brill assumes a latent population of circuits and optimizes how to spend a fixed capacity budget. Whether training finds that allocation is a separate question.</p>',
          },
          theory: {
            id: 'theory',
            pos: inside(A),
            title: 'What frustration means',
            body: '<p>Frustration means competing learning constraints cannot all be satisfied. In Vaintrob’s denoising model, there are more feature directions than hidden-layer dimensions. These directions cannot all be mutually orthogonal. The post studies how the resulting constraints produce disorder in the trained weights.</p>',
          },
          data: {
            id: 'data',
            pos: inside(Cv, 0.8),
            title: k === 3 ? 'How the probe comparison is scored' : 'A decoder added by the researcher',
            body:
              k === 3
                ? '<p>At the ridge setting from the authors’ code, we report “no gain” for a group if at every site a majority of eligible bins have a median trained-to-initial error ratio of at least 0.95. A cutoff of 1.00 gives a mixed result. With stronger ridge regularization, most site-by-bin comparisons in the first run favor the trained network.</p>'
                : '<p>Researchers fit a separate linear decoder at each tree depth. The probe predicts latent values on points withheld from its fit, within the same tree. The network’s own prediction may use a different computation.</p>',
          },
          tools: {
            id: 'tools',
            pos: inside(B),
            title: 'How the kernel comparison is scored',
            body: '<p>In this presentation, the kernel is computed from the gradients of the model’s output scores with respect to its weights. The calculation uses every input pair, including training pairs, and requires no answer labels.</p><p>To score the overlap, we omit the nearly constant leading eigenvector. The known Fourier solution sets the size of each group. We measure overlap with the space spanned by each group. Lin identifies the groups from spectral gaps in her modulo-29 experiment.</p>',
          },
          elicitation: {
            id: 'elicitation',
            pos: inside(B),
            title: 'What the coding test compares',
            body: '<p>The 70B model was trained to underperform without the password. The authors selected one adapter on validation data and scored it on held-out coding problems. Their prose gives 0.46 for the locked baseline, while Table 3 gives 0.43. Either value implies recovery of most of the gap. Norm-matched random edits were evaluated on separate 8B tasks.</p>',
          },
          parity: {
            id: 'parity',
            pos: inside(B, 0.52),
            title: 'What the bottleneck and edits test',
            body: '<p>The decoder combines the selected directions using their signed scores and restores the input norm. Attention still uses the dense activation.</p>' +
              '<p>The edit test compares two token scores. The new indirect-object name must outrank the unchanged subject. For the pronoun and number tasks, the target must outrank the original answer. The models, layers and edit rules differ across methods.</p>',
          },
          scale: {
            id: 'scale',
            pos: response(0.79, 0.82),
            title: 'Why the error bound holds',
            body:
              '<p>The network outputs $abx$, and we estimate its output as $x/2$. We assume that the product $ab$ lies between ¼ and ¾. For inputs from 0 to 1, the largest possible error is $x/4$ at each input. It reaches ¼ at $x=1$.</p>' +
              '<p>For a trained network, first choose the output to predict and the inputs the explanation should cover. Then ask how much that output could vary among networks that agree on the features the explanation keeps. A bound on prediction error must hold across all the allowed differences in the omitted details.</p>',
          },
        };
        // the notes of the slide on show, in reading order: top to bottom, left to right
        const bySlide: string[][] = [['loop'], ['allocation', 'theory'], ['data', 'tools'], ['data'], ['elicitation', 'parity'], ['scale']];
        return bySlide[k].filter((id) => k !== 3 || id !== 'data' || !!ch1Control()).map((id) => notes[id]);
      },
    };
    return chapter;
  },
};

export default finale;
