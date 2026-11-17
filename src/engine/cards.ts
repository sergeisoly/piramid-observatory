import type { Corner } from './mark';
import type { Pillar } from './types';

/**
 * What a chapter's opening card says, and where the chapter sits in the program.
 *
 * `line` introduces the research question and stays in the running head. The figures show which
 * part of that question the experiments test. A card is on screen for about five seconds, so title and
 * line together have at most twelve words.
 */
export interface Card {
  line?: string;
  /** Seconds the card is held before the chapter starts. 0 = no card. */
  hold?: number;
}

export const CARDS: Record<string, Card> = {
  prologue: { hold: 0 },
  percolation: { line: 'How can we test whether interpretability methods recover hidden structure?' },
  generality: { line: 'When does limited capacity favor computation shared across tasks?' },
  meanfield: { line: 'How do individual neurons learn together in a wide network?' },
  frustration: { line: "Why do some trained weights scatter like noise?" },
  entk: { line: "Can a network’s sensitivity to weight changes reveal learned patterns?" },
  elicitation: { line: 'How can we uncover capabilities a model doesn’t normally show?' },
  parity: { line: 'Can sparse representations make a network’s computations easier to trace?' },
  scales: { line: 'How much can omitted detail change a prediction?' },
  finale: { line: 'Which differences between networks matter for their predictions?' },
};

/**
 * Plain-language summaries of the three teams in Introducing PIRAMID. The poster and prologue print these.
 * Known data-generating patterns provide a reference for tests, not proof of what a network learns or uses.
 */
export const TEAM_ROLE: Record<Corner, string> = {
  data: 'models data and task structure',
  theory: 'predicts how networks learn from data',
  tools: 'builds methods to inspect neural networks',
};

/**
 * What passes along each side of the loop, in the direction the loop runs, and the way back. The prologue draws the
 * loop with these sentences and the finale redraws it with the same ones. The way back follows the post: "Tool
 * failures, in turn, reveal where theories or data models are incomplete."
 */
export const LOOP_SIDE = {
  theoryTools: 'theory predicts which patterns to look for',
  toolsData: 'check recovered patterns<br>against known data patterns',
  dataTheory: 'compare observed learning<br>with theoretical predictions',
  back: 'mismatches guide changes to theories, tests, or data',
};

/** The finale groups results by the question that their comparison can answer. */
export const EVIDENCE = {
  readable: 'readable',
  trained: 'changed by training',
  used: 'used',
  bounded: 'bounded',
};

/** The finale uses the percolation chapter's own result for its before-training comparison. */
export const BEFORE_TRAINING_KIND = 'before-training control';

/** Seconds a chapter's opening card is held, unless the card says otherwise. */
export const CARD_HOLD = 4.4;

/** The chapters of each team, in film order: the emblem draws one pip per chapter. */
export const TEAM_CHAPTERS: Record<Corner, string[]> = {
  data: ['percolation', 'generality'],
  theory: ['meanfield', 'frustration'],
  tools: ['entk', 'elicitation', 'parity'],
};

/** Short names for the three teams, as the emblem's corner labels print them. */
export const TEAM_LABEL: Record<Corner, string> = {
  data: 'Data models and validation',
  theory: 'Learning theory',
  tools: 'Interpretability applications',
};

/** Pips for the emblem while chapter `id` is on show: every earlier chapter filled, this one ringed. */
export function pipsFor(id: string, order: string[]) {
  const i = order.indexOf(id);
  const out: Partial<Record<Corner, { n: number; done: number; at: number; labels: string[] }>> = {};
  (Object.keys(TEAM_CHAPTERS) as Corner[]).forEach((c) => {
    const ids = TEAM_CHAPTERS[c];
    out[c] = {
      n: ids.length,
      done: ids.filter((x) => order.indexOf(x) >= 0 && order.indexOf(x) < i).length,
      at: ids.indexOf(id),
      labels: ids.map((x) => String(Math.max(0, order.indexOf(x))).padStart(2, '0')),
    };
  });
  return out;
}

/** The part of the program each chapter belongs to, known before the chapter is built (the poster needs it). */
export const PILLAR_OF: Record<string, Pillar> = {
  prologue: 'frame', percolation: 'data', generality: 'data', meanfield: 'theory', frustration: 'theory',
  entk: 'tools', elicitation: 'tools', parity: 'tools', scales: 'frame', finale: 'frame',
};

/** Chapter names for the poster's table of contents until the chapters are built (the built chapter's title then replaces them). */
export const TITLES: Record<string, string> = {
  percolation: 'Critical percolation', generality: 'General capabilities', meanfield: 'Mean field', frustration: 'Frustration',
  entk: 'Empirical NTK', elicitation: 'Causal elicitation', parity: 'Parity bottleneck',
  scales: 'Coarse-graining', finale: 'Research loop',
};
