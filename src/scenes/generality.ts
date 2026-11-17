import * as THREE from 'three/webgpu';
import type { Chapter, ChapterFactory, FigureSource, FrameApi, Hotspot, NoteDetails } from '../engine/types';
import { loadFigureImage } from '../engine/figure-image';
import { C, CSS, type RGB } from '../engine/palette';
import { LineField } from '../kit/lines';
import { smooth } from '../lib/math';

/**
 * Ari Brill's capacity-allocation theory, rather than a trained-network experiment.
 * The schematic has no quantitative box scale. Every plotted scientific mark is
 * an unmodified raster of the September 16, 2025 primary PDF, rendered at 300 dpi:
 *   figure2-absolute: Fig. 2 top left; figure2-share: Fig. 2 bottom right;
 *   figure2-full: all four Fig. 2 panels; figure3: both Fig. 3 panels.
 * No numeric points were recovered from pixels, and no optimization is rerun.
 * The full figure is available independently of Notes from the attribution label.
 */
const SOURCE_URL = 'https://openreview.net/notes/edits/attachment?id=kaw40vezr3&name=pdf';
const WORK = 'A Model for Scaling Laws of General Intelligence';
const FIG2: FigureSource = {
  asset: 'figures/generality/figure2-full.png',
  title: `${WORK} · Figure 2`,
  actionLabel: 'Brill · Figure 2',
  alt: 'Four original panels plot absolute general degrees of freedom, loss, general capacity fraction, and task-frequency-weighted general DOF fraction against model capacity. Colors encode c/D. Points joined by dotted lines are numerically optimized allocations; solid curves are analytical approximations.',
  credit: `Ari Brill, “${WORK}” (revised September 16, 2025), Figure 2, p. 11. Original scientific plots. The main view pairs the upper-left and lower-right panels; this view includes all four. These are results of an idealized capacity-allocation model, not trained neural networks.`,
  sourceURL: SOURCE_URL,
};
const FIG3: FigureSource = {
  asset: 'figures/generality/figure3.png',
  title: `${WORK} · Figure 3`,
  actionLabel: 'Brill · Figure 3',
  alt: 'Original numerical allocation results show the task-frequency-weighted general DOF fraction against alpha on the left and c/D on the right, for three gamma values. The two vertical scales differ.',
  credit: `Ari Brill, “${WORK}” (revised September 16, 2025), Figure 3, p. 12. Both original panels and their original axes are preserved. The two vertical scales differ; visual slopes do not measure the relative effects of the two interventions.`,
  sourceURL: SOURCE_URL,
};

type V3 = [number, number, number];
// This camera sees 1600 × 900 design pixels at z=0; the engine raises its optical center by 4.5%.
const DIST = 26.14, FOV = 38, UNIT = DIST * Math.tan(FOV * Math.PI / 360) / 450;
const sp = (x: number, y: number): V3 => [(x - 800) * UNIT, -(y - 409.5) * UNIT, 0];
const T1 = 0.3, T2 = 1.65, T3 = 2.45, DUR = 4.45;
const visibility = (t: number) => {
  // Default slide holds are next.at − 0.05; a completed slide must remain unmixed there.
  // Fade through paper so unrelated axes and labels never become one overprinted graph.
  const enter = smooth(0.1, T1, t);
  return [
    enter * (1 - smooth(T2, T2 + 0.18, t)),
    smooth(T2 + 0.24, T2 + 0.48, t) * (1 - smooth(T3, T3 + 0.18, t)),
    smooth(T3 + 0.24, T3 + 0.48, t),
  ] as const;
};

const generality: ChapterFactory = {
  id: 'generality',
  label: 'general capabilities · optimal capacity allocation across tasks',
  async build() {
    const group = new THREE.Group();
    const lines = new LineField(140, { width: 2 });
    group.add(lines.object);
    const [absolute, share, diversity] = await Promise.all([
      loadFigureImage({ ...FIG2, asset: 'figures/generality/figure2-absolute.png' }, { width: 440 * UNIT, height: 430 * UNIT }),
      loadFigureImage({ ...FIG2, asset: 'figures/generality/figure2-share.png' }, { width: 470 * UNIT, height: 430 * UNIT }),
      loadFigureImage(FIG3, { width: 1120 * UNIT, height: 470 * UNIT }),
    ]);
    absolute.object.position.set(...sp(520, 425));
    share.object.position.set(...sp(1080, 425));
    diversity.object.position.set(...sp(800, 418));
    group.add(absolute.object, share.object, diversity.object);
    const label = (api: FrameApi, id: string, x: number, y: number, text: string, alpha: number, opts: Parameters<FrameApi['label']>[3] = {}) => {
      if (alpha > 0.004) api.label(id, sp(x, y), text, { align: 'middle', size: 'md', alpha, ...opts });
    };
    const details: NoteDetails[] = [
      {
        title: 'The allocation problem',
        body: '<p>Brill assigns each task a distinct manifold of the same dimension $D$. Task frequencies follow $p_k\\propto k^{-(1+\\alpha)}$. A task-specific circuit supplies effective degrees of freedom (DOF) only to its task; these can represent basis functions, prototypes, or memorized examples and need not be human-interpretable.</p>' +
          '<p>A general circuit supplies the same expected DOF to every task. Brill assumes declining marginal efficiency, $dn_G/dN_G=\\epsilon_0 n_G^{-\\gamma}$, so the total cost of the general allocation is $N_G=m n_G^{1+\\gamma}$, where $m=[\\epsilon_0(1+\\gamma)]^{-1}$. The predictor minimizes frequency-weighted loss subject to $N=\\sum_k n_k+N_G$. Task loss scales as $\\max(1,n_k+n_G)^{-c/D}$. A task with zero allocated DOF still has its baseline loss. Here $c$ measures expressivity and $D$ is task dimension.</p>' +
          '<p>Brill assumes unlimited training data and computation. A real optimizer may find a different allocation. The cost and loss laws are specified in Sections 2.1–2.2 and 2.4.</p>',
      },
      {
        title: 'Absolute amount and relative share',
        body: '<p>The right panel measures $\\langle n_G/n\\rangle=\\sum_k p_k\\,n_G/(n_k+n_G)$, weighted by task frequency $p_k$. Brill’s Figure 2 fixes $\\alpha=1$, $\\epsilon_0=0.01$, and $\\gamma=0.5$, while colors vary the task approximation exponent $c/D$. The theory has infinitely many ranked tasks. For both figures, Brill optimized 5,000 specialist allocations and one shared allocation, penalizing deviations from the capacity budget.</p>' +
          '<p>The complete Figure 2 also shows loss and the general capacity fraction $N_G/N$. A pronounced change in the slope of the loss curve requires $c/D>\\alpha$. The ideal predictor has unlimited training data and computation. Connecting the threshold to a network size would require an empirical mapping from capacity to parameter count.</p>',
      },
      {
        title: 'What makes shared computation worthwhile?',
        body: '<p>Figure 3 holds capacity at $N=10^5$ and the efficiency prefactor at $\\epsilon_0=0.01$. The left fixes $c/D=1$ and varies $\\alpha$ in $p_k\\propto k^{-(1+\\alpha)}$; smaller $\\alpha$ gives rare tasks more weight within the same infinite task set. The right fixes $\\alpha=1$ and varies $c/D$; larger expressivity $c$ at fixed task dimension $D$ lets the same DOF approximate a task more accurately. Each color is a different efficiency-decay exponent $\\gamma$.</p>' +
          '<p>Testing this explanation in learned models requires measuring the task distribution, relating capacity and DOF to network quantities, identifying shared circuits, and checking whether training reaches the assumed optimum. Source: Brill, Sections 4.5–4.7.</p>',
      },
    ];

    return {
      id: 'generality',
      pillar: 'data',
      kicker: 'DATA MODELS AND VALIDATION',
      title: 'GENERAL CAPABILITIES',
      source: `Ari Brill · ${WORK} · 2025`,
      links: [{ text: WORK, url: SOURCE_URL }],
      duration: DUR,
      group,
      beats: [
        {
          at: T1,
          caption: 'Ari Brill asks how a predictor with a fixed capacity budget can minimize loss across tasks, weighted by their frequency. Specialized circuits help one task each. General circuits add the same expected degrees of freedom (DOF) to every task, and each additional general DOF costs more.',
        },
        {
          at: T2,
          caption: 'At small capacities, Brill’s model favors specialists for the most frequent tasks. At a threshold, the optimal shared allocation jumps from zero to a positive amount. Its task-weighted share peaks early and then falls, even as the number of general DOF keeps growing.',
        },
        {
          at: T3,
          caption: 'At fixed capacity, giving rare tasks more weight sharply increases the task-weighted share of general DOF. Better approximation per DOF produces a smaller rise across the range shown. Brill suggests that a more expressive predictor can afford to specialize in more tasks. Testing the model’s predictions in trained networks would require identifying shared circuits and tracking their contribution across model sizes.',
        },
      ],
      shots: [
        { t: 0, pos: [0, 0, DIST], look: [0, 0, 0], fov: FOV },
        { t: DUR, pos: [0, 0, DIST], look: [0, 0, 0], fov: FOV },
      ],
      update(t, api) {
        const [a, b, c] = visibility(t);
        absolute.setOpacity(b); share.setOpacity(b); diversity.setOpacity(c);
        lines.clear();
        const line = (x0: number, y0: number, x1: number, y1: number, color: RGB, alpha: number) => lines.push(sp(x0, y0), sp(x1, y1), color, alpha);
        const box = (x0: number, y0: number, x1: number, y1: number, color: RGB) => lines.polyline([sp(x0, y0), sp(x1, y0), sp(x1, y1), sp(x0, y1)], color, 0.7 * a, true);
        const arrow = (x0: number, y0: number, x1: number, y1: number, color: RGB) => {
          line(x0, y0, x1, y1, color, a * 0.7);
          const dx = x1 - x0, dy = y1 - y0, length = Math.hypot(dx, dy), ux = dx / length, uy = dy / length;
          line(x1, y1, x1 - ux * 11 + uy * 6, y1 - uy * 11 - ux * 6, color, a * 0.7);
          line(x1, y1, x1 - ux * 11 - uy * 6, y1 - uy * 11 + ux * 6, color, a * 0.7);
        };
        if (a > 0.004) {
          label(api, 'allocation-title', 800, 180, 'Capacity allocation', a, { size: 'title', help: '<p>Box sizes carry no quantity. Each task has its own specialized allocation. The shared allocation adds the same expected $n_G$ to every task.</p>' });
          label(api, 'allocation-evidence', 800, 210, 'Illustration', a, { size: 'xs', color: CSS.slate });
          box(602, 252, 998, 345, C.violet);
          label(api, 'shared-title', 800, 280, 'General circuits', a, { size: 'title', color: CSS.violet });
          label(api, 'shared-cost', 800, 315, 'Capacity cost $N_G=m n_G^{1+\\gamma}$', a);
          const xs = [420, 800, 1180], ks = ['1', '2', 'k'];
          for (let j = 0; j < xs.length; j++) {
            const x = xs[j], k = ks[j];
            arrow(800, 345, x - 70, 450, C.violet);
            box(x - 140, 450, x + 140, 535, C.slate);
            label(api, `effective-${j}`, x, 479, `$n_G+n_${k}$`, a, { size: 'lg' });
            label(api, `task-dof-${j}`, x, 515, `Task ${k} · effective DOF`, a, { size: 'sm' });
            box(x - 140, 585, x + 140, 655, C.white);
            label(api, `task-${j}`, x, 609, `Task ${k} · specialized circuits`, a, { size: 'sm' });
            label(api, `cost-${j}`, x, 637, `Capacity cost $n_${k}$`, a);
            arrow(x + 70, 585, x + 70, 535, C.white);
          }
          label(api, 'shared-benefit', 800, 393, '$n_G$ expected DOF for every task', a, { color: CSS.violet, plate: true });
          label(api, 'budget', 800, 700, '$N=\\sum_k n_k+N_G$', a, { size: 'lg', help: '<p>Brill chooses units so one task-specific DOF costs one unit of capacity. General DOF and general capacity are different quantities, related by the cost function above.</p>' });
        }
        label(api, 'absolute-title', 520, 178, 'General DOF', b, { size: 'title' });
        label(api, 'share-title', 1080, 178, 'Task-weighted general DOF share', b, { size: 'title', help: '<p>The share averages $n_G/(n_k+n_G)$ across tasks, weighting each task by its frequency. The published axis shows the full range up to a share of 1.</p>' });
        label(api, 'fig2-key', 800, 661, 'Dots · optimized allocations&emsp;Solid lines · analytic approximations', b, { size: 'sm', help: '<p>$N$ is model capacity. Dotted lines connect the numerically optimized allocations. The left panel uses a logarithmic scale, so zero is visible only on the right. Color marks $c/D$, which controls how quickly task loss falls as effective DOF grow.</p>' });
        label(api, 'fig2-source', 800, 696, 'Reported by Ari Brill · Figure 2', b, { size: 'xs', color: CSS.slate, figure: FIG2 });
        label(api, 'diversity-title', 510, 176, 'Task frequencies', c, { size: 'title', help: '<p>Task frequencies become more even as $\\alpha$ decreases from left to right. This gives rare tasks more weight.</p>' });
        label(api, 'expressivity-title', 1090, 176, 'Approximation per DOF', c, { size: 'title', help: '<p>Larger $c/D$ lets the same number of degrees of freedom approximate a task more accurately.</p>' });
        label(api, 'fig3-source', 800, 674, 'Reported by Ari Brill · Figure 3 · vertical scales differ', c, { size: 'xs', color: CSS.slate, figure: FIG3, help: '<p>Both panels plot the task-weighted general DOF share. Color marks the efficiency-decay exponent $\\gamma$. The panels use different axes, so the visible slopes cannot be compared directly.</p>' });
        lines.commit();
      },
      details(t) { return details[t < T2 ? 0 : t < T3 ? 1 : 2]; },
      inspect(t): Hotspot[] {
        if ((t >= T2 && t < T2 + 0.48) || (t >= T3 && t < T3 + 0.48)) return [];
        if (t < T2) return [{ id: 'generality-cost', pos: sp(998, 255), dx: 15, title: 'Declining efficiency is an assumption', body: '<p>Each additional unit of general DOF costs more capacity, with $\\gamma>0$. A general circuit contributes equally to all tasks in expectation. Both properties are assumptions about the population of available circuits.</p>' }];
        if (t < T3) return [{ id: 'generality-share', pos: sp(1325, 185), title: 'Growing amount, falling share', body: '<p>The amount $n_G$ and the frequency-weighted share answer different questions. Specialized DOF can grow faster, lowering the share while the absolute general allocation grows. The curves report optimal allocation variables; neither axis is a benchmark score.</p>' }];
        return [{ id: 'generality-empirical', pos: sp(1340, 180), title: 'An optimum and a learning process', body: '<p>The model describes which allocation would minimize loss if it could be reached. It leaves practical circuit learnability, training history, finite data and computation, and empirical identification of its circuit classes unresolved.</p>' }];
      },
    } satisfies Chapter;
  },
};

export default generality;
