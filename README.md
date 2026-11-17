# PIRAMID Observatory

[Open the interactive presentation](https://sergeisoly.github.io/piramid-observatory/)

An independent, interactive presentation of [PIRAMID](https://princint.ai/piramid/), a research division of Principles of Intelligence. How can we tell what a neural network has learned, and which features it uses to produce an answer?

The eight research chapters combine the authors’ published results with small experiments you can run in the browser. The [percolation study](offline/percolation/README.md) was run separately at the paper’s scale. The results ship with the presentation. You only need Python to rerun the offline study. Sergei Solovev created the presentation independently. The cited researchers have not reviewed it.

## Run locally

With Node.js 22.12 or newer and npm installed, start the dev server from the repository root.

```sh
npm ci
npm run dev
```

Open [the local presentation](http://127.0.0.1:5199/) in a browser with WebGPU enabled. The renderer can fall back to WebGL2. Browser checks use Chrome with WebGPU.

Choose **Read as slides** to step through the figures, or **Play as a film** for automatic playback. In Slides, → or Space advances to the next slide. During an animation, the first press skips to its end. **Notes** opens the explanation for the current slide. Small **?** buttons explain graphical marks. Use the **?** in the playback bar for keyboard shortcuts, or **Home** to return to the overview.

## Chapters and sources

| Chapter | Primary source |
| --- | --- |
| Critical percolation | [Brill & Carlson, *Critical Percolation as a Synthetic Data Model for Interpretability*](https://arxiv.org/abs/2606.20347) |
| General capabilities | [Brill, *A Model for Scaling Laws of General Intelligence*](https://openreview.net/notes/edits/attachment?id=kaw40vezr3&name=pdf) |
| Mean field | [Vaintrob & Greenspan, *Mean field sequence*](https://www.lesswrong.com/posts/rduzFkTKx5pGKWKcL/mean-field-sequence-an-introduction) and [Vaintrob & Hänni](https://arxiv.org/abs/2607.05735) |
| Frustration | [Vaintrob, *A tale of three theories*](https://www.lesswrong.com/posts/siu22scEfuKxpSgfK/a-tale-of-three-theories-sparsity-frustration-and) |
| Empirical NTK | [Lin, *Feature Identification via the Empirical NTK*](https://arxiv.org/abs/2510.00468) |
| Causal elicitation | [Mack, Panickssery & Turner, *Mechanistically Eliciting Latent Behaviors in Language Models*](https://arxiv.org/abs/2606.29604) |
| Parity bottleneck | [Mack et al., *Scaling Interpretable Transformers with Parity Bottleneck Layers*](https://arxiv.org/abs/2607.20652) |
| Coarse-graining | [Greenspan et al., *Towards Worst-Case Guarantees with Scale-Aware Interpretability*](https://arxiv.org/abs/2602.05184) |

The opening and closing chapters draw on the team’s [introduction](https://www.lesswrong.com/posts/nbSJhbLERTZFeNxY7/introducing-piramid-physics-informed-research-for-ambitious) and [progress report](https://www.lesswrong.com/posts/T2REsZneix3bmAKtL/piramid-progress-and-plans).

## Working on the presentation

- [src/scenes/](src/scenes/) contains the chapters, captions, Notes and experiments. [The chapter manifest](src/scenes/index.ts) defines their order and URL IDs.
- [src/engine/](src/engine/) handles rendering, playback, framing, figure inspection and Notes. [src/kit/](src/kit/) provides drawing primitives.
- [public/figures/](public/figures/) holds the source figures and their attribution. [public/data/](public/data/) holds the precomputed probe results.

To work on one chapter, open [Mean field on its own](http://127.0.0.1:5199/?only=meanfield&ch=meanfield&t=12.5&slides=1). `only` selects the chapters to load, `ch` and `t` set the starting chapter and local time in seconds. Use `explore=1` to open Notes or `p.key=value` to override a chapter parameter. See [the entry point](src/main.ts) for the other switches.

The dev server preserves your reading position across edits. Use `npm run dev:hold` to batch several edits and `npm run dev:release` to refresh once. `npm run dev:updates` shows the current state. All three accept `-- --port=PORT` for a nondefault server.

### Checks

```sh
npm run typecheck
npm run build
```

With the dev server running in another terminal, run `npm run check` for playback, Notes, URL parameters and renderer recovery. Run `node tools/smoke.mjs` for a walkthrough of the full presentation. The checks default to the macOS Chrome executable. Pass `--chrome=/path/to/chrome` and `--url=http://127.0.0.1:PORT/` as needed (`npm run check -- --chrome=…` through npm).

The [browser tools](tools/) also capture frames (`shot.mjs`), measure figure bounds (`boxes.mjs`), export captions and Notes (`script.mjs`, `dump-notes.mjs`) and check Notes navigation and layout. Each script documents its arguments. Recompute the affected chapter’s framing after changing its figure geometry.

### Blog build

`npm run build` writes a static site with relative asset paths to `dist/`. To check it locally, stop the dev server and run `npm run preview`, which also uses port 5199.

The personal blog’s **Projects** page links directly to `/piramid-observatory/`. To refresh that page, replace the contents of the blog repository’s `piramid-observatory/` directory with this build and run `quarto render` in the blog repository. Its README includes the refresh command.

The link preview uses [public/social-card.jpg](public/social-card.jpg). To update it, run `npm run shot -- --poster=1 --card=1 --w=1200 --h=630 --tag=card` against the local dev server and use the resulting poster image.
