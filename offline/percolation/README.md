# Offline percolation study

This study reruns the one-cluster experiment at the paper’s scale and adds a paired control. Can a linear
probe recover the same latent values from the network before training? Chapter 1 reads the resulting
[phase 2 export](../../public/data/percolation-offline.json), schema `percolation-offline/1`.
[Brill and Carlson, *Critical Percolation as a Synthetic Data Model for Interpretability*, §5 and Appendix B](https://arxiv.org/abs/2606.20347).

## Run the study

Start at the repository root with Git and Python 3.13. The example keeps the environment, generator and
runs outside the repository. Set `WORK` to an absolute path you can reuse when resuming a run.

```sh
WORK="$HOME/piramid-percolation"
GEN="$WORK/generator"
RUNS="$WORK/runs"
mkdir -p "$WORK"
python3.13 -m venv "$WORK/.venv"
PYTHON="$WORK/.venv/bin/python"
"$PYTHON" -m pip install -r offline/percolation/requirements.txt
git clone https://github.com/aribrill/percolation-synthetic-data.git "$GEN"
git -C "$GEN" checkout --detach 3dd47a808f02d161a34615570814f107ab1c1627
```

The pinned requirements describe the macOS environment. The original GPU run used Python 3.12,
NumPy 2.1.2 and PyTorch 2.8.0+cu128. Use a CUDA-enabled PyTorch build for a CUDA run. The scripts select
CUDA, then MPS, then CPU. On MPS, float64 probe arithmetic runs on the CPU.

```sh
"$PYTHON" offline/percolation/run_phase1.py --generator "$GEN" --out "$RUNS" && \
  "$PYTHON" offline/percolation/run_phase2.py --generator "$GEN" --out "$RUNS" --jobs 4
```

- **Phase 1** generates `tree0`, scores the baselines, trains `tree0-init0`, fits probes and exports the
  declared run. The driver stops after a failed data integrity check, baseline check or network check
  (A, B or C). It records the probe check (D) in the analysis. Reruns skip stages with existing completion
  files and resume interrupted training from a saved checkpoint.
- **Phase 2** uses the same `RUNS` folder. It backs up the original probes before refitting them, adds
  `tree0-init1`, `tree0-init2`, `tree1-init0` and `tree2-init0`, and runs held-out, ridge and value-swap
  diagnostics. It writes `phase2_summary.json` and merges it into the export. Completed jobs with output
  markers are skipped on reruns. Either phase writes logs to `$RUNS/logs/` and timings to
  `$RUNS/phase1_steps.json` or `$RUNS/phase2_steps.json`.
- **Resource use.** Phase 1 took about 6 minutes on an RTX 4080 SUPER with 16 GB VRAM, with peaks of 3.6 GB
  RAM during probes and 4.8 GB VRAM during nearest-neighbor search. Phase 2 took about 24 minutes on the
  same GPU, including restarts. Four concurrent jobs fit that run's memory budget. Eight ran out of VRAM.

Both phases write to `$RUNS/export/percolation-offline.json`. To use a fresh phase 2 export in a local
build, copy it to `public/data/percolation-offline.json`. For a separate re-export from finished phase 2
outputs, pass both the summary and output path.

```sh
"$PYTHON" offline/percolation/export.py --tree "$RUNS/tree0" --run "$RUNS/tree0-init0" \
  --summary "$RUNS/phase2_summary.json" --out "$RUNS/percolation-offline.reexport.json"
```

To run the phase 1 stages individually, use these commands from the repository root. Add `--resume` to
`train.py` to continue from an existing checkpoint.

```sh
"$PYTHON" offline/percolation/generate.py --generator "$GEN" --out "$RUNS/tree0"
"$PYTHON" offline/percolation/baselines.py --tree "$RUNS/tree0"
"$PYTHON" offline/percolation/train.py --tree "$RUNS/tree0" --out "$RUNS/tree0-init0"
"$PYTHON" offline/percolation/probe.py --tree "$RUNS/tree0" --state raw --sklearn-check raw:50
"$PYTHON" offline/percolation/probe.py --tree "$RUNS/tree0" --run "$RUNS/tree0-init0" --state init --sklearn-check hidden0:50
"$PYTHON" offline/percolation/probe.py --tree "$RUNS/tree0" --run "$RUNS/tree0-init0" --state final --sklearn-check hidden0:50
"$PYTHON" offline/percolation/analyze.py --tree "$RUNS/tree0" --run "$RUNS/tree0-init0"
"$PYTHON" offline/percolation/export.py --tree "$RUNS/tree0" --run "$RUNS/tree0-init0" \
  --out "$RUNS/export/percolation-offline.json"
```

Tree folders contain `dataset.npz`, `data_summary.json`, `baselines.json`, raw probe files and, on
`tree0` after phase 2, `values_v20001.{npz,json}`. Run folders contain `init.pt`, checkpoints every
50 epochs, `final.pt`, `train.log`, `metrics.json`, network probe files and `analysis*.json`. Probe
files come in NPZ, CSV and JSON.

## Protocol

- **Data.** The generator makes one cluster of 200,000 points with 100 input coordinates each, following
  Table 4. It assigns a value to every latent and derives point targets from the scaled path sum
  (Eq. 4). The first tree uses graph, embedding and value seeds 0, 10,000 and 20,000. The other two
  trees add 1 or 2 to each seed. The compact tree stores parents and ancestors at depths 0, 5, …, 200.
- **Baselines and network.** Ridge on the raw inputs and 1-nearest-neighbor are scored on the network test
  split. The neighbor search uses the full dataset with the query itself excluded, proposes 16 candidates
  in float32 and reranks them in float64. The result includes a tie-averaged score. The Table 5 residual MLP
  has width 256, three blocks with fourfold hidden expansion and ReLU. It trains on the point target with
  AdamW for 500 epochs, with cosine decay, batches of 1,024 and an 80/10/10 split. It saves the initial
  weights before training. The first tree’s repeats keep the split fixed and change initialization and
  batch order. See [common.py](common.py) and [train.py](train.py) for constants and seed order.
- **Probes.** At each depth from 10 to 200 in steps of 5, a ridge probe predicts the value of each point’s
  ancestor at that depth. Eligible points lie below that depth and have an ancestor with at least 150
  points. The raw input, three hidden layers, and residual stream after the input layer and each block
  give eight sites. The paired network states use the same 70/30 probe split at each depth and the same
  rows, feature standardization on probe-training points and ridge strength of 1. Error is per-latent
  mean squared error (MSE) on probe-validation points. The strength comes from the
  [authors’ public probe configuration](https://github.com/tomingebretsencarlson/mechinterp-percolation/blob/a5fd5846350c68ef5528374cbcfce266ed8e622d/configs/interp/probes.yaml#L15).
- **Variants.** Use `probe.py --fit-rows heldout` to fit and score only on points held out from network
  training, or `analyze.py --metric ho` to restrict scoring only. Use `probe.py --alpha` to vary ridge
  strength. The value-swap diagnostic uses `generate.py --value-seed` on the existing tree and passes the
  resulting NPZ through `train.py --values` or `probe.py --values`. Network probes also record a probe
  from the scalar output. The export maps roughly 8,000 points at the probed depth with the most eligible
  latents onto the first two principal axes of the inputs. Sampling retains all of each family up to
  200 points and gives families above that floor a shared sampling rate where the floor permits.

### Decision rule

The rule was fixed before training. Pool latents across depths in five log-spaced bins of figure of merit,
the number of points under the latent divided by the square root of cluster size. Keep latents with
constant-reference MSE of at least 0.0001. In each bin with at least 10 latents, compute the median of paired
`MSE(trained) / MSE(untrained)` at each site.

- **Helps** requires a median below 0.85 in every eligible bin at every site in the group.
- **No gain** requires a median of at least 0.95 in a strict majority of eligible bins at every site in the group.
- **Mixed** covers the other outcomes.

Classify the hidden layers and residual stream separately. The across-run label requires agreement
across all five runs. See [analyze.py](analyze.py) and [diagnostics.py](diagnostics.py).

## Results and limitations

The declared run passed checks A–D. A verifies graph edge lengths and target path sums and compares the
ancestor table with direct walks on 1,000 sampled points. For B and C, ridge, nearest neighbor and MLP
test R² were 0.505, 0.940 and 0.935, within 0.03 of the paper’s 0.51, 0.94 and 0.94. For D, each trained
hidden layer and block output had a median error ratio to raw input below one in every eligible bin.
The pooled ratios were closer to one in the largest eligible bin than the smallest. An eight-bin
comparison with medians read from Fig. 4a gave similar ordering and scale. Exact figure arrays were
unavailable, and the depth curves in Fig. 5 differed in scale under the aggregations tried. Ridge test
R² on the first additional tree was 0.446, outside the Table 1 band for the declared tree.

All five runs met the no-gain rule at both site groups at ridge strength 1. The median hidden-layer
error ratios by bin were mostly between 1.0 and 1.4 across four runs and between 0.93 and 1.14 on the
fifth. In the declared run, trained probes had lower errors for 24–43% of latents at hidden layers and
37–45% at residual sites. In this setup, the untrained ReLU features already improved linear readability
relative to the raw input. Whether the network uses those values remains open.

- Lowering both thresholds by 0.05 preserves the calls. Raising them by 0.05 makes the hidden-layer call
  mixed on the two additional trees and the residual call mixed on all five runs. Several input-layer
  and block-output ratios fall just below the raised 1.00 cutoff. An injective affine input layer
  preserves linear expressivity, while ridge depends on feature coordinates.
- At the original thresholds, removing the input-layer site leaves the residual call unchanged. In the
  declared run, held-out scoring and held-out fitting both preserve the calls. Ridge strength 100 makes
  the residual call mixed. At 10,000 both calls become mixed, with lower trained errors in most site and
  bin pairs and higher absolute errors in most probes, especially for smaller latents.
- The network trained on one new draw of latent values over `tree0` reads the original values about as
  well as the one trained on those original values. The median ratio is 1.00, with a range of 0.90–1.13.
  The five main runs have three initializations on one tree and one each on two additional trees. Latents
  within a tree are dependent.
- Depths 0 and 5 yield constant targets under the 150-point filter and are omitted. The published
  generator recreates the dataset, whose identity with the paper’s files is unverified. Ridge probes
  use float64. The float32 scikit-learn refits at depth 50 differed by less than 1% in per-latent MSE.

## Provenance and reanalysis

The MIT-licensed data generator (© 2025 Ari Brill) is imported from the pinned checkout after
[common.py](common.py) verifies the SHA-256 of `percolation_dataset.py`.
[Generator at commit `3dd47a8`](https://github.com/aribrill/percolation-synthetic-data/tree/3dd47a808f02d161a34615570814f107ab1c1627).

The authors’ experiment repository supplied protocol details. The inspected revision has no license
file. The offline trainer and probes implement that protocol independently.
[Experiment repository at commit `a5fd584`](https://github.com/tomingebretsencarlson/mechinterp-percolation/tree/a5fd5846350c68ef5528374cbcfce266ed8e622d).

Training and probe fits ran on CUDA. In a macOS reanalysis, log-space rounding could move a bin endpoint
past the observation that defined it. The corrected analysis sets both endpoints to the observed values
and uses right-closed intervals that exclude the minimum and include the maximum. It applies this rule
to the five- and eight-bin analyses and the held-out, ridge and value-swap diagnostics. The five-bin
counts changed from `[136, 87, 10, 35, 41]` to `[135, 87, 10, 35, 42]` on the first tree and from
`[105, 56, 7, 0, 11]` to `[104, 56, 7, 0, 12]` on the third. The second tree’s counts stayed the same.
Across five runs and four variants, the largest change in a paired median ratio was 0.020. All
classifications at the original and shifted thresholds held.

The checked-in export combines archived errors and metrics with the corrected summaries. Its provenance
records the separate training, probe and export script hashes. The regenerated input matrix matches the
archived macOS hash and differs in its low-order bits from the CUDA matrix. The declared checkpoint gave
test R² 0.935261 on the regenerated inputs. The projection fixes the signs of the principal axes to the
original map orientation.
