# Learning-Augmented Densest At-Most-k Subgraph

Experiments on learning-augmented algorithms for the **Densest At-Most-k Subgraph (DamkS)** problem: given a graph G and integer k, find a subset S of vertices with |S| ≤ k that maximizes the density |E(S)| / |S|.

The key idea is to use a machine-learning predictor to seed the search, then use a fast augmentation procedure to correct the prediction, trading the slow exact solver for a near-optimal result in milliseconds.

## Problem Setup

- **Densest Subgraph (DS)**: find S ⊆ V maximizing |E(S)| / |S| (solved exactly by Goldberg's max-flow algorithm).
- **Densest At-Most-k Subgraph (DamkS)**: same with the additional constraint |S| ≤ k. Solved exactly via iterative pruning + brute-force enumeration with clique upper bounds.

## Approach

1. **Train a random forest** on node features (degree, average neighbor degree, graph size) to predict which nodes belong to the DamkS solution.
2. **Augmented algorithm** at inference time:
   - Let S = predicted DamkS set from the classifier.
   - Add ε·|S|/(1−ε) external nodes that are most connected to S.
   - Trim by repeatedly removing the lowest-degree node within S until |S| ≤ k.
3. **Compare** approximation ratio and runtime against the exact DamkS solver.

## Repository Structure

```
densest_subgraph.jl            # All graph algorithms (Goldberg DS, peeling, DamkS exact)
create_dataset.jl              # Step 1 (SNAP datasets only): sample connected subgraphs from a raw edge-list
create_labeled_dataset_damks.jl  # Step 2: compute exact DamkS labels for each subgraph
damks_experiment.jl            # Step 3: train classifier, run augmented vs exact comparison
run_pipeline.sh                # End-to-end runner ensuring consistent K across all steps
data/                          # JSON datasets (raw and labeled)
outputs/                       # Saved plots
datasets/damks/                # Intermediate CSVs for classifier training
```

## Datasets

### Twitch Ego Networks (recommended — no Step 1 needed)

The [Twitch Ego Networks](https://snap.stanford.edu/data/twitch_ego_nets.html) dataset contains ~127k small ego-network graphs directly, so they can be fed straight into Step 2 without sampling. Put `twitch_edges.json` in the `data/` directory.


### SNAP edge-list datasets (Cit-HepPh, Amazon)

These are single large graphs, so Step 1 (`create_dataset.jl`) is needed first to sample connected subgraphs via BFS.
Files use the SNAP format (lines of `u v`, `#`-prefixed comment lines ignored). Tested on:

- [Cit-HepPh](https://snap.stanford.edu/data/cit-HepPh.html) — citation network
- [Amazon0302](https://snap.stanford.edu/data/amazon0302.html) — co-purchase network

Place the `.txt` file in the repository root, edit `create_dataset.jl` to point at it and set `num_graphs` / `sample_size`, then run all three steps. `num_graphs` controls how many subgraphs are sampled, and `sample_size` controls the number of nodes in each subgraph (50 or 100 recommended for DamkS experiments to keep exact solver runtime reasonable).

## Running the Pipeline

### Full pipeline (recommended)

```bash
./run_pipeline.sh data/Cit-HepPh.json 15
```

This runs Steps 2 and 3 end-to-end with K=15, deriving train/val filenames automatically.

### Step by step

**Step 1 — sample subgraphs** (edit `create_dataset.jl` to point at your `.txt` file and set `num_graphs` / `sample_size`):
```bash
julia create_dataset.jl
# outputs: data/<name>.json
```

**Step 2 — compute DamkS labels** (multi-threaded, ~2 min timeout per graph):
```bash
julia --threads auto create_labeled_dataset_damks.jl data/<name>.json 15
# outputs: data/<name>_train.json, data/<name>_val.json
```

**Step 3 — run experiment**:
```bash
julia --threads auto damks_experiment.jl data/<name>_train.json data/<name>_val.json 15
# outputs: outputs/damks_*.png
```

### Dependencies

Install Julia dependencies once from the project root:
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Output

The experiment reports:

- **Classifier accuracy** on a held-out split of the training data.
- **Runtime comparison**: mean/median/total time for the augmented algorithm vs the exact solver.
- **Approximation ratio** (augmented density / optimal density): mean, median, min, max, and fraction of graphs achieving ≥ 0.95.
- **Plots** saved to `outputs/`:
  - `damks_augmented_vs_optimal.png` — scatter: augmented density vs optimal density
  - `damks_predictor_vs_optimal.png` — scatter: raw predictor density vs optimal density
  - `damks_predictor_vs_augmented.png` — scatter: raw predictor density vs augmented density
  - `damks_approx_ratio_hist.png` — histogram of approximation ratios
  - `damks_example.png` — visualization of one training graph with DamkS nodes highlighted
