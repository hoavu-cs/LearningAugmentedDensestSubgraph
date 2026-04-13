## Learning-Augmented Densest Subgraph

Implements learning-augmented algorithms for **Densest Subgraph (DS)** and **Densest At-Most-k Subgraph (DamkS)** on SNAP-format graphs.

## Pipeline

### Step 1 — Sample subgraphs
```bash
julia create_dataset.jl [--input FILE] [--num_graphs N] [--max_n N]
```
Reads a SNAP edge-list, samples `num_graphs` BFS-induced subgraphs of up to `max_n` nodes, writes `data/<name>.json`.

### Step 2 — Label with exact DamkS
```bash
julia --threads auto create_labeled_dataset_damks.jl <input.json> [K]
```
Computes the exact DamkS solution (size constraint `K`) for each graph in parallel, writes `data/<name>_train.json` and `data/<name>_val.json`.

### Step 3 — Run experiment
```bash
julia --threads auto damks_experiment.jl <train.json> <val.json> [K]
```
Trains a random forest on node features (degree, avg neighbor degree, graph size) to predict DamkS membership. At validation time runs the **augmented algorithm**: expand predicted set with ε·|S|/(1−ε) external neighbors, trim to `|S| ≤ K`. Reports approximation ratio and speedup vs. exact solver.

### Run full pipeline
```bash
./run_pipeline.sh data/Cit-HepPh.json 15
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--input` | `Cit-HepPh.txt` | Source edge-list (SNAP format) |
| `--num_graphs` | `5000` | Number of subgraphs to sample |
| `--max_n` | `100` | Max nodes per subgraph |
| `K` | `15` | DamkS size constraint — must be identical across steps 2 and 3 |

## Outputs

| Path | Description |
|---|---|
| `data/<name>.json` | Sampled subgraph dataset |
| `data/<name>_train.json` | Labeled training set |
| `data/<name>_val.json` | Labeled validation set |
| `datasets/damks/*.csv` | Node feature matrix |
| `outputs/*.png` | Plots (graph visualizations, density scatter, approx ratio histogram) |
