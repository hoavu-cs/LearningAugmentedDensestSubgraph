# ---------------------------------------------------------------------------
# Step 3 of the data pipeline: run the learning-augmented DamkS experiment.
#
# Trains a random forest classifier on node features (degree, average neighbor
# degree, graph size) to predict DamkS membership, then evaluates an augmented
# algorithm on validation graphs and compares it against the exact solver.
#
# Augmented algorithm (at validation time):
#   1. Predict DamkS membership for each node using the trained classifier.
#   2. Expand the predicted set S by adding epsilon*|S|/(1-epsilon) external
#      neighbors most connected to S (to hedge against false negatives).
#   3. Trim by removing the lowest-degree node until |S| <= K.
# epsilon is derived from the classifier's test FPR/FNR.
#
# Usage:
#   julia --threads auto damks_experiment.jl <train.json> <val.json> [K]
#   julia --threads auto damks_experiment.jl <train.json> <val.json> --k K
#
# Arguments:
#   train.json   labeled training set from create_labeled_dataset_damks.jl
#   val.json     labeled validation set from create_labeled_dataset_damks.jl
#   K            size constraint for DamkS (default: 15); must match the value
#                used in create_labeled_dataset_damks.jl
#
# Outputs:
#   datasets/damks/node_features_with_labels_damks.csv   training feature matrix
#   outputs/damks_example.png                            example graph with DamkS highlighted
#   outputs/damks_predictor_vs_augmented.png             predictor vs augmented density scatter
#   outputs/damks_augmented_vs_optimal.png               augmented vs optimal density scatter
#   outputs/damks_predictor_vs_optimal.png               predictor vs optimal density scatter
#   outputs/damks_approx_ratio_hist.png                  approximation ratio distribution
# ---------------------------------------------------------------------------

using Graphs, GraphPlot, Compose, Colors
using JSON, Statistics, StatsBase, Plots, Random
using DecisionTree, DataFrames, CSV
using ProgressMeter
import Cairo, Fontconfig

include("densest_subgraph.jl")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

train_file = length(ARGS) >= 1 ? ARGS[1] : "data/twitch_edges_train.json"
val_file   = length(ARGS) >= 2 ? ARGS[2] : "data/twitch_edges_val.json"

let k_idx = findfirst(==("--k"), ARGS)
    global K = k_idx !== nothing ? parse(Int, ARGS[k_idx + 1]) :
               length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 15
end

mkpath("outputs")
mkpath("datasets/damks")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""Build a SimpleGraph from [[u,v],...] edge list (1-indexed vertices)."""
function build_julia_graph(size::Int, edges)
    G = SimpleGraph(size)
    for e in edges
        add_edge!(G, Int(e[1]), Int(e[2]))
    end
    return G
end

"""Compute degree and average neighbor degree for each vertex in G."""
function compute_features(G)
    degs = degree(G)
    avg_neigh_deg = zeros(Float64, nv(G))
    for v in 1:nv(G)
        nbrs = neighbors(G, v)
        if !isempty(nbrs)
            avg_neigh_deg[v] = mean(degs[nbrs])
        end
    end
    return degs, avg_neigh_deg
end

"""Compute density of a set S in graph G."""
function compute_density_set(G, S::Set{Int})
    isempty(S) && return 0.0
    e_s = 0
    for v in S
        for u in neighbors(G, v)
            u in S && (e_s += 1)
        end
    end
    return (e_s / 2) / length(S)
end

"""
Augment predicted DamkS set S:
1. Add epsilon*|S|/(1-epsilon) external nodes with most connections to S.
2. Trim by removing the minimum-degree node until |S| <= k.
"""
function augment_damks(G, S::Set{Int}, k::Int, epsilon=0.1)
    t = Dict{Int,Int}()
    for v in S
        for u in neighbors(G, v)
            if u ∉ S
                t[u] = get(t, u, 0) + 1
            end
        end
    end

    sorted_nodes = sort(collect(t), by=x -> x[2], rev=true)
    num_to_add   = floor(Int, length(S) * epsilon / (1 - epsilon))
    new_S        = copy(S)
    for (node, _) in sorted_nodes[1:min(num_to_add, length(sorted_nodes))]
        push!(new_S, node)
    end

    while length(new_S) > k
        min_v = argmin(v -> count(u -> u in new_S, neighbors(G, v)), collect(new_S))
        delete!(new_S, min_v)
    end

    return new_S, compute_density_set(G, new_S)
end

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

println("Loading training data from $train_file ...")
train_data = JSON.parsefile(train_file)
train_keys = sort(collect(keys(train_data)), by=k -> parse(Int, k))
println("Training graphs: $(length(train_keys))")

println("Loading validation data from $val_file ...")
val_data = JSON.parsefile(val_file)
val_keys = sort(collect(keys(val_data)), by=k -> parse(Int, k))
println("Validation graphs: $(length(val_keys))")

# ---------------------------------------------------------------------------
# Visualize one example from training data
# ---------------------------------------------------------------------------

example_gid = first(gid for gid in train_keys if length(train_data[gid]["edges"]) >= 100)
example_data = train_data[example_gid]
G_ex = build_julia_graph(Int(example_data["size"]), example_data["edges"])
damks_set_ex = Set{Int}(Int(v) for v in example_data["damks_vlist"])

node_colors       = [v in damks_set_ex ? colorant"orange" : colorant"lightgray" for v in 1:nv(G_ex)]
node_stroke_colors = [v in damks_set_ex ? colorant"black" : colorant"gray"      for v in 1:nv(G_ex)]
node_stroke_widths = [v in damks_set_ex ? 2.5 : 0.5                             for v in 1:nv(G_ex)]

draw(PNG("outputs/damks_example.png", 600, 600),
    gplot(G_ex,
        nodefillc=node_colors,
        nodestrokec=node_stroke_colors,
        nodestrokelw=node_stroke_widths,
        edgestrokec=colorant"black",
        edgelinewidth=1.0,
        NODESIZE=0.04))

println("Saved example graph ($(nv(G_ex)) nodes, DamkS size=$(length(damks_set_ex)), density=$(round(example_data["damks_density"], digits=3))) → outputs/damks_example.png")

# ---------------------------------------------------------------------------
# Write training features to CSV
# ---------------------------------------------------------------------------

csv_path = "datasets/damks/node_features_with_labels_damks.csv"

open(csv_path, "w") do io
    println(io, "degree,avg_neighbor_degree,number_of_nodes,is_in_damks")
    for gid in train_keys
        data  = train_data[gid]
        G     = build_julia_graph(Int(data["size"]), data["edges"])
        degs, avg_neigh_deg = compute_features(G)
        damks_set = Set{Int}(Int(v) for v in data["damks_vlist"])
        n = nv(G)

        for v in 1:n
            println(io, "$(degs[v]),$(avg_neigh_deg[v]),$n,$(v in damks_set ? 1 : 0)")
        end
    end
end

# ---------------------------------------------------------------------------
# Train random forest
# ---------------------------------------------------------------------------

df    = CSV.read(csv_path, DataFrame)
X_all = Matrix{Float64}(df[:, [:degree, :avg_neighbor_degree, :number_of_nodes]])
y_all = Int.(df[:, :is_in_damks])

println("Features shape: ", size(X_all), ", Labels shape: ", size(y_all))

Random.seed!(42)
n_samples = size(X_all, 1)
n_train   = floor(Int, 0.8 * n_samples)
idx       = shuffle(1:n_samples)

X_train, y_train = X_all[idx[1:n_train], :],       y_all[idx[1:n_train]]
X_test,  y_test  = X_all[idx[n_train+1:end], :],   y_all[idx[n_train+1:end]]

forest = build_forest(y_train, X_train, 2, 10, 0.7, -1; rng=42)

y_pred_test = apply_forest(forest, X_test)
acc = mean(y_pred_test .== y_test)
println("Classifier accuracy: ", round(acc, digits=4))
println(DecisionTree.confusion_matrix(y_test, y_pred_test))

tp_test = sum((y_pred_test .== 1) .& (y_test .== 1))
fp_test = sum((y_pred_test .== 1) .& (y_test .== 0))
fn_test = sum((y_pred_test .== 0) .& (y_test .== 1))
tn_test = sum((y_pred_test .== 0) .& (y_test .== 0))
fpr_test = (fp_test + tn_test) > 0 ? fp_test / (fp_test + tn_test) : 0.0
fnr_test = (tp_test + fn_test) > 0 ? fn_test / (tp_test + fn_test) : 0.0
epsilon  = max(fpr_test, fnr_test)
println("Test FPR: $(round(fpr_test, digits=4)), FNR: $(round(fnr_test, digits=4)), epsilon: $(round(epsilon, digits=4))")

# ---------------------------------------------------------------------------
# Evaluate on validation graphs
# ---------------------------------------------------------------------------

n_val = length(val_keys)
predict_density_vals   = Vector{Float64}(undef, n_val)
augmented_density_vals = Vector{Float64}(undef, n_val)
optimal_density_vals   = Vector{Float64}(undef, n_val)
exact_density_vals     = Vector{Float64}(undef, n_val)
augmented_times        = Vector{Float64}(undef, n_val)
exact_times            = Vector{Float64}(undef, n_val)

# --- Augmented algorithm ---
progress = Progress(n_val, desc="Running augmented algorithm: ")
Threads.@threads for i in eachindex(val_keys)
    gid  = val_keys[i]
    data = val_data[gid]
    G    = build_julia_graph(Int(data["size"]), data["edges"])
    degs, avg_neigh_deg = compute_features(G)
    n = nv(G)

    X_pred = hcat(Float64.(degs), avg_neigh_deg, fill(Float64(n), n))
    y_pred = apply_forest(forest, X_pred)
    S      = Set(v for (v, p) in enumerate(y_pred) if p == 1)

    predict_density_vals[i] = compute_density_set(G, S)

    t0 = time()
    _, aug_dens            = augment_damks(G, S, K, epsilon)
    augmented_times[i]     = time() - t0
    augmented_density_vals[i] = aug_dens

    optimal_density_vals[i] = Float64(data["damks_density"])
    next!(progress)
end

# --- Exact algorithm ---
progress = Progress(n_val, desc="Running exact algorithm:      ")
Threads.@threads for i in eachindex(val_keys)
    gid  = val_keys[i]
    data = val_data[gid]
    G    = build_julia_graph(Int(data["size"]), data["edges"])

    t0 = time()
    _, exact_dens      = densest_at_most_k_subgraph(G, K)
    exact_times[i]     = time() - t0
    exact_density_vals[i] = exact_dens
    next!(progress)
end

println("Validation graphs evaluated: $n_val")
println("\nRuntime Comparison:")
println("-----------------------------------------------------")
println("Augmented algorithm — mean: $(round(mean(augmented_times)*1000, digits=2))ms, median: $(round(median(augmented_times)*1000, digits=2))ms, total: $(round(sum(augmented_times), digits=2))s")
println("Exact algorithm     — mean: $(round(mean(exact_times)*1000, digits=2))ms, median: $(round(median(exact_times)*1000, digits=2))ms, total: $(round(sum(exact_times), digits=2))s")
println("Speedup (exact/aug) — mean: $(round(mean(exact_times) / mean(augmented_times), digits=1))x")
println("-----------------------------------------------------")

# ---------------------------------------------------------------------------
# Approximation ratio
# ---------------------------------------------------------------------------

approx_ratios = [o > 0 ? a / o : 0.0 for (a, o) in zip(augmented_density_vals, optimal_density_vals)]

avg_ratio    = mean(approx_ratios)
median_ratio = median(approx_ratios)
min_ratio    = minimum(approx_ratios)
max_ratio    = maximum(approx_ratios)
high_quality = sum(r >= 0.95 for r in approx_ratios)
high_pct     = 100.0 * high_quality / length(approx_ratios)

println("\nApproximation Ratio Report (Augmented / Optimal):")
println("-----------------------------------------------------")
println("Average Approximation Ratio:  $(round(avg_ratio, digits=4))")
println("Median Approximation Ratio:   $(round(median_ratio, digits=4))")
println("Worst Case (Min Ratio):       $(round(min_ratio, digits=4))")
println("Best Case (Max Ratio):        $(round(max_ratio, digits=4))")
println("-----------------------------------------------------")
println("High Quality Solutions (>= 0.95): $high_quality/$(length(approx_ratios)) ($(round(high_pct, digits=2))%)")
println("-----------------------------------------------------")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

function plot_comparison(x_vals, y_vals, x_lbl, y_lbl, fname)
    lim = max(maximum(x_vals), maximum(y_vals)) * 1.05
    p = scatter(x_vals, y_vals,
        xlabel=x_lbl, ylabel=y_lbl,
        title="$x_lbl vs $y_lbl",
        alpha=0.6, markerstrokecolor=:black, label="Graphs", size=(500, 500))
    plot!(p, [0, lim], [0, lim], color=:red, linestyle=:dash, label="y = x (Equality)")
    savefig(p, fname)
end

plot_comparison(predict_density_vals, augmented_density_vals,
    "Predictor's output density", "Algorithm's output density",
    "outputs/damks_predictor_vs_augmented.png")

plot_comparison(optimal_density_vals, augmented_density_vals,
    "Optimal density (Ground Truth)", "Algorithm's output density",
    "outputs/damks_augmented_vs_optimal.png")

plot_comparison(optimal_density_vals, predict_density_vals,
    "Optimal density (Ground Truth)", "Predictor's output density",
    "outputs/damks_predictor_vs_optimal.png")

p_hist = histogram(approx_ratios,
    xlabel="Approximation Ratio (Augmented / Optimal)",
    ylabel="Number of Graphs",
    title="Approximation Ratio Distribution",
    bins=20, legend=false, color=:teal, alpha=0.7)
vline!(p_hist, [avg_ratio], color=:red, linestyle=:dash, linewidth=2,
    label="Avg: $(round(avg_ratio, digits=3))")
savefig(p_hist, "outputs/damks_approx_ratio_hist.png")

println("\nPlots saved to outputs/")
