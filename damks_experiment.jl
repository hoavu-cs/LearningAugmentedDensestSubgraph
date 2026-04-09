using Graphs
using JSON, Statistics, StatsBase, Plots, Random
using DecisionTree, DataFrames, CSV, ProgressMeter
import Cairo, Fontconfig

# ---------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------

file_path = "outputs/all.json"
raw_data = JSON.parsefile(file_path)

println("Loaded $(length(raw_data)) graphs.")

K = 15  # DamkS parameter k

mkpath("outputs")
mkpath("datasets/damks")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""Build a SimpleGraph from [[u,v],...] edge list (1-indexed vertices)."""
function build_julia_graph(size::Int, edges)
    G = SimpleGraph(size)
    for e in edges
        u, v = Int(e[1]), Int(e[2])
        add_edge!(G, u, v)
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
            if u in S
                e_s += 1
            end
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
    # Count connections from external nodes into S
    t = Dict{Int,Int}()
    for v in S
        for u in neighbors(G, v)
            if u ∉ S
                t[u] = get(t, u, 0) + 1
            end
        end
    end

    sorted_nodes = sort(collect(t), by=x -> x[2], rev=true)
    num_to_add = floor(Int, length(S) * epsilon / (1 - epsilon))
    new_S = copy(S)
    for (node, _) in sorted_nodes[1:min(num_to_add, length(sorted_nodes))]
        push!(new_S, node)
    end

    # Trimming: remove min-degree node in S until |S| <= k
    while length(new_S) > k
        min_v = argmin(v -> count(u -> u in new_S, neighbors(G, v)), collect(new_S))
        delete!(new_S, min_v)
    end

    d = compute_density_set(G, new_S)
    return new_S, d
end

# ---------------------------------------------------------------------------
# Build nx_graphs structure
# ---------------------------------------------------------------------------

all_keys = sort(collect(keys(raw_data)), by=k -> parse(Int, k))
n_graphs = length(all_keys)
training_size = floor(Int, 0.8 * n_graphs)

println("Training graphs: $training_size, Testing graphs: $(n_graphs - training_size)")

# ---------------------------------------------------------------------------
# Write training features to CSV
# ---------------------------------------------------------------------------

csv_path = "datasets/damks/node_features_with_labels_damks.csv"

open(csv_path, "w") do io
    println(io, "degree,avg_neighbor_degree,number_of_nodes,is_in_damks")

    @showprogress "Processing training graphs: " for gid in all_keys[1:training_size]
        data = raw_data[gid]
        G = build_julia_graph(Int(data["size"]), data["edges"])
        degs, avg_neigh_deg = compute_features(G)
        damks_set = Set{Int}(Int(v) for v in data["damks_vlist"])
        n = nv(G)

        for v in 1:n
            label = v in damks_set ? 1 : 0
            println(io, "$(degs[v]),$(avg_neigh_deg[v]),$n,$label")
        end
    end
end

# ---------------------------------------------------------------------------
# Train random forest
# ---------------------------------------------------------------------------

df = CSV.read(csv_path, DataFrame)
X_all = Matrix{Float64}(df[:, [:degree, :avg_neighbor_degree, :number_of_nodes]])
y_all = Int.(df[:, :is_in_damks])

println("Features shape: ", size(X_all), ", Labels shape: ", size(y_all))

Random.seed!(42)
n_samples = size(X_all, 1)
n_train = floor(Int, 0.8 * n_samples)
idx = shuffle(1:n_samples)
train_idx = idx[1:n_train]
test_idx  = idx[n_train+1:end]

X_train, y_train = X_all[train_idx, :], y_all[train_idx]
X_test,  y_test  = X_all[test_idx,  :], y_all[test_idx]

forest = build_forest(y_train, X_train, 2, 10, 0.7, -1; rng=42)

y_pred_test = apply_forest(forest, X_test)
acc = mean(y_pred_test .== y_test)
println("Test accuracy: ", round(acc, digits=4))
println(DecisionTree.confusion_matrix(y_test, y_pred_test))

# ---------------------------------------------------------------------------
# Evaluate on test graphs
# ---------------------------------------------------------------------------

predict_density_vals   = Float64[]
augmented_density_vals = Float64[]
optimal_density_vals   = Float64[]

@showprogress "Evaluating test graphs: " for gid in all_keys[training_size+1:end]
    data = raw_data[gid]
    G = build_julia_graph(Int(data["size"]), data["edges"])
    degs, avg_neigh_deg = compute_features(G)
    n = nv(G)

    X_pred = hcat(Float64.(degs), avg_neigh_deg, fill(Float64(n), n))
    y_pred = apply_forest(forest, X_pred)
    S = Set(v for (v, p) in enumerate(y_pred) if p == 1)

    pred_dens = compute_density_set(G, S)
    _, aug_dens = augment_damks(G, S, K, 0.1)
    optimal_dens = Float64(data["damks_density"])

    push!(predict_density_vals,   pred_dens)
    push!(augmented_density_vals, aug_dens)
    push!(optimal_density_vals,   optimal_dens)
end

println("Test graphs evaluated: $(length(optimal_density_vals))")

# ---------------------------------------------------------------------------
# Approximation ratio
# ---------------------------------------------------------------------------

approx_ratios = [
    o > 0 ? a / o : 0.0
    for (a, o) in zip(augmented_density_vals, optimal_density_vals)
]

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

plot_comparison(
    predict_density_vals, augmented_density_vals,
    "Predictor's output density", "Algorithm's output density",
    "outputs/damks_predictor_vs_augmented.png"
)

plot_comparison(
    augmented_density_vals, optimal_density_vals,
    "Algorithm's output density", "Optimal density (Ground Truth)",
    "outputs/damks_augmented_vs_optimal.png"
)

plot_comparison(
    predict_density_vals, optimal_density_vals,
    "Predictor's output density", "Optimal density (Ground Truth)",
    "outputs/damks_predictor_vs_optimal.png"
)

p_hist = histogram(approx_ratios,
    xlabel="Approximation Ratio (Augmented / Optimal)",
    ylabel="Number of Graphs",
    title="Approximation Ratio Distribution",
    bins=20, legend=false, color=:teal, alpha=0.7)
vline!(p_hist, [avg_ratio], color=:red, linestyle=:dash, linewidth=2,
    label="Avg: $(round(avg_ratio, digits=3))")
savefig(p_hist, "outputs/damks_approx_ratio_hist.png")

println("\nPlots saved to outputs/")
