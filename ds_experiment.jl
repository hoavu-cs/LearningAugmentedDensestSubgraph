using Graphs, GraphPlot, Colors, Compose
using JSON, Statistics, StatsBase, Plots, Random
using DecisionTree, DataFrames, CSV, ProgressMeter
import Cairo, Fontconfig

include("densest_subgraph.jl")

# ---------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------

file_path = "amazon0302_edges.json"
raw_data = JSON.parsefile(file_path)

# Convert "0" => [[1, 2], [1, 3], [2, 4], ...] to  "0" => [(1,2), (1,3), (2,4), ...]
parsed_graphs = Dict(
    graph_id => [(Int(e[1]), Int(e[2])) for e in edges]
    for (graph_id, edges) in raw_data
)

mkpath("outputs")
mkpath("datasets/ds")

K = 1000
ε = 0.25  # augmentation parameter
all_keys = collect(keys(parsed_graphs))
data_set = Dict(k => parsed_graphs[k] for k in all_keys[1:K])

# ---------------------------------------------------------------------------
# Visualize one graph with densest subgraph highlighted
# ---------------------------------------------------------------------------

graph_id_vis = ""
for gid in all_keys[1:K]
    if length(data_set[gid]) >= 150
        global graph_id_vis = gid
        break
    end
end

vis_edges = data_set[graph_id_vis]
node_list = sort(unique(vcat([[u, v] for (u, v) in vis_edges]...)))
node_to_idx = Dict(n => i for (i, n) in enumerate(node_list))
G_vis = SimpleGraph(length(node_list))
for (u, v) in vis_edges
    add_edge!(G_vis, node_to_idx[u], node_to_idx[v])
end

Sstar_vis, _ = densest_subgraph(G_vis)
Sstar_vis_set = Set(Sstar_vis)
node_colors = [v in Sstar_vis_set ? colorant"gray" : colorant"white" for v in 1:nv(G_vis)]
draw(PNG("outputs/graph_vis.png", 800, 600),
     gplot(G_vis, nodefillc=node_colors, nodestrokec=colorant"black", nodestrokelw=1.5))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""Build a SimpleGraph from (u, v) edge tuples with arbitrary integer node IDs.
Returns (G, node_list, node_to_idx)."""
function build_julia_graph(raw_edges)
    node_set = sort(unique(vcat([[u, v] for (u, v) in raw_edges]...)))
    node_to_idx = Dict(n => i for (i, n) in enumerate(node_set))
    G = SimpleGraph(length(node_set))
    for (u, v) in raw_edges
        add_edge!(G, node_to_idx[u], node_to_idx[v])
    end
    return G, node_set, node_to_idx
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

"""Compute density of a Set{Int} S in graph G."""
function compute_density_set(G, S)
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

"""Augment the predicted dense subgraph S by adding epsilon*|S|/(1-epsilon)
external nodes with the most connections into S, then trim by peeling the
lowest-degree node until density no longer improves."""
function augment_ds(G, S::Set{Int}, epsilon=0.1)
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

    # Trimming: iteratively remove the lowest-degree node within new_S
    best_S = copy(new_S)
    best_d = density(G, collect(new_S))
    while length(new_S) > 1
        min_v = argmin(v -> count(u -> u in new_S, neighbors(G, v)), collect(new_S))
        delete!(new_S, min_v)
        d = density(G, collect(new_S))
        if d > best_d
            best_d = d
            best_S = copy(new_S)
        end
    end

    return best_S, best_d
end

# ---------------------------------------------------------------------------
# Process training graphs and write node features to CSV
# ---------------------------------------------------------------------------

csv_path = "datasets/ds/node_features_with_labels.csv"
num_graphs = 0

open(csv_path, "w") do io
    println(io, "degree,avg_neighbor_degree,number_of_nodes,is_in_densest")

    @showprogress "Processing graphs: " for gid in all_keys[1:K]
        raw_edges = data_set[gid]
        if length(raw_edges) < 100
            continue
        end

        G, _, _ = build_julia_graph(raw_edges)
        degs, avg_neigh_deg = compute_features(G)
        Sstar, _ = densest_subgraph(G)
        Sstar_set = Set(Sstar)
        n = nv(G)

        for v in 1:n
            println(io, "$(degs[v]),$(avg_neigh_deg[v]),$n,$(v in Sstar_set ? 1 : 0)")
        end

        global num_graphs += 1
    end
end

println("Number of graphs processed: $num_graphs")

# ---------------------------------------------------------------------------
# Train a random forest classifier
# ---------------------------------------------------------------------------

df = CSV.read(csv_path, DataFrame)
X_all = Matrix{Float64}(df[:, [:degree, :avg_neighbor_degree, :number_of_nodes]])
y_all = Int.(df[:, :is_in_densest])

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
# Evaluate on test graphs K+1 to 2K
# ---------------------------------------------------------------------------

augmented_density_vals    = Float64[]
non_augmented_density_vals = Float64[]
peeling_density_vals      = Float64[]

test_size = 0

@showprogress "Evaluating test graphs: " for k in all_keys[K+1:2*K]
    raw_edges = parsed_graphs[k]
    if length(raw_edges) < 150
        continue
    end
    global test_size += 1

    G, _, _ = build_julia_graph(raw_edges)
    degs, avg_neigh_deg = compute_features(G)
    n = nv(G)

    X_pred = hcat(Float64.(degs), avg_neigh_deg, fill(Float64(n), n))
    y_pred = apply_forest(forest, X_pred)
    S = Set(v for (v, p) in enumerate(y_pred) if p == 1)

    push!(non_augmented_density_vals, compute_density_set(G, S))
    _, aug_dens = augment_ds(G, S, ε)
    push!(augmented_density_vals, aug_dens)

    _, peel_dens = densest_subgraph_peeling(G)
    push!(peeling_density_vals, peel_dens)
end

println("Test size: $test_size")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


max_val = max(maximum(peeling_density_vals), maximum(augmented_density_vals))
p1 = scatter(peeling_density_vals, augmented_density_vals,
    xlabel="Output density of Charikar's greedy algorithm",
    ylabel="Output density of Algorithm 1",
    alpha=0.7, markerstrokecolor=:black, label="Graphs", size=(500, 500))
plot!(p1, [0, max_val], [0, max_val], color=:gray, linestyle=:dash, label="Equal Density")
savefig(p1, "outputs/peeling_vs_augmented.png")

max_val2 = max(maximum(non_augmented_density_vals), maximum(augmented_density_vals))
p2 = scatter(non_augmented_density_vals, augmented_density_vals,
    xlabel="Output density given by the predictor",
    ylabel="Output density of Algorithm 1",
    alpha=0.7, markerstrokecolor=:black, label="Graphs", size=(500, 500))
plot!(p2, [0, max_val2], [0, max_val2], color=:gray, linestyle=:dash, label="Equal Density")
savefig(p2, "outputs/predictor_vs_augmented.png")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

improvement_over_non_aug = [
    na > 0 ? 100.0 * (a - na) / na : 0.0
    for (a, na) in zip(augmented_density_vals, non_augmented_density_vals)
]
improvement_over_peeling = [
    p > 0 ? 100.0 * (a - p) / p : 0.0
    for (a, p) in zip(augmented_density_vals, peeling_density_vals)
]

avg_improvement_non_aug  = mean(improvement_over_non_aug)
median_improve_non_aug   = median(improvement_over_non_aug)
avg_improvement_peeling  = mean(improvement_over_peeling)
median_improve_peeling   = median(improvement_over_peeling)

println("Reports:")
println("-----------------------------------------------------")
println("Average % improvement over predictor's solution: $(round(avg_improvement_non_aug, digits=2))%")
println("Average % improvement over Charikar's greedy: $(round(avg_improvement_peeling, digits=2))%")
println("-----------------------------------------------------")
println("Median % improvement over predictor's solution: $(round(median_improve_non_aug, digits=2))%")
println("Median % improvement over Charikar's greedy: $(round(median_improve_peeling, digits=2))%")

p3 = histogram(improvement_over_non_aug,
    title="Improvement over predictor's output",
    xlabel="Percent Improvement", ylabel="Count", legend=false)
savefig(p3, "outputs/hist_improvement_non_aug.png")

p4 = histogram(improvement_over_peeling,
    title="Improvement over Charikar's greedy algorithm's output",
    xlabel="Percent Improvement", ylabel="Count", legend=false)
savefig(p4, "outputs/hist_improvement_peeling.png")
