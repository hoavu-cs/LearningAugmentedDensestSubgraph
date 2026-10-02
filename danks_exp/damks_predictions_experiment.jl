# ---------------------------------------------------------------------------
# Learning-augmented Densest At-Most-k Subgraph (Algorithm 3 of
# "Learning-Augmented Graph Algorithms with Bounded L1 Error") on Twitch ego-nets.
#
# Predictor: a random forest giving p(v) ~ 1[v in H*] from node features
# (degree, avg neighbor degree, graph size, core number, local clustering).
# S = {v : p(v) >= 1/2}  (Lemma 3.1).
#
# Algorithm 3:
#   1. r <- min(ceil(eps/(1-eps) * |S|), |V \ S|)
#   2. U <- the r vertices of V \ S with the largest e(v, S)
#   3. T <- S ∪ U
#   4. while |T| > k: remove a vertex of minimum degree in G[T]
#   5. return T
#
# Compared against "predictor only" (output S as-is), plus the top-k vertices
# by p(v) as a reference point.
#
# Evaluation is 5-fold cross-validation over graphs, so every labeled graph is
# scored by a model that did not see it. eps is estimated per fold on a held-out
# calibration split of the training graphs as mean |S Δ H*| / |H*|.
#
# Usage (from repo root):
#   julia --project=. --threads auto danks_exp/damks_predictions_experiment.jl \
#       [edges.json] [label_dir] [K]
#
# Defaults:
#   edges.json  danks_exp/twitch_egos/twitch_edges.json
#   label_dir   danks_exp/twitch_egos_densest_k15_ilp   (one "<gid>.txt" per graph,
#               CSV "id, g" with 0-based node ids and g = 1[v in H*])
#   K           15
#
# Outputs (danks_exp/outputs/):
#   damks_pred_results.csv                    per-graph results
#   damks_pred_predictor_vs_algorithm.png     predictor-only vs Algorithm 3 density
#   damks_pred_ratio_hist.png                 approximation ratio distributions
#   damks_pred_eps_sweep.png                  Algorithm 3 ratio vs fixed eps
# ---------------------------------------------------------------------------

using Graphs, JSON, Statistics, Random, Printf
using DecisionTree, Plots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const SCRIPT_DIR = @__DIR__
edges_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(SCRIPT_DIR, "twitch_egos", "twitch_edges.json")
label_dir  = length(ARGS) >= 2 ? ARGS[2] : joinpath(SCRIPT_DIR, "twitch_egos_densest_k15_ilp")
const K    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 15

const N_FOLDS    = 5
const CALIB_FRAC = 0.2
const EPS_MAX    = 0.9
const EPS_SWEEP  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
const N_TREES    = 100

out_dir = joinpath(SCRIPT_DIR, "outputs")
mkpath(out_dir)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""Build a 1-indexed SimpleGraph from a 0-indexed [[u,v],...] edge list."""
function build_graph(edges)
    n = maximum(max(e[1], e[2]) for e in edges) + 1
    G = SimpleGraph(n)
    for e in edges
        add_edge!(G, Int(e[1]) + 1, Int(e[2]) + 1)
    end
    return G
end

"""Read a "id, g" label file into a 1-indexed 0/1 vector of length n."""
function read_labels(path, n)
    g = zeros(Int, n)
    for line in Iterators.drop(eachline(path), 1)
        isempty(strip(line)) && continue
        id, lab = parse.(Int, strip.(split(line, ",")))
        g[id + 1] = lab
    end
    return g
end

"""Per-vertex feature matrix: degree, avg neighbor degree, n, core number, clustering."""
function node_features(G)
    n = nv(G)
    degs = degree(G)
    avg_nd = [isempty(neighbors(G, v)) ? 0.0 : mean(degs[neighbors(G, v)]) for v in 1:n]
    return hcat(Float64.(degs), avg_nd, fill(Float64(n), n),
                Float64.(core_number(G)), local_clustering_coefficient(G))
end

"""Density |E(S)|/|S| of a vertex set S."""
function density_set(G, S)
    isempty(S) && return 0.0
    inS = falses(nv(G)); inS[collect(S)] .= true
    e = 0
    for v in S, u in neighbors(G, v)
        inS[u] && (e += 1)
    end
    return (e / 2) / length(S)
end

"""
Remove a minimum-degree vertex of G[T] until |T| <= k (trimming step).
Degrees inside T are maintained incrementally.
"""
function trim_to_k!(G, T::Set{Int}, k::Int)
    length(T) <= k && return T
    deg_T = Dict(v => count(u -> u in T, neighbors(G, v)) for v in T)
    while length(T) > k
        v_min = argmin(v -> (deg_T[v], v), collect(T))
        delete!(T, v_min)
        delete!(deg_T, v_min)
        for u in neighbors(G, v_min)
            haskey(deg_T, u) && (deg_T[u] -= 1)
        end
    end
    return T
end

"""Algorithm 3: Densest at-most-k Subgraph from Predictions."""
function damks_from_predictions(G, S::Set{Int}, k::Int, eps::Float64)
    outside = [v for v in vertices(G) if v ∉ S]
    r = min(ceil(Int, eps / (1 - eps) * length(S)), length(outside))

    # e(v, S) for v ∉ S; pick the r largest
    e_vS = [count(u -> u in S, neighbors(G, v)) for v in outside]
    order = sortperm(e_vS, rev=true)
    U = outside[order[1:r]]

    T = union(S, U)
    trim_to_k!(G, T, k)
    return T
end

"""Train a probability forest on the given graphs."""
function train_forest(graphs, ids; seed=42)
    X = reduce(vcat, [graphs[i].X for i in ids])
    y = reduce(vcat, [graphs[i].g for i in ids])
    return build_forest(y, X, 2, N_TREES, 0.7, -1; rng=seed)
end

"""Predicted membership probabilities p(v) = P[v in H*]."""
predict_p(forest, X) = apply_forest_proba(forest, X, [0, 1])[:, 2]

"""Predicted set S = {v : p(v) >= 1/2}."""
predicted_set(p) = Set(findall(>=(0.5), p))

"""Relative symmetric difference |S Δ H*| / |H*|."""
rel_symdiff(S, H) = length(symdiff(S, H)) / length(H)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

println("Loading graphs from $edges_file ...")
all_edges = JSON.parsefile(edges_file)

label_files = filter(f -> endswith(f, ".txt"), readdir(label_dir))
gids = sort([replace(f, ".txt" => "") for f in label_files], by=s -> parse(Int, s))

graphs = map(gids) do gid
    G = build_graph(all_edges[gid])
    g = read_labels(joinpath(label_dir, gid * ".txt"), nv(G))
    H = Set(findall(==(1), g))
    @assert 1 <= length(H) <= K "graph $gid: |H*| = $(length(H)) not in [1, $K]"
    (gid=gid, G=G, g=g, H=H, opt=density_set(G, H), X=node_features(G))
end
all_edges = nothing

n_graphs = length(graphs)
println("Labeled graphs: $n_graphs  (n ∈ $(extrema(nv(x.G) for x in graphs)), |H*| ∈ $(extrema(length(x.H) for x in graphs)), K = $K)")

# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

Random.seed!(42)
perm  = shuffle(1:n_graphs)
folds = [perm[f:N_FOLDS:end] for f in 1:N_FOLDS]

# Warm up (JIT) so timings below measure the algorithm, not compilation
damks_from_predictions(graphs[1].G, graphs[1].H, K, 0.3)

res_pred_dens  = zeros(n_graphs)     # predictor only: S
res_pred_size  = zeros(Int, n_graphs)
res_topk_dens  = zeros(n_graphs)     # top-k by p(v)
res_alg_dens   = zeros(n_graphs)     # Algorithm 3 with estimated eps
res_alg_size   = zeros(Int, n_graphs)
res_eps        = zeros(n_graphs)
res_symdiff    = zeros(n_graphs)     # |S Δ H*| / |H*|
res_l1         = zeros(n_graphs)     # sum |p(v) - g(v)| / |H*|
res_alg_time   = zeros(n_graphs)
res_e2e_time   = zeros(n_graphs)
res_sweep      = zeros(n_graphs, length(EPS_SWEEP))

for (f, test_ids) in enumerate(folds)
    train_ids = setdiff(perm, test_ids)

    # Estimate eps on a calibration split of the training graphs
    n_cal   = round(Int, CALIB_FRAC * length(train_ids))
    cal_ids = train_ids[1:n_cal]
    fit_ids = train_ids[n_cal+1:end]
    cal_forest = train_forest(graphs, fit_ids)
    eps_hat = mean(rel_symdiff(predicted_set(predict_p(cal_forest, graphs[i].X)), graphs[i].H)
                   for i in cal_ids)
    eps_hat = clamp(eps_hat, 0.0, EPS_MAX)

    forest = train_forest(graphs, train_ids)
    @printf("Fold %d: train %d graphs, test %d graphs, eps_hat = %.3f\n",
            f, length(train_ids), length(test_ids), eps_hat)

    Threads.@threads for i in test_ids
        x = graphs[i]
        p = predict_p(forest, x.X)
        S = predicted_set(p)

        res_pred_dens[i] = density_set(x.G, S)
        res_pred_size[i] = length(S)
        res_symdiff[i]   = rel_symdiff(S, x.H)
        res_l1[i]        = sum(abs.(p .- x.g)) / length(x.H)
        res_topk_dens[i] = density_set(x.G, partialsortperm(p, 1:min(K, nv(x.G)), rev=true))
        res_eps[i]       = eps_hat

        t0 = time_ns()
        T = damks_from_predictions(x.G, S, K, eps_hat)
        res_alg_time[i] = (time_ns() - t0) / 1e6

        # End-to-end: features + forest inference + algorithm
        t0 = time_ns()
        damks_from_predictions(x.G, predicted_set(predict_p(forest, node_features(x.G))), K, eps_hat)
        res_e2e_time[i] = (time_ns() - t0) / 1e6
        res_alg_dens[i] = density_set(x.G, T)
        res_alg_size[i] = length(T)

        for (j, e) in enumerate(EPS_SWEEP)
            res_sweep[i, j] = density_set(x.G, damks_from_predictions(x.G, S, K, e))
        end
    end
end

@assert all(res_alg_size .<= K)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Labels are exact optima, so no feasible (|T| <= K) output may beat them
opt = [x.opt for x in graphs]
best_feasible = max.(res_alg_dens, res_topk_dens, ifelse.(res_pred_size .<= K, res_pred_dens, 0.0),
                     vec(maximum(res_sweep, dims=2)))
n_beaten = sum(best_feasible .> opt .+ 1e-9)
n_beaten > 0 && @warn "$n_beaten labels are beaten by a feasible output; labels are not optimal"
ratio(d) = d ./ opt

r_pred, r_alg, r_topk = ratio(res_pred_dens), ratio(res_alg_dens), ratio(res_topk_dens)
infeasible = res_pred_size .> K
feasible_pred = .!infeasible

summ(r) = @sprintf("%.4f  %.4f  %.4f  %5.1f%%  %5.1f%%",
                   mean(r), median(r), minimum(r), 100 * mean(r .>= 0.95), 100 * mean(r .>= 0.999))

println("\nPredictor quality (cross-validated):")
println("-----------------------------------------------------")
@printf("mean |S Δ H*| / |H*|          : %.4f\n", mean(res_symdiff))
@printf("mean L1 error / |H*|          : %.4f\n", mean(res_l1))
@printf("mean |S| = %.2f, mean |H*| = %.2f\n", mean(res_pred_size), mean(length(x.H) for x in graphs))
@printf("|S| > K (predictor infeasible): %d / %d\n", sum(infeasible), n_graphs)
@printf("|S| = 0 (empty prediction)    : %d / %d\n", sum(res_pred_size .== 0), n_graphs)
@printf("eps_hat per fold              : %s\n", join([@sprintf("%.3f", res_eps[fo[1]]) for fo in folds], ", "))

println("\nApproximation ratio  d(T) / d(H*)   over $n_graphs graphs")
println("-----------------------------------------------------------------------")
println("method                         mean    median  min     >=0.95  optimal")
println("LA-DamkS (eps_hat)             ", summ(r_alg))
println("Predictor only (S)             ", summ(r_pred))
println("  ... on graphs with |S|<=K    ", summ(r_pred[feasible_pred]), "   (n=$(sum(feasible_pred)))")
println("  ... infeasible counted as 0  ", summ(ifelse.(infeasible, 0.0, r_pred)))
println("Top-K by p(v)                  ", summ(r_topk))
println("-----------------------------------------------------------------------")

wins   = sum(res_alg_dens .> res_pred_dens .+ 1e-9)
ties   = sum(abs.(res_alg_dens .- res_pred_dens) .<= 1e-9)
losses = sum(res_alg_dens .< res_pred_dens .- 1e-9)
println("\nLA-DamkS vs predictor only (per graph): win $wins / tie $ties / loss $losses")
@printf("  losses where predictor's S was infeasible (|S|>K): %d\n", sum((res_alg_dens .< res_pred_dens .- 1e-9) .& infeasible))
@printf("  mean density gain: %+.4f   mean ratio gain: %+.4f\n",
        mean(res_alg_dens .- res_pred_dens), mean(r_alg .- r_pred))

println("\nLA-DamkS with fixed eps:")
for (j, e) in enumerate(EPS_SWEEP)
    @printf("  eps = %.1f   mean ratio %.4f   optimal %5.1f%%\n",
            e, mean(ratio(res_sweep[:, j])), 100 * mean(ratio(res_sweep[:, j]) .>= 0.999))
end

@printf("\nLA-DamkS runtime (excluding prediction): mean %.3f ms, median %.3f ms\n",
        mean(res_alg_time), median(res_alg_time))
@printf("End-to-end (features + inference + LA-DamkS): mean %.3f ms, median %.3f ms, max %.3f ms, total %.3f s\n",
        mean(res_e2e_time), median(res_e2e_time), maximum(res_e2e_time), sum(res_e2e_time) / 1e3)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

csv_path = joinpath(out_dir, "damks_pred_results.csv")
open(csv_path, "w") do io
    println(io, "gid,n,m,opt_size,opt_density,pred_size,pred_density,alg_size,alg_density,topk_density,eps_hat,symdiff_rel,l1_rel,alg_ms,e2e_ms")
    for (i, x) in enumerate(graphs)
        println(io, join([x.gid, nv(x.G), ne(x.G), length(x.H), x.opt, res_pred_size[i], res_pred_dens[i],
                          res_alg_size[i], res_alg_dens[i], res_topk_dens[i],
                          res_eps[i], res_symdiff[i], res_l1[i], res_alg_time[i], res_e2e_time[i]], ","))
    end
end

lim = max(maximum(res_pred_dens), maximum(res_alg_dens)) * 1.05
p1 = scatter(res_pred_dens[feasible_pred], res_alg_dens[feasible_pred],
    xlabel="Predictor-only density d(S)", ylabel="Learning-augmented DamkS density d(T)",
    title="Predictor only vs learning-augmented (K=$K)", alpha=0.5,
    markerstrokecolor=:black, label="|S| ≤ K", size=(550, 550))
scatter!(p1, res_pred_dens[infeasible], res_alg_dens[infeasible],
    alpha=0.6, marker=:utriangle, label="|S| > K (infeasible S)")
plot!(p1, [0, lim], [0, lim], color=:red, linestyle=:dash, label="y = x")
savefig(p1, joinpath(out_dir, "damks_pred_predictor_vs_algorithm.png"))

bins = range(0, max(1.0, maximum(r_pred)) + 0.025, step=0.025)
p2 = histogram(r_pred, bins=bins, alpha=0.5, label="Predictor only",
    xlabel="d(output) / d(H*)", ylabel="Number of graphs", title="Approximation ratio (K=$K)")
histogram!(p2, r_alg, bins=bins, alpha=0.5, label="Learning-augmented DamkS")
savefig(p2, joinpath(out_dir, "damks_pred_ratio_hist.png"))

p3 = plot(EPS_SWEEP, [mean(ratio(res_sweep[:, j])) for j in eachindex(EPS_SWEEP)],
    marker=:circle, label="Learning-augmented DamkS", xlabel="ε", ylabel="Mean d(T) / d(H*)",
    title="Sensitivity to ε (K=$K)")
hline!(p3, [mean(ifelse.(infeasible, 0.0, r_pred))], linestyle=:dash, label="Predictor only (infeasible = 0)")
savefig(p3, joinpath(out_dir, "damks_pred_eps_sweep.png"))

println("\nResults written to $out_dir")
