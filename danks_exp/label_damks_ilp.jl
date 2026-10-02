# ---------------------------------------------------------------------------
# Exact Densest At-Most-k Subgraph labels via ILP (JuMP + HiGHS).
#
# The objective e(H)/|H| is fractional, so we use Dinkelbach's method with an
# integer parameter: given the current best set with e edges on s vertices, solve
#
#   max  s * sum(y) - e * sum(x)
#   s.t. y_uv <= x_u,  y_uv <= x_v                       for every edge uv
#        sum_{u in N(v)} y_uv <= sum(x) - x_v             (deg in H <= |H| - 1)
#        sum_{u in N(v)} y_uv <= (k - 1) x_v
#        1 <= sum(x) <= k,   x binary,  y in [0, 1]
#
# All coefficients are integers, so any positive optimum is >= 1 and yields a
# strictly denser set; an optimum of 0 proves the current set optimal. The
# starting set comes from greedy peeling.
#
# Usage (from repo root):
#   julia --project=danks_exp/ilp_env --threads auto danks_exp/label_damks_ilp.jl \
#       [edges.json] [ids_dir] [out_dir] [K] [time_limit_sec]
#
# Defaults:
#   edges.json      danks_exp/twitch_egos/twitch_edges.json
#   ids_dir         danks_exp/twitch_egos_densest_k15   (graphs to label = "<gid>.txt" files here)
#   out_dir         danks_exp/twitch_egos_densest_k15_ilp
#   K               15
#   time_limit_sec  600 per ILP solve
#
# Outputs:
#   <out_dir>/<gid>.txt     "id, g" with 0-based node ids, g = 1[v in H*]
#   <out_dir>/summary.csv   per-graph density, size, iterations, time, status
# ---------------------------------------------------------------------------

using Graphs, JSON, JuMP, HiGHS, Printf

const SCRIPT_DIR = @__DIR__
edges_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(SCRIPT_DIR, "twitch_egos", "twitch_edges.json")
ids_dir    = length(ARGS) >= 2 ? ARGS[2] : joinpath(SCRIPT_DIR, "twitch_egos_densest_k15")
out_dir    = length(ARGS) >= 3 ? ARGS[3] : joinpath(SCRIPT_DIR, "twitch_egos_densest_k15_ilp")
const K          = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15
const TIME_LIMIT = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 600.0

mkpath(out_dir)

"""Build a 1-indexed SimpleGraph from a 0-indexed [[u,v],...] edge list."""
function build_graph(edges)
    n = maximum(max(e[1], e[2]) for e in edges) + 1
    G = SimpleGraph(n)
    for e in edges
        add_edge!(G, Int(e[1]) + 1, Int(e[2]) + 1)
    end
    return G
end

"""Number of edges induced by vertex set H."""
edges_in(G, H) = count(e -> src(e) in H && dst(e) in H, edges(G))

"""Greedy peeling: best density among peeled sets of size <= k."""
function greedy_peel_k(G, k)
    T = Set(vertices(G))
    deg_T = Dict(v => degree(G, v) for v in T)
    m_T = ne(G)
    best_T, best_e = Set{Int}(), 0
    while !isempty(T)
        if length(T) <= k && m_T * max(length(best_T), 1) > best_e * length(T)
            best_T, best_e = copy(T), m_T
        end
        v = argmin(v -> (deg_T[v], v), collect(T))
        m_T -= deg_T[v]
        delete!(T, v); delete!(deg_T, v)
        for u in neighbors(G, v)
            haskey(deg_T, u) && (deg_T[u] -= 1)
        end
    end
    isempty(best_T) && (best_T = Set([1]))
    return best_T
end

"""Exact DamkS by Dinkelbach + ILP. Returns (H, e(H), iterations, status)."""
function damks_ilp(G, k; time_limit=TIME_LIMIT)
    n = nv(G)
    E = [(src(e), dst(e)) for e in edges(G)]
    inc = [Int[] for _ in 1:n]
    for (i, (u, v)) in enumerate(E)
        push!(inc[u], i); push!(inc[v], i)
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "threads", 1)
    set_attribute(model, "time_limit", time_limit)
    set_attribute(model, "mip_rel_gap", 0.0)
    set_attribute(model, "mip_abs_gap", 0.5)   # objective is integer

    @variable(model, x[1:n], Bin)
    @variable(model, 0 <= y[1:length(E)] <= 1)
    for (i, (u, v)) in enumerate(E)
        @constraint(model, y[i] <= x[u])
        @constraint(model, y[i] <= x[v])
    end
    for v in 1:n
        @constraint(model, sum(y[i] for i in inc[v]; init=0) <= sum(x) - x[v])
        @constraint(model, sum(y[i] for i in inc[v]; init=0) <= (k - 1) * x[v])
    end
    @constraint(model, 1 <= sum(x) <= k)

    H = greedy_peel_k(G, k)
    e_H = edges_in(G, H)
    iters = 0
    while true
        iters += 1
        s = length(H)
        @objective(model, Max, s * sum(y) - e_H * sum(x))
        for v in 1:n
            set_start_value(x[v], v in H ? 1.0 : 0.0)
        end
        optimize!(model)
        st = termination_status(model)
        st == MOI.OPTIMAL || return H, e_H, iters, string(st)
        objective_value(model) < 0.5 && return H, e_H, iters, "OPTIMAL"

        H_new = Set(v for v in 1:n if value(x[v]) > 0.5)
        e_new = edges_in(G, H_new)
        @assert e_new * s > e_H * length(H_new) "ILP returned a non-improving set"
        H, e_H = H_new, e_new
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

println("Loading graphs from $edges_file ...")
all_edges = JSON.parsefile(edges_file)
gids = sort([replace(f, ".txt" => "") for f in readdir(ids_dir) if endswith(f, ".txt") && f != "summary.csv"],
            by=s -> parse(Int, s))
graphs = [build_graph(all_edges[g]) for g in gids]
all_edges = nothing
println("Graphs to label: $(length(gids)), K = $K, threads = $(Threads.nthreads())")

damks_ilp(path_graph(5), 3)   # warm up JIT

n_g = length(gids)
res_H = Vector{Set{Int}}(undef, n_g)
res_e = zeros(Int, n_g); res_it = zeros(Int, n_g)
res_t = zeros(n_g); res_st = fill("", n_g)
done = Threads.Atomic{Int}(0)
t_start = time()

Threads.@threads :dynamic for i in 1:n_g
    G = graphs[i]
    t0 = time()
    H, e_H, it, st = damks_ilp(G, K)
    res_t[i] = time() - t0
    res_H[i], res_e[i], res_it[i], res_st[i] = H, e_H, it, st

    open(joinpath(out_dir, gids[i] * ".txt"), "w") do io
        println(io, "id, g")
        for v in 1:nv(G)
            println(io, "$(v - 1), $(v in H ? 1 : 0)")
        end
    end

    d = Threads.atomic_add!(done, 1) + 1
    d % 50 == 0 && @printf("  %d / %d done (%.0fs elapsed)\n", d, n_g, time() - t_start)
end

open(joinpath(out_dir, "summary.csv"), "w") do io
    println(io, "gid,n,m,size,edges,density,iterations,time_sec,status")
    for i in 1:n_g
        println(io, join([gids[i], nv(graphs[i]), ne(graphs[i]), length(res_H[i]), res_e[i],
                          res_e[i] / length(res_H[i]), res_it[i], round(res_t[i], digits=4), res_st[i]], ","))
    end
end

n_opt = count(==("OPTIMAL"), res_st)
@printf("\nOptimal: %d / %d   solve time: mean %.3fs, median %.3fs, max %.2fs   wall %.1fs\n",
        n_opt, n_g, sum(res_t) / n_g, sort(res_t)[cld(n_g, 2)], maximum(res_t), time() - t_start)
n_opt < n_g && println("NOT proven optimal: ", join(gids[res_st .!= "OPTIMAL"], ", "))
println("Labels written to $out_dir")
