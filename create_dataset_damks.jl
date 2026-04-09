using Graphs, JSON, ProgressMeter
using .Threads

include("densest_subgraph.jl")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

input_file  = "twitch_edges.json"  # {"0": [[u,v],...], "1": [[u,v],...], ...}
output_file = "outputs/all.json"   # output for damks_experiment.jl
K           = 15                   # DamkS parameter k
min_edges   = 100                  # skip graphs with fewer edges
num_graphs  = 600                  # how many graphs to use (set to nothing to use all)

mkpath("outputs")

println("Running with $(nthreads()) thread(s).")

# ---------------------------------------------------------------------------
# Load graphs from JSON
# ---------------------------------------------------------------------------

println("Loading $input_file ...")
raw_data = JSON.parsefile(input_file)
all_keys = sort(collect(keys(raw_data)), by=k -> parse(Int, k))
println("Found $(length(all_keys)) graphs in file.")

# ---------------------------------------------------------------------------
# Step 1 (single-threaded): build SimpleGraphs, filter by min_edges
# ---------------------------------------------------------------------------

struct SubgraphData
    G::SimpleGraph{Int}
    node_list::Vector{Int}
    gid::String
end

subgraphs = SubgraphData[]

for gid in all_keys
    raw_edges = raw_data[gid]
    length(raw_edges) < min_edges && continue

    node_list   = sort(unique(vcat([[Int(e[1]), Int(e[2])] for e in raw_edges]...)))
    node_to_idx = Dict(n => i for (i, n) in enumerate(node_list))

    G = SimpleGraph(length(node_list))
    for e in raw_edges
        add_edge!(G, node_to_idx[Int(e[1])], node_to_idx[Int(e[2])])
    end

    push!(subgraphs, SubgraphData(G, node_list, gid))
end

n_qualified = length(subgraphs)
if num_graphs !== nothing
    if n_qualified < num_graphs
        @warn "Only $n_qualified graphs meet the min_edges=$min_edges requirement (needed $num_graphs)."
    end
    subgraphs = subgraphs[1:min(num_graphs, n_qualified)]
end

println("Qualified graphs (>= $min_edges edges): $n_qualified — using $(length(subgraphs))")

# ---------------------------------------------------------------------------
# Step 2 (multi-threaded): compute DamkS for each graph in parallel
# ---------------------------------------------------------------------------

results = Vector{Pair{String, Dict{String,Any}}}(undef, length(subgraphs))

progress = Progress(length(subgraphs), desc="Computing DamkS: ")
done_count = Threads.Atomic{Int}(0)
@threads for i in eachindex(subgraphs)
    sg = subgraphs[i]
    damks_vlist_idx, damks_density = densest_at_most_k_subgraph(sg.G, K)
    # Map 1-indexed vertices back to original node IDs
    damks_vlist = [sg.node_list[v] for v in damks_vlist_idx]
    results[i] = sg.gid => Dict(
        "size"          => nv(sg.G),
        "edges"         => [[src(e), dst(e)] for e in edges(sg.G)],
        "damks_vlist"   => damks_vlist,
        "damks_density" => damks_density,
    )
    n = Threads.atomic_add!(done_count, 1) + 1
    if n % 10 == 0
        println("$n / $(length(subgraphs)) graphs done")
    end
    next!(progress)
end

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

dataset = Dict(results)

open(output_file, "w") do f
    JSON.print(f, dataset)
end

densities = [v["damks_density"] for (_, v) in results]
sizes     = [v["size"]           for (_, v) in results]

println("Written $(length(results)) graphs to $output_file")
println("Node count    — min: $(minimum(sizes)),  max: $(maximum(sizes)),  mean: $(round(sum(sizes)/length(sizes), digits=1))")
println("DamkS density — min: $(round(minimum(densities), digits=3)),  max: $(round(maximum(densities), digits=3)),  mean: $(round(sum(densities)/length(densities), digits=3))")
