using Graphs, JSON, ProgressMeter, Random
using .Threads

include("densest_subgraph.jl")

# ---------------------------------------------------------------------------
# Global variables and parameters
# ---------------------------------------------------------------------------

input_file = length(ARGS) >= 1 ? ARGS[1] : "data/input.json"
K          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15

# Derive output filenames: data/input.json → data/input_train.json, data/input_val.json
_base                  = splitext(input_file)[1]
training_output_file   = _base * "_train.json"
validation_output_file = _base * "_val.json"
train_size = 200   # how many graphs to use for training
val_size   = 200   # how many graphs to use for validation

mkpath("data")
println("Running with $(nthreads()) thread(s).")

# ---------------------------------------------------------------------------
# Load graphs from JSON
# ---------------------------------------------------------------------------

println("Loading $input_file ...")
raw_data  = JSON.parsefile(input_file)
graph_ids = sort(collect(keys(raw_data)), by=k -> parse(Int, k))
println("Found $(length(graph_ids)) graphs in file.")

# ---------------------------------------------------------------------------
# Build graphs, filter by min_edges
# ---------------------------------------------------------------------------

struct SubgraphData
    G::SimpleGraph{Int}
    node_list::Vector{Int}
    gid::String
end

subgraphs = SubgraphData[]

for gid in graph_ids
    raw_edges = raw_data[gid]

    all_ids     = [Int(endpoint) for e in raw_edges for endpoint in e]
    node_list   = sort(unique(all_ids))
    node_to_idx = Dict(n => i for (i, n) in enumerate(node_list))  # original ID → 1-based index

    G = SimpleGraph(length(node_list))
    for e in raw_edges
        add_edge!(G, node_to_idx[Int(e[1])], node_to_idx[Int(e[2])])
    end

    push!(subgraphs, SubgraphData(G, node_list, gid))
end

n_qualified = length(subgraphs)
n_needed    = train_size + val_size

if n_qualified < n_needed
    @warn "Only $n_qualified graphs meet the requirement (needed $n_needed)."
end

shuffle!(subgraphs)
subgraphs = subgraphs[1:min(n_needed, n_qualified)]

# ---------------------------------------------------------------------------
# Compute DamkS for each graph in parallel (training and validation separately)
# ---------------------------------------------------------------------------

timeout_sec = 2 * 60   # skip graphs that take longer than this

function compute_damks_parallel(sgs, label)
    res        = Vector{Union{Nothing, Pair{String, Dict{String,Any}}}}(nothing, length(sgs))
    done_count = Threads.Atomic{Int}(0)

    start_time = time()
    @threads for i in eachindex(sgs)
        sg = sgs[i]
        damks_vlist_idx, damks_density = densest_at_most_k_subgraph(sg.G, K, timeout_sec)

        if damks_vlist_idx === nothing
            println("Graph $(sg.gid) timed out — skipping.")
        else
            res[i] = sg.gid => Dict(
                "size"          => nv(sg.G),
                "edges"         => [[src(e), dst(e)] for e in edges(sg.G)],
                "damks_vlist"   => damks_vlist_idx,  # already 1-indexed, matches stored edges
                "damks_density" => damks_density,
            )
            n = Threads.atomic_add!(done_count, 1) + 1
            if n % 10 == 0
                println("$n / $(length(sgs)) $label graphs done")
            end
        end
    end

    elapsed = round(time() - start_time, digits=1)
    skipped = count(isnothing, res)
    println("\n$label finished in $(elapsed)s (skipped $skipped graphs due to timeout)")
    return [r for r in res if r !== nothing]
end

train_subgraphs = subgraphs[1:min(train_size, length(subgraphs))]
val_subgraphs   = subgraphs[min(train_size, length(subgraphs))+1:end]

train_results = compute_damks_parallel(train_subgraphs, "Training")
val_results   = compute_damks_parallel(val_subgraphs,   "Validation")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

open(training_output_file, "w") do f
    JSON.print(f, Dict(train_results))
end

open(validation_output_file, "w") do f
    JSON.print(f, Dict(val_results))
end

function print_stats(label, res)
    println("$label ($(length(res)) graphs):")
    if isempty(res)
        println("  (no graphs)")
        return
    end
    densities = [v["damks_density"] for (_, v) in res]
    sizes     = [v["size"]           for (_, v) in res]
    println("  Node count    — min: $(minimum(sizes)),  max: $(maximum(sizes)),  mean: $(round(sum(sizes)/length(sizes), digits=1))")
    println("  DamkS density — min: $(round(minimum(densities), digits=3)),  max: $(round(maximum(densities), digits=3)),  mean: $(round(sum(densities)/length(densities), digits=3))")
end

println()
print_stats("Training   → $training_output_file",   train_results)
print_stats("Validation → $validation_output_file", val_results)
