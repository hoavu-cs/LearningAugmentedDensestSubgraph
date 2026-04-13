# ---------------------------------------------------------------------------
# Step 1 of the data pipeline: sample connected induced subgraphs from a large
# SNAP-format edge-list (e.g. Cit-HepPh.txt) and write them to a JSON file
# (e.g. data/Cit-HepPh.json) for use by create_labeled_dataset_damks.jl.
#
# For each sample, a random seed node is chosen and BFS is run to collect up
# to max_n nodes; the induced subgraph edges over those nodes are stored.
#
# Output format: { "0": [[u,v],...], "1": [[u,v],...], ... }
# All vertex IDs are original (not remapped); remapping to 1-based indices
# is done in create_labeled_dataset_damks.jl at load time.
#
# Usage:
#   julia create_dataset.jl [--input FILE] [--num_graphs N] [--max_n N]
#
# Defaults:
#   --input      Cit-HepPh.txt   source edge-list (SNAP format, #-commented headers)
#   --num_graphs 5000            number of subgraphs to sample
#   --max_n      100             maximum nodes per sampled subgraph
#
# Outputs:
#   data/<basename>.json         sampled subgraph dataset
#   outputs/graph_vis.png        visualization of the first sampled subgraph
# ---------------------------------------------------------------------------


using JSON, Random, Graphs, GraphPlot, Compose, Colors
import Cairo, Fontconfig

# ---------------------------------------------------------------------------
# Arguments: [--input FILE] [--num_graphs N] [--max_n N]
# ---------------------------------------------------------------------------

input_file = "Cit-HepPh.txt"
num_graphs = 5000
max_n      = 100

let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--input" && i < length(ARGS)
            global input_file = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--num_graphs" && i < length(ARGS)
            global num_graphs = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--max_n" && i < length(ARGS)
            global max_n = parse(Int, ARGS[i+1]); i += 2
        else
            global input_file = ARGS[i]; i += 1
        end
    end
end

# Derive output path: Foo.txt → data/Foo.json
_base       = splitext(basename(input_file))[1]
output_file = joinpath("data", _base * ".json")

# ---------------------------------------------------------------------------
# Load edge list
# ---------------------------------------------------------------------------

edges_raw = Tuple{Int,Int}[]
all_nodes = Set{Int}()

open(input_file) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        parts = split(strip(line))
        length(parts) < 2 && continue
        u, v = parse(Int, parts[1]), parse(Int, parts[2])
        u == v && continue   # skip self-loops
        push!(edges_raw, (u, v))
        push!(all_nodes, u)
        push!(all_nodes, v)
    end
end

println("Input: $input_file → output: $output_file (num_graphs=$num_graphs, max_n=$max_n)")
println("Loaded $(length(edges_raw)) edges over $(length(all_nodes)) nodes")

# Build adjacency list (undirected) for fast induced subgraph lookup
adj = Dict{Int, Vector{Int}}()
for (u, v) in edges_raw
    push!(get!(adj, u, Int[]), v)
    push!(get!(adj, v, Int[]), u)
end

all_nodes_vec = collect(all_nodes)

# ---------------------------------------------------------------------------
# Sample induced subgraphs
# ---------------------------------------------------------------------------

Random.seed!(42)

function bfs_sample(adj, seed, target)
    visited = Set{Int}([seed])
    queue   = [seed]
    head    = 1
    while head <= length(queue) && length(visited) < target
        v = queue[head]; head += 1
        for u in get(adj, v, Int[])
            if u ∉ visited
                push!(visited, u)
                push!(queue, u)
            end
        end
    end
    nodes = collect(visited)
    shuffle!(nodes)
    nodes = nodes[1:min(end, target)]
    return nodes
end

dataset = Dict{String, Vector{Vector{Int}}}()

for i in 1:num_graphs
    seed    = all_nodes_vec[rand(1:length(all_nodes_vec))]
    sampled = Set(bfs_sample(adj, seed, max_n))

    graph_edges = Vector{Int}[]
    for u in sampled
        for v in get(adj, u, Int[])
            if v in sampled && u < v
                push!(graph_edges, [u, v])
            end
        end
    end

    dataset[string(i - 1)] = graph_edges
end

# ---------------------------------------------------------------------------
# Plot one example graph
# ---------------------------------------------------------------------------

mkpath("outputs")
example_edges = dataset["0"]
ex_nodes      = sort(unique(vcat([e[1] for e in example_edges], [e[2] for e in example_edges])))
ex_idx        = Dict(n => i for (i, n) in enumerate(ex_nodes))
G_ex          = SimpleGraph(length(ex_nodes))
for e in example_edges
    add_edge!(G_ex, ex_idx[e[1]], ex_idx[e[2]])
end

draw(PNG("outputs/graph_vis.png", 600, 600),
    gplot(G_ex,
        nodefillc=colorant"lightblue",
        nodestrokec=colorant"black",
        nodestrokelw=1.0,
        edgestrokec=colorant"black",
        edgelinewidth=1.0,
        NODESIZE=0.04))

println("Saved example graph ($(nv(G_ex)) nodes, $(ne(G_ex)) edges) → outputs/graph_vis.png")

# ---------------------------------------------------------------------------
# Write to JSON
# ---------------------------------------------------------------------------

mkpath("data")
open(output_file, "w") do f
    JSON.print(f, dataset)
end

sizes = [length(v) for v in values(dataset)]
println("Written $(num_graphs) graphs to $output_file")
println("Edge counts — min: $(minimum(sizes)), max: $(maximum(sizes)), mean: $(round(sum(sizes)/length(sizes), digits=1))")
