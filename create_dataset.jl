using JSON, Random, Graphs, GraphPlot, Compose, Colors
import Cairo, Fontconfig

# ---------------------------------------------------------------------------
# Load Cit-HepPh.txt
# ---------------------------------------------------------------------------

edges_raw = Tuple{Int,Int}[]
all_nodes = Set{Int}()

open("Amazon0302.txt") do f
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
num_graphs  = 5000
sample_size = 50

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
    length(nodes) > target && shuffle!(nodes)
    return nodes[1:min(target, length(nodes))]
end

dataset = Dict{String, Vector{Vector{Int}}}()

for i in 1:num_graphs
    seed    = all_nodes_vec[rand(1:length(all_nodes_vec))]
    sampled = Set(bfs_sample(adj, seed, sample_size))

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

open("data/Amazon0302.json", "w") do f
    JSON.print(f, dataset)
end

sizes = [length(v) for v in values(dataset)]
println("Written $(num_graphs) graphs to data/Cit-HepPh.json")
println("Edge counts — min: $(minimum(sizes)), max: $(maximum(sizes)), mean: $(round(sum(sizes)/length(sizes), digits=1))")
