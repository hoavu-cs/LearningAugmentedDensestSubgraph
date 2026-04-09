using JSON, Random

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
num_graphs  = 2000
sample_size = 5000

dataset = Dict{String, Vector{Vector{Int}}}()

for i in 1:num_graphs
    sampled = Set(randperm(length(all_nodes_vec))[1:sample_size] .|> j -> all_nodes_vec[j])

    graph_edges = Vector{Int}[]
    for u in sampled
        for v in get(adj, u, Int[])
            if v in sampled && u < v   # each edge once, undirected
                push!(graph_edges, [u, v])
            end
        end
    end

    dataset[string(i - 1)] = graph_edges
end

# ---------------------------------------------------------------------------
# Write to JSON
# ---------------------------------------------------------------------------

open("amazon0302_edges.json", "w") do f
    JSON.print(f, dataset)
end

sizes = [length(v) for v in values(dataset)]
println("Written $(num_graphs) graphs to amazon0302_edges.json")
println("Edge counts — min: $(minimum(sizes)), max: $(maximum(sizes)), mean: $(round(sum(sizes)/length(sizes), digits=1))")
