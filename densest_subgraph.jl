using Graphs, GraphsFlows
using Combinatorics
using SparseArrays

"""
    create_aux_graph(G::AbstractGraph, λ::Float64)

Create an auxiliary graph H for the Goldberg densest subgraph algorithm.

# Arguments
- `G::AbstractGraph`: The input undirected graph.
- `λ::Float64`: The density parameter λ for the max flow formulation.

# Returns
- A tuple `(H, cap, s, t)` containing:
  - `H::DiGraph`: The auxiliary directed graph with n+2 vertices.
  - `cap::SparseMatrixCSC`: Capacity matrix for the flow network.
  - `s::Int`: Source vertex (n+1).
  - `t::Int`: Sink vertex (n+2).

# Description
The auxiliary graph is constructed for the binary search on λ in Goldberg's algorithm.
The source is connected to all original vertices with capacity equal to their degree,
and all vertices are connected to the sink with capacity 2λ. Original edges have
capacity 1 in both directions.
"""
function create_aux_graph(G::AbstractGraph, λ::Float64)
    n = nv(G)
    N = n + 2
    s = n + 1
    t = n + 2

    H = DiGraph(N)
    cap = spzeros(Float64, N, N)

    for v ∈ 1:n
        add_edge!(H, s, v)
        cap[s, v] = degree(G, v)
        add_edge!(H, v, t)           
        cap[v, t] = 2.0 * λ
    end

    for e ∈ edges(G)
        u, v = src(e), dst(e)
        add_edge!(H, u, v); cap[u, v] = 1.0
        add_edge!(H, v, u); cap[v, u] = 1.0
    end

    return H, cap, s, t
end

"""
    density(G::AbstractGraph, S::Vector{Int})

Compute the density of a subgraph S in G, defined as |E(S)|/|S|.

# Arguments
- `G::AbstractGraph`: The input undirected graph.
- `S::Vector{Int}`: A vector of vertex indices representing the subgraph.

# Returns
- `Float64`: The density of the subgraph, i.e., the ratio of edges to vertices.
  Returns 0.0 if S is empty.
"""
function density(G::AbstractGraph, S::Vector{Int})
    if isempty(S)
        return 0.0
    end
    Sset = Set(S)

    eS = 0  
    for v ∈ S
        for u ∈ neighbors(G, v)
            if u ∈ Sset
                eS += 1
            end
        end
    end

    eS = eS / 2.0  
    return eS / length(S)
end

"""
    densest_subgraph(G::AbstractGraph, num_iterations::Int = 40; algorithm::Symbol = :goldberg)

Find the densest subgraph in an undirected graph using Goldberg's algorithm.

# Arguments
- `G::AbstractGraph`: The input undirected graph.
- `num_iterations::Int`: Maximum number of iterations for binary search (default: 40).
- `algorithm::Symbol`: Algorithm to use (currently only `:goldberg` is supported).

# Returns
- `Tuple{Vector{Int}, Float64}`: A tuple containing:
  - `best_S`: Vector of vertex indices forming the densest subgraph.
  - `best_density`: The density of the found subgraph (|E(S)|/|S|).

# Throws
- `ArgumentError`: If `G` is a directed graph.

# Description
Goldberg's algorithm uses a max-flow formulation with binary search to find the
densest subgraph. It creates an auxiliary flow network and uses min-cut to
determine the optimal subgraph for each density value λ.
"""
function densest_subgraph(G::AbstractGraph, num_iterations::Int = 40, algorithm=:goldberg)
    if is_directed(G)
        throw(ArgumentError("densest_subgraph only supports undirected graphs"))
    end

    n = nv(G)
    m = ne(G)

    low = 0.0
    high = maximum(degree(G)) / 2.0

    best_S = collect(1:n)
    best_λ = 0.0

    while high - low ≥ 1/(n * (n - 1))
        mid = (low + high) / 2.0

        H, cap, s, t = create_aux_graph(G, mid)

        part_s, part_t, cut_value = GraphsFlows.mincut(H, s, t, cap, PushRelabelAlgorithm())
        S = [v for v in part_s if 1 ≤ v ≤ n]

        # S non-empty means a subgraph denser than mid was found (min-cut < 2m).
        # S empty means all vertices are on the t-side (min-cut = 2m), so no
        # subgraph with density > mid exists.
        if !isempty(S)
            low = mid
            best_λ = mid
            best_S = S
        else
            high = mid
        end
    end

    return best_S, density(G, best_S)
end

"""
    densest_subgraph_peeling(G::AbstractGraph)

Find the densest subgraph using Charikar's peeling algorithm (1/2-approximation).

# Arguments
- `G::AbstractGraph`: The input undirected graph.

# Returns
- `Tuple{Vector{Int}, Float64}`: A tuple containing:
  - `best_S`: Vector of vertex indices forming the densest subgraph.
  - `best_density`: The density of the found subgraph (|E(S)|/|S|).

# Throws
- `ArgumentError`: If `G` is a directed graph.

# Description
Charikar's peeling algorithm iteratively removes vertices with minimum degree,
tracking the density at each step. This provides a 1/2-approximation to the
densest subgraph problem. The algorithm runs in O(n+m) time.
"""
function densest_subgraph_peeling(G::AbstractGraph)
    if is_directed(G)
        throw(ArgumentError("densest_subgraph_peeling only supports undirected graphs"))
    end

    H = copy(G)
    n = nv(H)

    active = Set(1:n)
    best_S = copy(active)
    best_density = density(H, collect(best_S))
    remaining = n

    Δ = maximum(degree(H))
    B = Dict(d => Set(v for v in vertices(H) if degree(H, v) == d) for d in 0:Δ)
    d = 0

    while remaining > 0
        current_density = ne(H) / remaining
        if current_density > best_density
            best_density = current_density
            best_S = copy(active)
        end
        
        # Find the next minimum degree d
        d > Δ && break
        while isempty(B[d])
            d += 1
            d > Δ && break
        end
        d > Δ && break

        v = pop!(B[d]) # Remove a minimum degree vertex v
        for u in collect(neighbors(H, v))
            du = degree(H, u)
            rem_edge!(H, u, v)
            delete!(B[du], u) 
            push!(B[du - 1], u)
        end
        delete!(active, v)
        d = max(d-1, 0)
        remaining -= 1
    end

    return best_S, best_density
end

"""
    densest_at_most_k_subgraph(G::AbstractGraph, k::Int)

Find a subset S of vertices with |S| ≤ k that maximizes density(S) = |E(S)|/|S|.

# Arguments
- `G::AbstractGraph`: The input undirected graph.
- `k::Int`: Maximum size of the subgraph.

# Returns
- `Tuple{Vector{Int}, Float64}`: A tuple containing:
  - `best_S`: Vector of vertex indices forming the densest subgraph with at most k vertices.
  - `best_density`: The density of the found subgraph.

# Throws
- `ArgumentError`: If `G` is a directed graph.

# Description
The algorithm uses a combination of pruning and brute force:
1. Iteratively remove vertices with minimum degree until only k vertices remain.
2. Remove vertices with degree strictly smaller than the current best density.
3. Perform brute force search on the remaining graph.
"""
function densest_at_most_k_subgraph(G::AbstractGraph, k::Int)
    if is_directed(G)
        throw(ArgumentError("densest_at_most_k_subgraph only supports undirected graphs"))
    end

    n = nv(G)
    if k ≥ n
        return densest_subgraph(G)
    end
    
    # Step 1: prune the graphs by iteratively remove the lowest degree vertices
    # Store the best snapshot density once the number of vertices is at most k in dprime
    H = copy(G)
    Δ = maximum(degree(H))
    B = Dict(d => Set(v for v in vertices(H) if degree(H, v) == d) for d in 0:Δ)
    d = 0
    dprime = -Inf
    remaining = n

    while remaining > 0
        if remaining ≤ k
            dprime = max(dprime, ne(H) / remaining)
        end

        # Find the next minimum degree d 
        d > Δ && break
        while isempty(B[d])
            d += 1
            d > Δ && break
        end
        d > Δ && break

        v = pop!(B[d]) # Remove a minimum degree vertex v
        for u in collect(neighbors(H, v))
            du = degree(H, u)
            rem_edge!(H, u, v)
            delete!(B[du], u) 
            push!(B[du - 1], u)
        end

        d = max(d-1, 0)
        remaining -= 1
    end

    # Step 2: Remove nodes with degree < dprime until no such node remains
    # Remove nodes will be isolated in the remaining graph. 
    H = copy(G)
    while nv(H) > 0
        removes = [v for v in vertices(H) if 0 < degree(H, v) < dprime]
        isempty(removes) && break
        for v in removes
            for u in collect(neighbors(H, v))
                rem_edge!(H, u, v)
            end
        end
    end

    # Step 3: Brute force search on the remaining graph H with at most k vertices
    best_S = Int[]
    best_density = 0.0
    H = [v for v in vertices(H) if degree(H, v) > 0]
    for size in 1:min(k, length(H))
        for S in combinations(H, size)
            dS = density(G, S)
            if dS > best_density
                best_density = dS
                best_S = S
            end
        end
    end

    return best_S, best_density
end

precompile(create_aux_graph, (SimpleGraph{Int}, Float64))
precompile(density, (SimpleGraph{Int}, Vector{Int}))
precompile(densest_subgraph, (SimpleGraph{Int}, Int, Symbol))
precompile(densest_at_most_k_subgraph, (SimpleGraph{Int}, Int))