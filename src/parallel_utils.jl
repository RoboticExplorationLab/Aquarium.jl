"""
    tcollect_sparse(f, range, ::Type{T}; n_chunks=Threads.nthreads()) where T

Chunked parallel sparse triplet collection. Partitions `range` into chunks,
spawns one task per chunk, and concatenates the resulting (I, J, V) vectors.

`f(chunk)` must return `(I_local::Vector{Int}, J_local::Vector{Int}, V_local::Vector{T})`.

Falls back to serial execution when `length(range) < 64` or `nthreads() == 1`.
"""
function tcollect_sparse(f, range, ::Type{T}; n_chunks::Int=Threads.nthreads()) where T
    n = length(range)

    # Serial fallback for small problems or single-threaded execution
    if n < 2 || Threads.nthreads() == 1 || n_chunks <= 1
        return f(range)
    end

    n_chunks = min(n_chunks, n)
    chunk_size = cld(n, n_chunks)
    chunks = Iterators.partition(range, chunk_size)

    tasks = map(chunks) do chunk
        Threads.@spawn f(chunk)
    end

    results = fetch.(tasks)

    I_vec = reduce(vcat, (r[1] for r in results))
    J_vec = reduce(vcat, (r[2] for r in results))
    V_vec = reduce(vcat, (r[3] for r in results))

    return I_vec, J_vec, V_vec
end

"""
    taccumulate(f, range; n_chunks=Threads.nthreads())

Chunked parallel dense accumulation with element-wise summation merge.

`f(chunk)` must return a `Tuple` of vectors. Results from all chunks are summed
element-wise.

Falls back to serial execution when `length(range) < 64` or `nthreads() == 1`.
"""
function taccumulate(f, range; n_chunks::Int=Threads.nthreads())
    n = length(range)

    # Serial fallback
    if n < 2 || Threads.nthreads() == 1 || n_chunks <= 1
        return f(range)
    end

    n_chunks = min(n_chunks, n)
    chunk_size = cld(n, n_chunks)
    chunks = Iterators.partition(range, chunk_size)

    tasks = map(chunks) do chunk
        Threads.@spawn f(chunk)
    end

    results = fetch.(tasks)

    # Sum element-wise across all chunk results
    output = results[1]
    for k in 2:length(results)
        r = results[k]
        output = ntuple(i -> output[i] .+ r[i], length(output))
    end

    return output
end

"""
    taccumulate_max(f, range; n_chunks=Threads.nthreads())

Like `taccumulate` but merges with element-wise `max` instead of summation.

`f(chunk)` must return a `Tuple` of vectors.
"""
function taccumulate_max(f, range; n_chunks::Int=Threads.nthreads())
    n = length(range)

    if n < 2 || Threads.nthreads() == 1 || n_chunks <= 1
        return f(range)
    end

    n_chunks = min(n_chunks, n)
    chunk_size = cld(n, n_chunks)
    chunks = Iterators.partition(range, chunk_size)

    tasks = map(chunks) do chunk
        Threads.@spawn f(chunk)
    end

    results = fetch.(tasks)

    output = results[1]
    for k in 2:length(results)
        r = results[k]
        output = ntuple(i -> max.(output[i], r[i]), length(output))
    end

    return output
end

"""
    tforeach(f, range; n_chunks=Threads.nthreads())

Chunked parallel for-each. Calls `f(item)` for each item in `range`, distributed
across spawned tasks. No return value.

Safe for in-place operations where each item writes to disjoint memory
(e.g., column-partitioned sparse matrix ops).

Falls back to serial execution when `length(range) < 64` or `nthreads() == 1`.
"""
function tforeach(f, range; n_chunks::Int=Threads.nthreads())
    n = length(range)

    if n < 2 || Threads.nthreads() == 1 || n_chunks <= 1
        for item in range
            f(item)
        end
        return nothing
    end

    n_chunks = min(n_chunks, n)
    chunk_size = cld(n, n_chunks)
    chunks = Iterators.partition(range, chunk_size)

    @sync for chunk in chunks
        Threads.@spawn for item in chunk
            f(item)
        end
    end

    return nothing
end
