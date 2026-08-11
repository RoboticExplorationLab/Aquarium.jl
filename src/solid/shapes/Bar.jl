struct Bar{S} <: Shape
    length::S
end

n_geometric_params(::Type{<:Bar}) = 1

valid_roles(::Bar) = (:tip, :root, :com)

n_boundary_segments(::Bar, n_nodes::Int) = n_nodes - 1   # open polyline

function local_attachment_point(bar::Bar{S}, role::Symbol) where {S}
    half_L = bar.length / 2
    if role === :tip
        return S[half_L, zero(S)]
    elseif role === :root
        return S[-half_L, zero(S)]
    elseif role === :com
        return S[zero(S), zero(S)]
    else
        error("Bar has no attachment role $(role); supported: :tip, :root, :com")
    end
end

function generate_boundary_nodes(bar::Bar{S}, n_nodes::Int) where {S}
    n_nodes >= 2 || error("generate_boundary_nodes: need n_nodes >= 2, got $n_nodes")
    half_L = bar.length / 2
    xs = collect(range(-half_L, half_L, length=n_nodes))
    ys = zeros(S, n_nodes)
    starts = collect(1:(n_nodes-1))
    ends = collect(2:n_nodes)
    return xs, ys, starts, ends
end

# `plot_shape!(ax, bar, body, origin_world, θ, plot_params)` lives in plot_solid_system.jl
# because it references the later-loaded `RigidBody` type.


@testitem "Bar" begin
    using AquariumClosed
    @testset "construction" begin
        bar = Bar(1.5)
        @test bar.length == 1.5
        @test n_geometric_params(Bar) == 1
    end

    @testset "local_attachment_point" begin
        bar = Bar(2.0)
        @test local_attachment_point(bar, :tip)  == [1.0, 0.0]
        @test local_attachment_point(bar, :root) == [-1.0, 0.0]
        @test local_attachment_point(bar, :com)  == [0.0, 0.0]
        @test_throws ErrorException local_attachment_point(bar, :nope)
    end

    @testset "generate_boundary_nodes" begin
        bar = Bar(2.0)
        n_nodes = 5
        xs, ys, starts, ends = generate_boundary_nodes(bar, n_nodes)

        @test length(xs) == n_nodes
        @test length(ys) == n_nodes
        # Endpoints should be at ±length/2 along x
        @test xs[1]  ≈ -1.0
        @test xs[end] ≈ 1.0
        # y-coordinates should be zero
        @test all(ys .≈ 0.0)
        # Nodes evenly spaced along x
        @test xs ≈ range(-1.0, 1.0, length=n_nodes)
        # Segments: n_nodes-1 total, each linking node i to i+1
        @test length(starts) == n_nodes - 1
        @test length(ends) == n_nodes - 1
        @test starts == collect(1:n_nodes-1)
        @test ends == collect(2:n_nodes)
    end
end
