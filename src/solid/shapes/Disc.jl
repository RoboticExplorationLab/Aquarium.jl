struct Disc{S} <: Shape
    radius::S
end

n_geometric_params(::Type{<:Disc}) = 1

valid_roles(::Disc) = (:center,)

n_boundary_segments(::Disc, n_nodes::Int) = n_nodes     # closed polygon wraps back

function local_attachment_point(disc::Disc{S}, role::Symbol) where {S}
    if role === :center
        return S[zero(S), zero(S)]
    else
        error("Disc has no attachment role $(role); supported: :center")
    end
end

function generate_boundary_nodes(disc::Disc{S}, n_nodes::Int) where {S}
    n_nodes >= 3 || error("generate_boundary_nodes: need n_nodes >= 3 for Disc, got $n_nodes")
    r = disc.radius
    θs = range(0, 2π, length=n_nodes + 1)[1:end-1]   # exclude wrap-around
    xs = [r * cos(θ) for θ in θs]
    ys = [r * sin(θ) for θ in θs]
    starts = collect(1:n_nodes)
    ends = vcat(collect(2:n_nodes), [1])     # last segment wraps back to node 1
    return xs, ys, starts, ends
end

@testitem "Disc" begin
    using Aquarium
    using LinearAlgebra

    @testset "construction" begin
        d = Disc(1.25)
        @test d.radius == 1.25
        @test n_geometric_params(Disc) == 1
        @test valid_roles(d) == (:center,)
    end

    @testset "attachment point" begin
        d = Disc(1.0)
        pt = local_attachment_point(d, :center)
        @test pt == [0.0, 0.0]
    end

    @testset "generate_boundary_nodes" begin
        d = Disc(0.5)
        xs, ys, starts, ends = generate_boundary_nodes(d, 8)
        @test length(xs) == 8
        @test length(ys) == 8

        for i in 1:8
            @test xs[i]^2 + ys[i]^2 ≈ 0.25 atol=1e-12
        end

        @test starts == collect(1:8)
        @test ends == [2, 3, 4, 5, 6, 7, 8, 1]

        @test n_boundary_segments(d, 8) == 8
    end

    @testset "FreeDisc construction" begin
        system = FreeDisc(0.01; radius=1.25, mass=4.0, moi=2.5, n_boundary_nodes=12,
                          ib_method=:original)
        @test system.n_bodies == 1
        @test system.bodies[1].shape.radius == 1.25
        @test system.bodies[1].mass == 4.0
        @test system.bodies[1].moi == 2.5
        @test system.topology.n_boundary_nodes == 12
        @test system.topology.n_boundary_segments == 12
    end

    @testset "Disc differentiable params round-trip" begin
        system = FreeDisc(0.01; radius=1.25, mass=4.0, moi=2.5, n_boundary_nodes=8,
                          ib_method=:original)
        p = collect_differentiable_params(system)
        system2 = inject_differentiable_params(system, p)
        @test system2.bodies[1].shape.radius == 1.25
        @test system2.bodies[1].mass == 4.0
        @test collect_differentiable_params(system2) ≈ p
    end
end
