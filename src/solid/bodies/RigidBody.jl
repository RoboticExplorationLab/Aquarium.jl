struct RigidBody{Sh <: Shape, S} <: AbstractRigidBody
    shape::Sh
    mass::S
    moi::S
    com_offset::Vector{S}
    n_boundary_nodes::Int
    ib_method::Symbol
    discrete_delta_kind::Symbol
end

function RigidBody(shape::Shape;
    mass::Real,
    moi::Real,
    com_offset::AbstractVector=[0.0, 0.0],
    n_boundary_nodes::Int=16,
    ib_method::Symbol=:weak_form,
    discrete_delta_kind::Symbol=:one_point,
)
    length(com_offset) == 2 || error("RigidBody.com_offset must be 2D, got length $(length(com_offset))")
    S = promote_type(typeof(mass), typeof(moi), eltype(com_offset), _shape_param_type(shape))
    Sh = typeof(_convert_shape(shape, S))
    return RigidBody{Sh, S}(
        _convert_shape(shape, S),
        convert(S, mass),
        convert(S, moi),
        convert(Vector{S}, com_offset),
        n_boundary_nodes,
        ib_method,
        discrete_delta_kind,
    )
end

_shape_param_type(::Bar{S}) where {S} = S
_convert_shape(shape::Bar, ::Type{S}) where {S} = Bar{S}(convert(S, shape.length))

_shape_param_type(::Disc{S}) where {S} = S
_convert_shape(shape::Disc, ::Type{S}) where {S} = Disc{S}(convert(S, shape.radius))


@testitem "RigidBody" begin
    using AquariumClosed
    body = RigidBody(Bar(1.0);
        mass=2.5, moi=0.3, com_offset=[0.1, -0.2],
        n_boundary_nodes=16, ib_method=:weak_form)

    @test body.shape isa Bar
    @test body.shape.length == 1.0
    @test body.mass == 2.5
    @test body.moi == 0.3
    @test body.com_offset == [0.1, -0.2]
    @test body.n_boundary_nodes == 16
    @test body.ib_method === :weak_form
    @test body isa AbstractRigidBody

    @testset "defaults" begin
        minimal = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
        @test minimal.com_offset == [0.0, 0.0]
        @test minimal.n_boundary_nodes == 16
        @test minimal.ib_method === :weak_form
        @test minimal.discrete_delta_kind === :one_point
    end

    @testset "discrete_delta_kind" begin
        body_1pt = RigidBody(Bar(1.0); mass=1.0, moi=0.1, discrete_delta_kind=:one_point)
        @test body_1pt.discrete_delta_kind === :one_point

        body_3pt = RigidBody(Bar(1.0); mass=1.0, moi=0.1, discrete_delta_kind=:three_point)
        @test body_3pt.discrete_delta_kind === :three_point
    end
end
