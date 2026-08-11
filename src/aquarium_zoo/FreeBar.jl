#############################################################################################
## FreeBar constructor function — a single free-floating bar-shaped rigid body.
## Returns a PassiveSystem with one RigidBody{Bar} and no joints.
#############################################################################################

function FreeBar(time_step::Real;
    bar_length::Real = 1.0,
    mass::Real = 1.0,
    moi::Real = 1/12,
    com_offset::AbstractVector = [0.0, 0.0],
    n_boundary_nodes::Int = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
)
    body = RigidBody(Bar(bar_length);
        mass = mass,
        moi = moi,
        com_offset = com_offset,
        n_boundary_nodes = n_boundary_nodes,
        ib_method = ib_method,
        discrete_delta_kind = discrete_delta_kind,
    )
    return PassiveSystem(time_step, [body], Joint[];
        gravity = gravity,
        plot_params = plot_params,
    )
end
