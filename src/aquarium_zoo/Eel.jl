#############################################################################################
## Eel constructor function (new composition-based architecture).
##
## Returns a PassiveSystem with `n_links` bar-shaped bodies connected in a free-floating
## chain by (n_links − 1) PinJoints. No world attachment.
#############################################################################################

function Eel(time_step::Real, n_links::Int;
    bar_lengths::AbstractVector = ones(n_links),
    masses::AbstractVector = ones(n_links),
    mois::AbstractVector = fill(1/12, n_links),
    com_offsets::AbstractVector = [[0.0, 0.0] for _ in 1:n_links],
    equilibrium_angles::AbstractVector = zeros(n_links - 1),
    stiffnesses::AbstractVector = zeros(n_links - 1),
    dampings::AbstractVector = zeros(n_links - 1),
    n_boundary_nodes_per_link::Union{Int, AbstractVector{Int}} = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
)
    n_links >= 1 || error("Eel: n_links must be >= 1")
    length(bar_lengths)        == n_links     || error("Eel: bar_lengths must have n_links entries")
    length(masses)             == n_links     || error("Eel: masses must have n_links entries")
    length(mois)               == n_links     || error("Eel: mois must have n_links entries")
    length(com_offsets)        == n_links     || error("Eel: com_offsets must have n_links entries")
    length(equilibrium_angles) == n_links - 1 || error("Eel: equilibrium_angles must have n_links - 1 entries")
    length(stiffnesses)        == n_links - 1 || error("Eel: stiffnesses must have n_links - 1 entries")
    length(dampings)           == n_links - 1 || error("Eel: dampings must have n_links - 1 entries")

    nbn_per_link = n_boundary_nodes_per_link isa Int ?
        fill(n_boundary_nodes_per_link, n_links) : n_boundary_nodes_per_link
    length(nbn_per_link) == n_links || error("Eel: n_boundary_nodes_per_link must have n_links entries")

    bodies = [
        RigidBody(Bar(bar_lengths[i]);
            mass = masses[i],
            moi = mois[i],
            com_offset = com_offsets[i],
            n_boundary_nodes = nbn_per_link[i],
            ib_method = ib_method,
            discrete_delta_kind = discrete_delta_kind,
        )
        for i in 1:n_links
    ]

    joints = Joint[
        PinJoint(i, :tip, i + 1, :root;
            equilibrium_angle = equilibrium_angles[i],
            stiffness = stiffnesses[i],
            damping = dampings[i],
        )
        for i in 1:(n_links - 1)
    ]

    return PassiveSystem(time_step, bodies, joints;
        gravity = gravity,
        plot_params = plot_params,
    )
end

# Convert minimal coordinates [x1, y1, θ1, θ2, ..., θ_n] to maximal coordinates
# [x1, y1, θ1, x2, y2, θ2, ..., xn, yn, θn]. The first three entries are the (x, y, θ)
# of body 1's center; subsequent entries are the absolute angles of bodies 2..n.
function eel_maximal_from_minimal(eel::PassiveSystem, q_min::AbstractVector, n_links::Int)
    length(q_min) == n_links + 2 ||
        error("eel_maximal_from_minimal: expected q_min of length $(n_links+2), got $(length(q_min))")

    x1, y1 = q_min[1], q_min[2]
    θ1 = q_min[3]

    maximal = Vector{eltype(q_min)}(undef, 3 * n_links)
    maximal[1], maximal[2], maximal[3] = x1, y1, θ1

    # Walk the chain forward using each PinJoint's :tip → :root attachment pair.
    for i in 2:n_links
        prev_body = eel.bodies[i - 1]
        curr_body = eel.bodies[i]
        joint = eel.joints[i - 1]
        joint isa PinJoint || error("eel_maximal_from_minimal: joint $(i-1) must be PinJoint")

        prev_center = [maximal[3 * (i - 2) + 1], maximal[3 * (i - 2) + 2]]
        θ_prev = maximal[3 * (i - 2) + 3]
        R_prev = rotation_2d(θ_prev)

        # World position of the prev body's :tip attachment.
        tip_local = prev_body.com_offset .+ local_attachment_point(prev_body.shape, joint.role_A)
        tip_world = prev_center .+ R_prev * tip_local

        # Current body's center: its :root attachment must coincide with tip_world.
        θ_curr = q_min[i + 2]
        R_curr = rotation_2d(θ_curr)
        root_local = curr_body.com_offset .+ local_attachment_point(curr_body.shape, joint.role_B)
        curr_center = tip_world .- R_curr * root_local

        maximal[3 * (i - 1) + 1] = curr_center[1]
        maximal[3 * (i - 1) + 2] = curr_center[2]
        maximal[3 * (i - 1) + 3] = θ_curr
    end

    return maximal
end
