# Generic system-level plot dispatch for the new PassiveSystem / ActuatedSystem types.
# Iterates the body list and delegates each body's drawing to its shape via plot_shape!.

function plot_shape!(
    ax::Axis,
    bar::Bar,
    body::RigidBody,
    body_origin_world::AbstractVector,
    θ::Real,
    plot_params::Dict{Symbol, Any};
    kwargs...,
)
    xs_local, ys_local, _, _ = generate_boundary_nodes(bar, body.n_boundary_nodes)

    # Apply com_offset shift (body frame = COM frame under Interpretation P)
    xs_local .+= body.com_offset[1]
    ys_local .+= body.com_offset[2]

    # Rotate and translate to world frame
    c, s = cos(θ), sin(θ)
    xs_world = similar(xs_local)
    ys_world = similar(ys_local)
    @inbounds for i in eachindex(xs_local)
        xs_world[i] = body_origin_world[1] + c * xs_local[i] - s * ys_local[i]
        ys_world[i] = body_origin_world[2] + s * xs_local[i] + c * ys_local[i]
    end

    lines!(ax, xs_world, ys_world;
        color = plot_params[:bodycolor],
        linewidth = plot_params[:linewidth],
    )

    if get(plot_params, :showboundarynodes, false)
        scatter!(ax, xs_world, ys_world;
            color = plot_params[:boundarynodecolor],
            markersize = plot_params[:boundarynodesize],
        )
    end

    return nothing
end

function plot_shape!(
    ax::Axis,
    disc::Disc,
    body::RigidBody,
    body_origin_world::AbstractVector,
    θ::Real,
    plot_params::Dict{Symbol, Any};
    kwargs...,
)
    # Filled circle at the body origin shifted by com_offset (rotated into world frame).
    c, s = cos(θ), sin(θ)
    cx = body_origin_world[1] + c * body.com_offset[1] - s * body.com_offset[2]
    cy = body_origin_world[2] + s * body.com_offset[1] + c * body.com_offset[2]
    circle = Circle(Point2f(cx, cy), Float32(disc.radius))
    poly!(ax, circle; color = plot_params[:bodycolor])

    if get(plot_params, :showorientation, false)
        # Small marker offset by radius/2 along the body's local x-axis.
        ox = cx + c * (disc.radius / 2) - s * 0.0
        oy = cy + s * (disc.radius / 2) + c * 0.0
        scatter!(ax, [ox], [oy];
            color = get(plot_params, :orientationcolor, :red),
            markersize = get(plot_params, :orientationsize, 5),
        )
    end

    if get(plot_params, :showboundarynodes, false)
        xs_local, ys_local, _, _ = generate_boundary_nodes(disc, body.n_boundary_nodes)
        xs_world = similar(xs_local)
        ys_world = similar(ys_local)
        @inbounds for i in eachindex(xs_local)
            xs_world[i] = body_origin_world[1] + c * (xs_local[i] + body.com_offset[1]) - s * (ys_local[i] + body.com_offset[2])
            ys_world[i] = body_origin_world[2] + s * (xs_local[i] + body.com_offset[1]) + c * (ys_local[i] + body.com_offset[2])
        end
        scatter!(ax, xs_world, ys_world;
            color = plot_params[:boundarynodecolor],
            markersize = plot_params[:boundarynodesize],
        )
    end

    return nothing
end

function plot_solid_system!(
    ax::Axis,
    system::SolidSystem,
    system_state::AbstractVector;
    kwargs...,
)
    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        cfg = body_configuration(system_state, i)
        body_origin_world = [cfg[1], cfg[2]]
        θ = cfg[3]
        plot_shape!(ax, body.shape, body, body_origin_world, θ, system.plot_params; kwargs...)
    end
    return nothing
end
