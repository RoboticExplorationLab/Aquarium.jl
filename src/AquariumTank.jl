struct AquariumTank{S<:Real}

    # Core components
    fluid::Fluid{Float64, S}
    bluff_body::SolidSystem
    swimmer::SolidSystem

    # Time stepping
    time_step::Float64

    # Number of constraints for fluid-structure coupling
    n_bluff_body_no_slip_constraints::Int
    n_swimmer_no_slip_constraints::Int
    n_no_slip_constraints::Int

    # State indices - pointing to different parts of the aquarium state vector
    # State structure: [fluid_velocity, swimmer_body_state, fluid_duals, swimmer_duals, bluff_body_no_slip_dual, swimmer_no_slip_dual]
    state_indices::Vector{Int}

    # Indices for full fluid and swimmer states (including all primal and dual variables)
    fluid_state_indices::Vector{Int}
    swimmer_state_indices::Vector{Int}

    # Indices for individual components
    fluid_velocity_indices::Vector{Int}
    swimmer_body_state_indices::Vector{Int}
    swimmer_configuration_indices::Vector{Int}
    swimmer_velocity_indices::Vector{Int}
    fluid_dual_indices::Vector{Int}
    swimmer_dual_indices::Vector{Int}
    bluff_body_no_slip_dual_indices::Vector{Int}
    swimmer_no_slip_dual_indices::Vector{Int}
    dual_indices::Vector{Int}
    primal_indices::Vector{Int}

    # Number of aquarium states
    n_states::Int

    # Number of full fluid and swimmer states
    n_fluid_states::Int
    n_swimmer_states::Int

    # Number of individual components
    n_fluid_velocities::Int
    n_swimmer_body_states::Int
    n_fluid_constraints::Int
    n_swimmer_constraints::Int
    n_bluff_body_no_slip_duals::Int
    n_swimmer_no_slip_duals::Int
    n_duals::Int

    # Bluff body is treated separately from swimmer
    n_bluff_body_states::Int

end

function AquariumTank(
    fluid::Fluid,
    bluff_body::SolidSystem,
    swimmer::SolidSystem,
)

    # Validate time steps match (skip for NoSystem — it has no time_step of its own)
    time_step = fluid.time_step
    if !(bluff_body isa NoSystem) && fluid.time_step != bluff_body.time_step
        error("Time steps must match for fluid and bluff_body")
    end
    if !(swimmer isa NoSystem) && fluid.time_step != swimmer.time_step
        error("Time steps must match for fluid and swimmer")
    end

    # Validate gravity matches (skip for NoSystem)
    if !(bluff_body isa NoSystem) && fluid.gravity_constant != -bluff_body.gravity[2]
        error("Gravity vectors must match for fluid and bluff_body")
    end
    if !(swimmer isa NoSystem) && fluid.gravity_constant != -swimmer.gravity[2]
        error("Gravity vectors must match for fluid and swimmer")
    end

    # Calculate number of no-slip constraints
    n_bluff_body_no_slip_constraints = bluff_body.topology.n_no_slip_constraints
    n_swimmer_no_slip_constraints = swimmer.topology.n_no_slip_constraints
    n_no_slip_constraints = n_bluff_body_no_slip_constraints + n_swimmer_no_slip_constraints

    # Calculate number of states
    n_bluff_body_states = bluff_body.n_states
    n_bluff_body_no_slip_duals = n_bluff_body_no_slip_constraints
    n_swimmer_no_slip_duals = n_swimmer_no_slip_constraints

    # Get component counts from fluid and swimmer
    n_fluid_velocities = fluid.n_velocities
    n_swimmer_body_states = swimmer.n_body_states
    n_fluid_constraints = fluid.n_constraints
    n_swimmer_constraints = swimmer.n_constraints

    # Total fluid and swimmer states
    n_fluid_states = n_fluid_velocities + n_fluid_constraints
    n_swimmer_states = n_swimmer_body_states + n_swimmer_constraints

    # Aquarium state structure (KKT form):
    # [fluid_velocity, swimmer_body_state, fluid_duals, swimmer_duals, bluff_body_no_slip_dual, swimmer_no_slip_dual]
    # Note: bluff_body states are prescribed externally, not part of aquarium state

    n_states = (n_fluid_velocities + n_swimmer_body_states +
                n_fluid_constraints + n_swimmer_constraints +
                n_bluff_body_no_slip_duals + n_swimmer_no_slip_duals)

    # Calculate state indices for aquarium state vector
    state_indices = collect(1:n_states)

    # Primal variables
    fluid_velocity_indices = collect(1:n_fluid_velocities)

    swimmer_body_state_indices = collect(
        (n_fluid_velocities + 1):(n_fluid_velocities + n_swimmer_body_states)
    )

    # Swimmer configuration and velocity indices (within swimmer_body_state_indices)
    swimmer_configuration_indices = swimmer_body_state_indices[swimmer.configuration_indices]
    swimmer_velocity_indices = swimmer_body_state_indices[swimmer.velocity_indices]

    # Dual variables
    offset = n_fluid_velocities + n_swimmer_body_states

    fluid_dual_indices = collect(
        (offset + 1):(offset + n_fluid_constraints)
    )
    offset += n_fluid_constraints

    swimmer_dual_indices = collect(
        (offset + 1):(offset + n_swimmer_constraints)
    )
    offset += n_swimmer_constraints

    bluff_body_no_slip_dual_indices = collect(
        (offset + 1):(offset + n_bluff_body_no_slip_duals)
    )
    offset += n_bluff_body_no_slip_duals

    swimmer_no_slip_dual_indices = collect(
        (offset + 1):(offset + n_swimmer_no_slip_duals)
    )

    # Full fluid and swimmer state indices
    fluid_state_indices = vcat(fluid_velocity_indices, fluid_dual_indices)
    swimmer_state_indices = vcat(swimmer_body_state_indices, swimmer_dual_indices)

    # All dual indices
    dual_indices = vcat(fluid_dual_indices, swimmer_dual_indices,
                        bluff_body_no_slip_dual_indices, swimmer_no_slip_dual_indices)
    n_duals = length(dual_indices)

    # All primal indices
    primal_indices = vcat(fluid_velocity_indices, swimmer_body_state_indices)

    S = typeof(fluid.density)
    return AquariumTank{S}(
        fluid,
        bluff_body,
        swimmer,
        time_step,
        n_bluff_body_no_slip_constraints,
        n_swimmer_no_slip_constraints,
        n_no_slip_constraints,
        state_indices,
        fluid_state_indices,
        swimmer_state_indices,
        fluid_velocity_indices,
        swimmer_body_state_indices,
        swimmer_configuration_indices,
        swimmer_velocity_indices,
        fluid_dual_indices,
        swimmer_dual_indices,
        bluff_body_no_slip_dual_indices,
        swimmer_no_slip_dual_indices,
        dual_indices,
        primal_indices,
        n_states,
        n_fluid_states,
        n_swimmer_states,
        n_fluid_velocities,
        n_swimmer_body_states,
        n_fluid_constraints,
        n_swimmer_constraints,
        n_bluff_body_no_slip_duals,
        n_swimmer_no_slip_duals,
        n_duals,
        n_bluff_body_states,
    )

end

# Convenience constructors for different configurations

# Fluid only (no solid bodies)
function AquariumTank_only_fluid(fluid::Fluid)
    return AquariumTank(fluid, NoSystem(), NoSystem())
end

# Fluid + swimmer only (no bluff body)
function AquariumTank_only_swimmer(fluid::Fluid, swimmer::SolidSystem)
    return AquariumTank(fluid, NoSystem(), swimmer)
end

# Fluid + bluff body only (no swimmer)
function AquariumTank_only_bluff_body(fluid::Fluid, bluff_body::SolidSystem)
    return AquariumTank(fluid, bluff_body, NoSystem())
end

function _assert_compatible_system_topology(old_system, new_system, role::String)
    fields = (:n_bodies, :n_configurations, :n_velocities, :n_constraints,
              :n_body_states, :n_states)
    for f in fields
        if getproperty(new_system, f) != getproperty(old_system, f)
            error("rebuild_tank_with_$(role): topology mismatch on field `$f`: " *
                  "original $role has $(getproperty(old_system, f)), " *
                  "replacement has $(getproperty(new_system, f)). " *
                  "Replacement must preserve every topology dimension so the tank's cached " *
                  "index vectors remain valid.")
        end
    end
    return nothing
end

_assert_compatible_swimmer_topology(tank::AquariumTank, new_swimmer) =
    _assert_compatible_system_topology(tank.swimmer, new_swimmer, "swimmer")

_assert_compatible_bluff_body_topology(tank::AquariumTank, new_bluff_body) =
    _assert_compatible_system_topology(tank.bluff_body, new_bluff_body, "bluff_body")

#############################################################################################
## Rebuild helpers — produce a new AquariumTank with one solid-system field replaced.
##
## Used by gradient closures (e.g. `calculate_stage_objective_gradients`) that need to
## swap a freshly-constructed (possibly Dual-typed) swimmer or bluff body into a tank
## without re-running the full tank constructor. The cached dimensions and index vectors
## remain valid as long as the replacement system has compatible topology with the
## original — see the topology check in `_assert_compatible_swimmer_topology`.
#############################################################################################

function rebuild_tank_with_swimmer(tank::AquariumTank, new_swimmer)
    _assert_compatible_swimmer_topology(tank, new_swimmer)
    return _rebuild_tank(tank; swimmer = new_swimmer)
end

function rebuild_tank_with_bluff_body(tank::AquariumTank, new_bluff_body)
    _assert_compatible_bluff_body_topology(tank, new_bluff_body)
    return _rebuild_tank(tank; bluff_body = new_bluff_body)
end

function rebuild_tank_with_fluid(tank::AquariumTank, new_fluid::Fluid)
    return _rebuild_tank(tank; fluid = new_fluid)
end

#############################################################################################
## Tank-level collect / inject differentiable params.
##
## Parameter layout: `[fluid_params, bluff_body_params, swimmer_params]`. Composition
## delegates layout within each component to that component's own collect/inject.
#############################################################################################

function collect_differentiable_params(tank::AquariumTank)
    return vcat(
        collect_differentiable_params(tank.fluid),
        collect_differentiable_params(tank.bluff_body),
        collect_differentiable_params(tank.swimmer),
    )
end

function inject_differentiable_params(tank::AquariumTank, params_vec::AbstractVector)
    n_fluid_params = n_differentiable_params(tank.fluid)
    n_bluff_body_params = n_differentiable_params(tank.bluff_body)
    n_swimmer_params = n_differentiable_params(tank.swimmer)

    expected_total = n_fluid_params + n_bluff_body_params + n_swimmer_params
    length(params_vec) == expected_total ||
        error("inject_differentiable_params(tank, p): expected length $(expected_total), got $(length(params_vec))")

    idx = 1
    fluid_slice = @view params_vec[idx:(idx + n_fluid_params - 1)]
    idx += n_fluid_params
    bluff_body_slice = @view params_vec[idx:(idx + n_bluff_body_params - 1)]
    idx += n_bluff_body_params
    swimmer_slice = @view params_vec[idx:(idx + n_swimmer_params - 1)]

    new_fluid = inject_differentiable_params(tank.fluid, fluid_slice)
    new_bluff_body = inject_differentiable_params(tank.bluff_body, bluff_body_slice)
    new_swimmer = inject_differentiable_params(tank.swimmer, swimmer_slice)

    return _rebuild_tank(tank;
        fluid = new_fluid,
        bluff_body = new_bluff_body,
        swimmer = new_swimmer,
    )
end

function _rebuild_tank(tank::AquariumTank;
    fluid      = tank.fluid,
    bluff_body = tank.bluff_body,
    swimmer    = tank.swimmer,
)
    S = typeof(fluid.density)
    return AquariumTank{S}(
        fluid,
        bluff_body,
        swimmer,
        tank.time_step,
        tank.n_bluff_body_no_slip_constraints,
        tank.n_swimmer_no_slip_constraints,
        tank.n_no_slip_constraints,
        tank.state_indices,
        tank.fluid_state_indices,
        tank.swimmer_state_indices,
        tank.fluid_velocity_indices,
        tank.swimmer_body_state_indices,
        tank.swimmer_configuration_indices,
        tank.swimmer_velocity_indices,
        tank.fluid_dual_indices,
        tank.swimmer_dual_indices,
        tank.bluff_body_no_slip_dual_indices,
        tank.swimmer_no_slip_dual_indices,
        tank.dual_indices,
        tank.primal_indices,
        tank.n_states,
        tank.n_fluid_states,
        tank.n_swimmer_states,
        tank.n_fluid_velocities,
        tank.n_swimmer_body_states,
        tank.n_fluid_constraints,
        tank.n_swimmer_constraints,
        tank.n_bluff_body_no_slip_duals,
        tank.n_swimmer_no_slip_duals,
        tank.n_duals,
        tank.n_bluff_body_states,
    )
end

#############################################################################################
## Extract components from aquarium state
#############################################################################################

# Extract full states (primal + dual variables)
function extract_fluid_state(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.fluid_state_indices]
end

function extract_swimmer_state(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.swimmer_state_indices]
end

# Extract primal variables
function extract_fluid_velocity(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.fluid_velocity_indices]
end

function extract_swimmer_body_state(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.swimmer_body_state_indices]
end

# Extract dual variables
function extract_fluid_dual(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.fluid_dual_indices]
end

function extract_swimmer_dual(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.swimmer_dual_indices]
end

function extract_bluff_body_no_slip_dual(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.bluff_body_no_slip_dual_indices]
end

function extract_swimmer_no_slip_dual(tank::AquariumTank, aquarium_state::AbstractVector)
    return aquarium_state[tank.swimmer_no_slip_dual_indices]
end

#############################################################################################
## Calculate aquarium energy
#############################################################################################

function calculate_aquarium_energy(
    tank::AquariumTank,
    aquarium_state::AbstractVector,
    bluff_body_state::AbstractVector,
)

    # Extract states
    fluid_state = extract_fluid_state(tank, aquarium_state)

    # Calculate energy components
    fluid_energy = calculate_total_energy(fluid, fluid_state)

    total_energy = fluid_energy

    # Swimmer energy
    swimmer_state = extract_swimmer_state(tank, aquarium_state)
    swimmer_energy = calculate_total_energy(swimmer, swimmer_state)
    total_energy += swimmer_energy

    # Bluff body energy
    bluff_body_energy = calculate_total_energy(bluff_body, bluff_body_state)
    total_energy += bluff_body_energy

    return total_energy

end

#############################################################################################
## Calculate aquarium dynamics residual
#############################################################################################

function calculate_aquarium_dynamics_residual(
    tank::AquariumTank,
    aquarium_state_kp1::AbstractVector,
    aquarium_state_k::AbstractVector,
    bluff_body_state_kp1::AbstractVector,
    swimmer_control_k::AbstractVector=zeros(0);
    is_midpoint_state_bluff_body::Bool=false,
    recompute_bc_vector::Bool=false
)

    T = promote_type(
        eltype(aquarium_state_kp1),
        eltype(aquarium_state_k),
        eltype(bluff_body_state_kp1),
        eltype(swimmer_control_k),
        typeof(tank.fluid.density),
        _system_param_type(tank.swimmer),
        _system_param_type(tank.bluff_body),
    )

    # extract fluid, bluff body, and swimmer
    fluid = tank.fluid
    bluff_body = tank.bluff_body
    swimmer = tank.swimmer

    # Extract full fluid and swimmer states at k+1 and k
    fluid_state_kp1 = extract_fluid_state(tank, aquarium_state_kp1)
    fluid_state_k = extract_fluid_state(tank, aquarium_state_k)

    swimmer_state_kp1 = extract_swimmer_state(tank, aquarium_state_kp1)
    swimmer_state_k = extract_swimmer_state(tank, aquarium_state_k)

    # Extract dual variables for no-slip constraints at k+1
    bluff_body_no_slip_dual_kp1 = extract_bluff_body_no_slip_dual(tank, aquarium_state_kp1)
    swimmer_no_slip_dual_kp1 = extract_swimmer_no_slip_dual(tank, aquarium_state_kp1)

    # Extract fluid velocities and swimmer body states for constraint evaluation
    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]

    swimmer_body_state_kp1 = swimmer_state_kp1[swimmer.body_state_indices]

    swimmer_configuration_kp1 = swimmer_body_state_kp1[swimmer.configuration_indices]
    swimmer_velocity_kp1 = swimmer_body_state_kp1[swimmer.velocity_indices]

    # Extract bluff body configurations and velocities (externally prescribed)
    bluff_body_configuration_kp1 = bluff_body_state_kp1[bluff_body.configuration_indices]
    bluff_body_velocity_kp1 = bluff_body_state_kp1[bluff_body.velocity_indices]

    #########################################################################################
    # 1. Calculate fluid dynamics residual with immersed boundary forces
    #########################################################################################

    # Basic fluid dynamics (without IB forces)
    fluid_dynamics_residual = calculate_fluid_dynamics_residual(
        fluid,
        fluid_state_kp1,
        fluid_state_k;
        recompute_bc_vector=recompute_bc_vector
    )

    # Get boundary configuration and forces for bluff body (evaluated at kp1)
    no_slip_vjp_bluff_body_fluid_velocity, _ =
        calculate_no_slip_constraint_vjp(
            fluid,
            bluff_body,
            fluid_velocity_kp1,
            bluff_body_configuration_kp1,
            bluff_body_velocity_kp1,
            bluff_body_no_slip_dual_kp1;
            is_midpoint_state=is_midpoint_state_bluff_body,
        )

    # Get boundary configuration and forces for swimmer (evaluated at kp1)
    no_slip_vjp_swimmer_fluid_velocity,
    no_slip_vjp_swimmer_swimmer_velocity =
        calculate_no_slip_constraint_vjp(
            fluid,
            swimmer,
            fluid_velocity_kp1,
            swimmer_configuration_kp1,
            swimmer_velocity_kp1,
            swimmer_no_slip_dual_kp1;
            is_midpoint_state=false,
        )

    # Add IB forces to fluid momentum equations (ForwardDiff-compatible)
    fluid_dynamics_residual_with_ib = similar(fluid_dynamics_residual, T)
    fluid_dynamics_residual_with_ib .= fluid_dynamics_residual
    fluid_dynamics_residual_with_ib[fluid.velocity_indices] =
        fluid_dynamics_residual[fluid.velocity_indices] +
        no_slip_vjp_bluff_body_fluid_velocity/fluid.cell_area +
        no_slip_vjp_swimmer_fluid_velocity/fluid.cell_area

    #########################################################################################
    # 2. Calculate swimmer dynamics residual with fluid forces
    #########################################################################################

    # Calculate swimmer dynamics residual
    swimmer_dynamics_residual = calculate_solid_dynamics_residual(
        swimmer,
        swimmer_state_kp1,
        swimmer_state_k,
        swimmer_control_k,
    )

    # Add fluid forces to swimmer dynamics (ForwardDiff-compatible)
    swimmer_dynamics_residual_with_forces = similar(swimmer_dynamics_residual, T)
    swimmer_dynamics_residual_with_forces .= swimmer_dynamics_residual
    swimmer_dynamics_residual_with_forces[swimmer.body_state_indices] =
        swimmer_dynamics_residual[swimmer.body_state_indices] +
        no_slip_vjp_swimmer_swimmer_velocity

    #########################################################################################
    # 3. Calculate no-slip constraint residuals
    #########################################################################################

    bluff_body_no_slip_residual = calculate_no_slip_constraint_residual(
        fluid,
        bluff_body,
        fluid_velocity_kp1,
        bluff_body_configuration_kp1,
        bluff_body_velocity_kp1;
        is_midpoint_state=is_midpoint_state_bluff_body,
    )

    swimmer_no_slip_residual = calculate_no_slip_constraint_residual(
        fluid,
        swimmer,
        fluid_velocity_kp1,
        swimmer_configuration_kp1,
        swimmer_velocity_kp1;
        is_midpoint_state=false,
    )

    #########################################################################################
    # 4. Assemble full aquarium residual in KKT structure
    # 
    #   [fluid_velocity_res;
    #    swimmer_body_res;
    #    fluid_dual_res;
    #    swimmer_dual_res;
    #    bluff_no_slip_res;
    #    swimmer_no_slip_res]
    #
    #########################################################################################

    aquarium_residual = zeros(T, tank.n_states)

    # Assign fluid dynamics residual to appropriate indices
    aquarium_residual[tank.fluid_state_indices] = fluid_dynamics_residual_with_ib

    # Assign swimmer dynamics residual to appropriate indices
    aquarium_residual[tank.swimmer_state_indices] = swimmer_dynamics_residual_with_forces

    # Assign no-slip constraint residuals
    aquarium_residual[tank.bluff_body_no_slip_dual_indices] = bluff_body_no_slip_residual
    aquarium_residual[tank.swimmer_no_slip_dual_indices] = swimmer_no_slip_residual

    return aquarium_residual

end

function calculate_aquarium_dynamics_jacobian(
    tank::AquariumTank,
    aquarium_state_kp1::AbstractVector,
    aquarium_state_k::AbstractVector,
    bluff_body_state_kp1::AbstractVector,
    swimmer_control_k::AbstractVector=zeros(0);
    is_midpoint_state_bluff_body::Bool=false,
)

    T = promote_type(
        eltype(aquarium_state_kp1),
        eltype(aquarium_state_k),
        eltype(bluff_body_state_kp1),
        eltype(swimmer_control_k),
        typeof(tank.fluid.density),
        _system_param_type(tank.swimmer),
        _system_param_type(tank.bluff_body),
    )

    # extract fluid, bluff body, and swimmer
    fluid = tank.fluid
    bluff_body = tank.bluff_body
    swimmer = tank.swimmer

    # extract number of states
    n_aquarium_states = tank.n_states

    # Extract full fluid and swimmer states at k+1 and k
    fluid_state_kp1 = extract_fluid_state(tank, aquarium_state_kp1)
    fluid_state_k = extract_fluid_state(tank, aquarium_state_k)

    swimmer_state_kp1 = extract_swimmer_state(tank, aquarium_state_kp1)
    swimmer_state_k = extract_swimmer_state(tank, aquarium_state_k)

    # Extract fluid velocities for constraint evaluation
    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]

    # Extract swimmer body states for constraint evaluation
    swimmer_body_state_kp1 = swimmer_state_kp1[swimmer.body_state_indices]

    swimmer_configuration_kp1 = swimmer_body_state_kp1[swimmer.configuration_indices]
    swimmer_velocity_kp1 = swimmer_body_state_kp1[swimmer.velocity_indices]

    # Extract bluff body configurations and velocities
    bluff_body_configuration_kp1 = bluff_body_state_kp1[bluff_body.configuration_indices]
    bluff_body_velocity_kp1 = bluff_body_state_kp1[bluff_body.velocity_indices]

    # Extract no-slip duals
    bluff_body_no_slip_dual_kp1 = extract_bluff_body_no_slip_dual(tank, aquarium_state_kp1)
    swimmer_no_slip_dual_kp1 = extract_swimmer_no_slip_dual(tank, aquarium_state_kp1)

    # Individual dynamics Jacobians
    ∂fluid_dynamics_∂fluid_state_kp1,
    ∂fluid_dynamics_∂fluid_state_k,
    ∂fluid_dynamics_∂fluid_properties = calculate_fluid_dynamics_jacobian(
        fluid,
        fluid_state_kp1,
        fluid_state_k,
    )

    # Calculate swimmer dynamics Jacobian
    ∂swimmer_dynamics_∂swimmer_state_kp1,
    ∂swimmer_dynamics_∂swimmer_state_k,
    ∂swimmer_dynamics_∂swimmer_control_k,
    ∂swimmer_dynamics_∂swimmer_params = calculate_solid_dynamics_jacobian(
        swimmer,
        swimmer_state_kp1,
        swimmer_state_k,
        swimmer_control_k,
    )

    # No-slip constraint Jacobians
    ∂no_slip_bluff_body_∂fluid_velocity_kp1,
    ∂no_slip_∂bluff_body_state_kp1,
    ∂no_slip_∂bluff_body_params =
        calculate_no_slip_constraint_jacobian(
            fluid,
            bluff_body,
            fluid_velocity_kp1,
            bluff_body_configuration_kp1,
            bluff_body_velocity_kp1;
            is_midpoint_state=is_midpoint_state_bluff_body,
        )

    ∂no_slip_swimmer_∂fluid_velocity_kp1,
    ∂no_slip_∂swimmer_body_state_kp1,
    ∂no_slip_∂swimmer_params =
        calculate_no_slip_constraint_jacobian(
            fluid,
            swimmer,
            fluid_velocity_kp1,
            swimmer_configuration_kp1,
            swimmer_velocity_kp1;
            is_midpoint_state=false,
        )

    # No-slip VJP Jacobians (evaluated at kp1)
    ∂no_slip_bluff_body_vjp_fluid_velocity_∂no_slip_dual_bluff_body,
    _,
    ∂no_slip_bluff_body_vjp_fluid_velocity_∂bluff_body_params,
    _,
    ∂no_slip_bluff_body_vjp_fluid_velocity_∂bluff_body_state_kp1,
    _ =
        calculate_no_slip_constraint_vjp_jacobian(
            fluid,
            bluff_body,
            fluid_velocity_kp1,
            bluff_body_configuration_kp1,
            bluff_body_velocity_kp1,
            bluff_body_no_slip_dual_kp1;
            is_midpoint_state=is_midpoint_state_bluff_body,
        )

    ∂no_slip_swimmer_vjp_fluid_velocity_∂no_slip_dual_swimmer,
    ∂no_slip_swimmer_vjp_swimmer_velocity_∂no_slip_dual_swimmer,
    ∂no_slip_swimmer_vjp_fluid_velocity_∂swimmer_params,
    ∂no_slip_swimmer_vjp_swimmer_velocity_∂swimmer_params,
    ∂no_slip_swimmer_vjp_fluid_velocity_∂swimmer_body_state_kp1,
    ∂no_slip_swimmer_vjp_swimmer_body_state_∂swimmer_body_state_kp1 =
        calculate_no_slip_constraint_vjp_jacobian(
            fluid,
            swimmer,
            fluid_velocity_kp1,
            swimmer_configuration_kp1,
            swimmer_velocity_kp1,
            swimmer_no_slip_dual_kp1;
            is_midpoint_state=false,
        )

    # Dynamics Jacobian w.r.t. aquarium states at time k+1
    # State structure: [fluid_velocity | swimmer_body_state | fluid_duals | swimmer_duals | bluff_body_no_slip_dual | swimmer_no_slip_dual]
    # Use native sparse [A B; C D] concatenation (32x faster than custom assembly)

    n_fv = tank.n_fluid_velocities
    n_sb = tank.n_swimmer_body_states
    n_fd = tank.n_fluid_constraints
    n_sd = tank.n_swimmer_constraints
    n_bbd = tank.n_bluff_body_no_slip_duals
    n_snd = tank.n_swimmer_no_slip_duals

    # Extract fluid dynamics sub-blocks: ∂[fv_res; fd_res]/∂[fv; fd]
    ∂fv_res_∂fv_kp1 = ∂fluid_dynamics_∂fluid_state_kp1[1:n_fv, 1:n_fv]
    ∂fv_res_∂fd_kp1 = ∂fluid_dynamics_∂fluid_state_kp1[1:n_fv, (n_fv+1):end]
    ∂fd_res_∂fv_kp1 = ∂fluid_dynamics_∂fluid_state_kp1[(n_fv+1):end, 1:n_fv]
    ∂fd_res_∂fd_kp1 = ∂fluid_dynamics_∂fluid_state_kp1[(n_fv+1):end, (n_fv+1):end]

    # Extract swimmer dynamics sub-blocks: ∂[sb_res; sd_res]/∂[sb; sd]
    ∂sb_res_∂sb_kp1 = ∂swimmer_dynamics_∂swimmer_state_kp1[1:n_sb, 1:n_sb]
    ∂sb_res_∂sd_kp1 = ∂swimmer_dynamics_∂swimmer_state_kp1[1:n_sb, (n_sb+1):end]
    ∂sd_res_∂sb_kp1 = ∂swimmer_dynamics_∂swimmer_state_kp1[(n_sb+1):end, 1:n_sb]
    ∂sd_res_∂sd_kp1 = ∂swimmer_dynamics_∂swimmer_state_kp1[(n_sb+1):end, (n_sb+1):end]

    # (fv, sb) block: VJP Hessian cross-coupling from the no-slip constraint.
    # Included in the KKT for accurate IFT gradient propagation (duals are O(1),
    # so this second-order term is non-negligible).
    ∂fv_res_∂sb_kp1_vjp = ∂no_slip_swimmer_vjp_fluid_velocity_∂swimmer_body_state_kp1 ./ fluid.cell_area

    # (sb, sb) block: VJP Hessian cross-coupling from the no-slip constraint.
    # The no-slip force on the swimmer depends on body state through boundary node geometry.
    # Same reasoning as the (fv, sb) block: duals are O(1), so this is non-negligible.
    ∂sb_res_∂sb_kp1_vjp = ∂no_slip_swimmer_vjp_swimmer_body_state_∂swimmer_body_state_kp1

    # Assemble kp1 Jacobian using native sparse block concatenation
    # Rows: [fv_res | sb_res | fd_res | sd_res | bbd_res | snd_res]
    # Cols: [fv     | sb     | fd     | sd     | bbd     | snd    ]
    ∂aquarium_dynamics_∂aquarium_states_kp1 = [
        ∂fv_res_∂fv_kp1                            ∂fv_res_∂sb_kp1_vjp                    ∂fv_res_∂fd_kp1                            spzeros(T,n_fv,n_sd)   ∂no_slip_bluff_body_vjp_fluid_velocity_∂no_slip_dual_bluff_body./fluid.cell_area   ∂no_slip_swimmer_vjp_fluid_velocity_∂no_slip_dual_swimmer./fluid.cell_area;
        spzeros(T,n_sb,n_fv)                       ∂sb_res_∂sb_kp1 + ∂sb_res_∂sb_kp1_vjp  spzeros(T,n_sb,n_fd)                       ∂sb_res_∂sd_kp1        spzeros(T,n_sb,n_bbd)                                                              ∂no_slip_swimmer_vjp_swimmer_velocity_∂no_slip_dual_swimmer;
        ∂fd_res_∂fv_kp1                            spzeros(T,n_fd,n_sb)   ∂fd_res_∂fd_kp1                            spzeros(T,n_fd,n_sd)   spzeros(T,n_fd,n_bbd)                                                              spzeros(T,n_fd,n_snd);
        spzeros(T,n_sd,n_fv)                       ∂sd_res_∂sb_kp1        spzeros(T,n_sd,n_fd)                       ∂sd_res_∂sd_kp1        spzeros(T,n_sd,n_bbd)                                                              spzeros(T,n_sd,n_snd);
        ∂no_slip_bluff_body_∂fluid_velocity_kp1    spzeros(T,n_bbd,n_sb)  spzeros(T,n_bbd,n_fd)                      spzeros(T,n_bbd,n_sd)  spzeros(T,n_bbd,n_bbd)                                                             spzeros(T,n_bbd,n_snd);
        ∂no_slip_swimmer_∂fluid_velocity_kp1       ∂no_slip_∂swimmer_body_state_kp1  spzeros(T,n_snd,n_fd)          spzeros(T,n_snd,n_sd)  spzeros(T,n_snd,n_bbd)                                                              spzeros(T,n_snd,n_snd)
    ]

    # Dynamics Jacobian w.r.t. states at time k
    ∂fv_res_∂fv_k = ∂fluid_dynamics_∂fluid_state_k[1:n_fv, 1:n_fv]
    ∂fv_res_∂fd_k = ∂fluid_dynamics_∂fluid_state_k[1:n_fv, (n_fv+1):end]
    ∂fd_res_∂fv_k = ∂fluid_dynamics_∂fluid_state_k[(n_fv+1):end, 1:n_fv]
    ∂fd_res_∂fd_k = ∂fluid_dynamics_∂fluid_state_k[(n_fv+1):end, (n_fv+1):end]

    ∂sb_res_∂sb_k = ∂swimmer_dynamics_∂swimmer_state_k[1:n_sb, 1:n_sb]
    ∂sb_res_∂sd_k = ∂swimmer_dynamics_∂swimmer_state_k[1:n_sb, (n_sb+1):end]
    ∂sd_res_∂sb_k = ∂swimmer_dynamics_∂swimmer_state_k[(n_sb+1):end, 1:n_sb]
    ∂sd_res_∂sd_k = ∂swimmer_dynamics_∂swimmer_state_k[(n_sb+1):end, (n_sb+1):end]

    ∂aquarium_dynamics_∂aquarium_states_k = [
        ∂fv_res_∂fv_k          spzeros(T,n_fv,n_sb)   ∂fv_res_∂fd_k          spzeros(T,n_fv,n_sd)   spzeros(T,n_fv,n_bbd)   spzeros(T,n_fv,n_snd);
        spzeros(T,n_sb,n_fv)   ∂sb_res_∂sb_k          spzeros(T,n_sb,n_fd)   ∂sb_res_∂sd_k          spzeros(T,n_sb,n_bbd)   spzeros(T,n_sb,n_snd);
        ∂fd_res_∂fv_k          spzeros(T,n_fd,n_sb)   ∂fd_res_∂fd_k          spzeros(T,n_fd,n_sd)   spzeros(T,n_fd,n_bbd)   spzeros(T,n_fd,n_snd);
        spzeros(T,n_sd,n_fv)   ∂sd_res_∂sb_k          spzeros(T,n_sd,n_fd)   ∂sd_res_∂sd_k          spzeros(T,n_sd,n_bbd)   spzeros(T,n_sd,n_snd);
        spzeros(T,n_bbd,n_fv)  spzeros(T,n_bbd,n_sb)  spzeros(T,n_bbd,n_fd)  spzeros(T,n_bbd,n_sd)  spzeros(T,n_bbd,n_bbd)  spzeros(T,n_bbd,n_snd);
        spzeros(T,n_snd,n_fv)  spzeros(T,n_snd,n_sb)  spzeros(T,n_snd,n_fd)  spzeros(T,n_snd,n_sd)  spzeros(T,n_snd,n_bbd)  spzeros(T,n_snd,n_snd)
    ]

    # Dynamics Jacobian w.r.t. fluid properties
    n_fluid_props = 4
    ∂fv_res_∂fluid_props = ∂fluid_dynamics_∂fluid_properties[1:n_fv, :]
    ∂fd_res_∂fluid_props = ∂fluid_dynamics_∂fluid_properties[(n_fv+1):end, :]
    ∂aquarium_dynamics_∂fluid_properties = [
        ∂fv_res_∂fluid_props;
        spzeros(T,n_sb,n_fluid_props);
        ∂fd_res_∂fluid_props;
        spzeros(T,n_sd,n_fluid_props);
        spzeros(T,n_bbd,n_fluid_props);
        spzeros(T,n_snd,n_fluid_props)
    ]

    # Dynamics Jacobian w.r.t. bluff body parameters
    n_bluff_body_params = length(collect_differentiable_params(bluff_body))
    ∂aquarium_dynamics_∂bluff_body_params = [
        ∂no_slip_bluff_body_vjp_fluid_velocity_∂bluff_body_params./fluid.cell_area;
        spzeros(T,n_sb,n_bluff_body_params);
        spzeros(T,n_fd,n_bluff_body_params);
        spzeros(T,n_sd,n_bluff_body_params);
        ∂no_slip_∂bluff_body_params;
        spzeros(T,n_snd,n_bluff_body_params)
    ]

    # Dynamics Jacobian w.r.t. bluff-body state
    ∂aquarium_dynamics_∂bluff_body_state_kp1 = [
        ∂no_slip_bluff_body_vjp_fluid_velocity_∂bluff_body_state_kp1./fluid.cell_area;
        spzeros(T,n_sb,bluff_body.n_body_states);
        spzeros(T,n_fd,bluff_body.n_body_states);
        spzeros(T,n_sd,bluff_body.n_body_states);
        ∂no_slip_∂bluff_body_state_kp1;
        spzeros(T,n_snd,bluff_body.n_body_states)
    ]

    # Dynamics Jacobian w.r.t. swimmer parameters
    n_swimmer_params = length(collect_differentiable_params(swimmer))
    ∂sb_res_∂swimmer_params = ∂swimmer_dynamics_∂swimmer_params[1:n_sb, :]
    ∂sd_res_∂swimmer_params = ∂swimmer_dynamics_∂swimmer_params[(n_sb+1):end, :]
    ∂aquarium_dynamics_∂swimmer_params = [
        ∂no_slip_swimmer_vjp_fluid_velocity_∂swimmer_params./fluid.cell_area;
        ∂sb_res_∂swimmer_params + ∂no_slip_swimmer_vjp_swimmer_velocity_∂swimmer_params;
        spzeros(T,n_fd,n_swimmer_params);
        ∂sd_res_∂swimmer_params;
        spzeros(T,n_bbd,n_swimmer_params);
        ∂no_slip_∂swimmer_params
    ]

    # Dynamics Jacobian w.r.t. swimmer control inputs
    n_swimmer_control = length(swimmer_control_k)
    ∂sb_res_∂control_k = ∂swimmer_dynamics_∂swimmer_control_k[1:n_sb, :]
    ∂sd_res_∂control_k = ∂swimmer_dynamics_∂swimmer_control_k[(n_sb+1):end, :]
    ∂aquarium_dynamics_∂swimmer_control_k = [
        spzeros(T,n_fv,n_swimmer_control);
        ∂sb_res_∂control_k;
        spzeros(T,n_fd,n_swimmer_control);
        ∂sd_res_∂control_k;
        spzeros(T,n_bbd,n_swimmer_control);
        spzeros(T,n_snd,n_swimmer_control)
    ]

    return ∂aquarium_dynamics_∂aquarium_states_kp1,
        ∂aquarium_dynamics_∂aquarium_states_k,
        ∂aquarium_dynamics_∂swimmer_control_k,
        ∂aquarium_dynamics_∂fluid_properties,
        ∂aquarium_dynamics_∂swimmer_params,
        ∂aquarium_dynamics_∂bluff_body_params,
        ∂aquarium_dynamics_∂bluff_body_state_kp1

end

@testitem "calculate_aquarium_dynamics_jacobian vs ForwardDiff" begin
    using Aquarium
    using ForwardDiff
    using FiniteDiff
    using LinearAlgebra
    using Random

    Random.seed!(42)

    time_step = 0.01
    gravity_constant = 98.0

    fluid = Fluid(time_step;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.01, 0.0],
        grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:freestream,
        gravity_constant=gravity_constant,
    )

    disc_radius = 0.05
    disc_mass = 1.0 * π * disc_radius^2
    disc_moi = 0.5 * disc_mass * disc_radius^2
    bluff_body = FreeDisc(time_step;
        radius=disc_radius, mass=disc_mass, moi=disc_moi,
        n_boundary_nodes=6, ib_method=:weak_form,
        gravity=[0.0, -gravity_constant],
    )

    n_links = 3
    # Reverse-engineer raw gains for effective Kp=Kd=100 (matching old ServoMotorPD defaults)
    Kp_raw = 100.0 / ((4096 / (2π)) * (9.3e6 / 885) / 128)
    Kd_raw = 100.0 / (0.001 * (4096 / (2π)) * (9.3e6 / 885) / 16)
    swimmer = RExEel(time_step, n_links;
        bar_lengths=fill(0.1, n_links),
        masses=fill(2.0, n_links),
        mois=(1/12) .* fill(2.0, n_links) .* (fill(0.1, n_links) .^ 2),
        Kps=fill(Kp_raw, n_links - 1),
        Kds=fill(Kd_raw, n_links - 1),
        max_torques=fill(Inf, n_links - 1),
        n_boundary_nodes_per_link=3,
        gravity=[0.0, -gravity_constant],
        actuation_mode=:pd,
    )

    tank = AquariumTank(fluid, bluff_body, swimmer)

    # Non-trivial random aquarium states and bluff-body state (off-center
    # so boundary nodes are not on grid lines; small magnitudes so the
    # Newton basin is benign).
    aquarium_state_k = 0.01 .* randn(tank.n_states)
    aquarium_state_kp1 = aquarium_state_k .+ 0.005 .* randn(tank.n_states)
    bluff_body_state_kp1 = [0.47, 0.53, 0.31, 0.02, -0.015, 0.03]
    swimmer_control_k = deg2rad.(10 .* randn(swimmer.n_control_inputs))

    (
        ∂D_∂xkp1,
        ∂D_∂xk,
        ∂D_∂uk,
        ∂D_∂fluid_props,
        ∂D_∂sw_params,
        ∂D_∂bb_params,
        ∂D_∂bb_state_kp1,
    ) = calculate_aquarium_dynamics_jacobian(
        tank, aquarium_state_kp1, aquarium_state_k,
        bluff_body_state_kp1, swimmer_control_k,
    )

    @test size(∂D_∂xkp1) == (tank.n_states, tank.n_states)
    @test size(∂D_∂xk) == (tank.n_states, tank.n_states)
    @test size(∂D_∂uk) == (tank.n_states, length(swimmer_control_k))
    @test size(∂D_∂fluid_props) == (tank.n_states, length(collect_differentiable_params(fluid)))
    @test size(∂D_∂sw_params) == (tank.n_states, length(collect_differentiable_params(swimmer)))
    @test size(∂D_∂bb_params) == (tank.n_states, length(collect_differentiable_params(bluff_body)))
    @test size(∂D_∂bb_state_kp1) == (tank.n_states, bluff_body.n_body_states)

    # --- State/control/bb_state Jacobians: ForwardDiff (exact AD) ---

    # ∂D/∂x_kp1 — the full KKT matrix
    ad_∂D_∂xkp1 = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(
            tank, x, aquarium_state_k, bluff_body_state_kp1, swimmer_control_k),
        aquarium_state_kp1)
    @test Matrix(∂D_∂xkp1) ≈ ad_∂D_∂xkp1 rtol=1e-3

    # ∂D/∂x_k
    ad_∂D_∂xk = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(
            tank, aquarium_state_kp1, x, bluff_body_state_kp1, swimmer_control_k),
        aquarium_state_k)
    @test Matrix(∂D_∂xk) ≈ ad_∂D_∂xk atol=1e-10

    # ∂D/∂u_k
    ad_∂D_∂uk = ForwardDiff.jacobian(
        u -> calculate_aquarium_dynamics_residual(
            tank, aquarium_state_kp1, aquarium_state_k, bluff_body_state_kp1, u),
        swimmer_control_k)
    @test Matrix(∂D_∂uk) ≈ ad_∂D_∂uk atol=1e-10

    # ∂D/∂fluid_props via inject_differentiable_params
    fd_∂D_∂fluid_props = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(fluid)) do p
        new_fluid = inject_differentiable_params(fluid, p)
        new_tank = rebuild_tank_with_fluid(tank, new_fluid)
        calculate_aquarium_dynamics_residual(
            new_tank, aquarium_state_kp1, aquarium_state_k,
            bluff_body_state_kp1, swimmer_control_k)
    end
    @test Matrix(∂D_∂fluid_props) ≈ fd_∂D_∂fluid_props rtol=1e-4

    # ∂D/∂swimmer_params via inject_differentiable_params
    fd_∂D_∂sw_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        new_tank = rebuild_tank_with_swimmer(tank, new_sw)
        calculate_aquarium_dynamics_residual(
            new_tank, aquarium_state_kp1, aquarium_state_k,
            bluff_body_state_kp1, swimmer_control_k)
    end
    @test Matrix(∂D_∂sw_params) ≈ fd_∂D_∂sw_params rtol=1e-4

    # ∂D/∂bluff_body_params via inject_differentiable_params
    fd_∂D_∂bb_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(bluff_body)) do p
        new_bb = inject_differentiable_params(bluff_body, p)
        new_tank = rebuild_tank_with_bluff_body(tank, new_bb)
        calculate_aquarium_dynamics_residual(
            new_tank, aquarium_state_kp1, aquarium_state_k,
            bluff_body_state_kp1, swimmer_control_k)
    end
    @test Matrix(∂D_∂bb_params) ≈ fd_∂D_∂bb_params rtol=1e-4

    # ∂D/∂bluff_body_state_kp1
    ad_∂D_∂bb_state = ForwardDiff.jacobian(
        bb -> calculate_aquarium_dynamics_residual(
            tank, aquarium_state_kp1, aquarium_state_k, bb, swimmer_control_k),
        bluff_body_state_kp1)
    @test Matrix(∂D_∂bb_state_kp1) ≈ ad_∂D_∂bb_state atol=1e-10
end

#############################################################################################
## Initialize aquarium state
#############################################################################################

function initialize_aquarium_state(
    tank::AquariumTank,
    fluid_initial_velocity::AbstractVector,
    swimmer_initial_body_state::AbstractVector=zeros(tank.swimmer.n_body_states)
)

    T = promote_type(
        eltype(fluid_initial_velocity),
        eltype(swimmer_initial_body_state),
    )

    swimmer=tank.swimmer
    
    aquarium_state = zeros(T, tank.n_states)
    
    # Set fluid initial state
    aquarium_state[tank.fluid_state_indices] .= initialize_fluid_state(
        tank.fluid,
        fluid_initial_velocity
    )
    
    # Set swimmer initial state
    if (!isempty(swimmer_initial_body_state))

        aquarium_state[tank.swimmer_state_indices] .= initialize_solid_state(
            swimmer,
            swimmer_initial_body_state
        )

    end
    
    # No-slip duals initialized to zero
    aquarium_state[tank.bluff_body_no_slip_dual_indices] .= zeros(T, tank.n_bluff_body_no_slip_duals)
    aquarium_state[tank.swimmer_no_slip_dual_indices] .= zeros(T, tank.n_swimmer_no_slip_duals)
    
    return aquarium_state

end

#############################################################################################
## Simulate aquarium dynamics using implicit time integration
#############################################################################################

function simulate_aquarium(
    tank::AquariumTank,
    aquarium_state_0::AbstractVector,
    final_time::Real,
    bluff_body_state_params::AbstractVector=zeros(0),
    swimmer_control_params::AbstractVector=zeros(0);
    is_midpoint_bluff_body::Bool=false,
    pivot_type::Symbol=:rcm,
    scaling_type::Symbol=:ruiz,
    solver_type::Symbol=:gmres,
    preconditioner_type::Symbol=:ilu,
    lazy::Bool=false,
    n_pardiso_threads::Int=Sys.CPU_THREADS,
    max_newton_iterations::Int=10,
    newton_tolerance::Float64=1e-6,
    ilu_drop_tolerance::Float64=1e-2,
    amg_smoother_type::Symbol=:forward_gs,
    gmres_tolerance::Float64=newton_tolerance * 1e-2,
    gmres_memory::Int=50,
    gmres_max_iterations::Int=500,
    dual_regularization=1e-6,
    primal_regularization=0.0,
    gradient_dual_regularization=1e-6,
    verbose::Bool=false,
    # Objective calculation options
    calculate_objective::Bool=false,
    gradient_method::Symbol=:forward,
    # Individual gradient flags (default to calculate_objective)
    calculate_gradient_wrt_fluid_properties::Bool=calculate_objective,
    calculate_gradient_wrt_swimmer_params::Bool=calculate_objective,
    calculate_gradient_wrt_bluff_body_params::Bool=calculate_objective,
    calculate_gradient_wrt_control_params::Bool=calculate_objective,
    calculate_gradient_wrt_bluff_body_state_params::Bool=calculate_objective,
    # Dynamics Jacobian flag
    compute_swimmer_dynamics_jacobian::Bool=false,
    # Bluff body state trajectory function (takes time and params, returns state)
    calculate_bluff_body_state_from_params::Function = (bluff_body, t, bluff_body_state_params; bluff_body_params=collect_differentiable_params(bluff_body)) -> bluff_body_state_params,
    calculate_bluff_body_state_from_params_jacobian::Function = (bluff_body, t, bluff_body_state_params; bluff_body_params=collect_differentiable_params(bluff_body)) ->
        (ForwardDiff.jacobian(_p -> calculate_bluff_body_state_from_params(bluff_body, t, _p), bluff_body_state_params),
        ForwardDiff.jacobian(_p -> calculate_bluff_body_state_from_params(bluff_body, t, bluff_body_state_params; bluff_body_params=_p), bluff_body_params)),
    # Control input functions
    calculate_control_input_from_params::Function = (swimmer, t, control_params) -> control_params,
    calculate_control_input_from_params_jacobian::Function = (swimmer, t, control_params) ->
        ForwardDiff.jacobian(_u -> calculate_control_input_from_params(swimmer, t, _u), control_params),
    # Objective functions
    calculate_stage_objective::Function = (tank, time, aquarium_state, bluff_body_state, swimmer_control) -> 0.0,
    calculate_terminal_objective::Function = (tank, time, aquarium_state, bluff_body_state) -> 0.0,
    # Gradients of objectives w.r.t. states and parameters.
    # The `rebuild_swimmer` / `rebuild_bluff_body` kwargs let callers opt into a
    # BeyondReachDOJO-style direct-construction pattern for gradient evaluation.
    # Defaults rebuild the relevant system via `inject_differentiable_params` and
    # swap it into a fresh tank via `rebuild_tank_with_swimmer` / `_with_bluff_body`,
    # then run the user's objective on the rebuilt tank. User-supplied objective
    # functions must NOT accept `swimmer_params` / `bluff_body_params` kwargs — read
    # parameters directly off the `tank.swimmer.*` / `tank.bluff_body.*` struct tree.
    calculate_stage_objective_gradients::Function = (tank, time, aquarium_state, bluff_body_state, swimmer_control;
        tank_params::AbstractVector = collect_differentiable_params(tank),
        rebuild_tank::Function = p -> inject_differentiable_params(tank, p),
    ) -> begin
        # Single tank-level AD pass over the combined [fluid; bluff_body; swimmer] params
        ∂tank = ForwardDiff.gradient(tank_params) do p
            new_tank = rebuild_tank(p)
            calculate_stage_objective(new_tank, time, aquarium_state, bluff_body_state, swimmer_control)
        end

        n_fp = n_differentiable_params(tank.fluid)
        n_bp = n_differentiable_params(tank.bluff_body)
        ∂fluid_params     = ∂tank[1:n_fp]
        ∂bluff_body_params = ∂tank[(n_fp + 1):(n_fp + n_bp)]
        ∂swimmer_params    = ∂tank[(n_fp + n_bp + 1):end]

        return (
            ForwardDiff.gradient(x -> calculate_stage_objective(tank, time, x, bluff_body_state, swimmer_control), aquarium_state),
            ForwardDiff.gradient(bb -> calculate_stage_objective(tank, time, aquarium_state, bb, swimmer_control), bluff_body_state),
            ForwardDiff.gradient(u -> calculate_stage_objective(tank, time, aquarium_state, bluff_body_state, u), swimmer_control),
            ∂fluid_params,
            ∂swimmer_params,
            ∂bluff_body_params,
        )
    end,
    calculate_terminal_objective_gradients::Function = (tank, time, aquarium_state, bluff_body_state;
        tank_params::AbstractVector = collect_differentiable_params(tank),
        rebuild_tank::Function = p -> inject_differentiable_params(tank, p),
    ) -> begin
        ∂tank = ForwardDiff.gradient(tank_params) do p
            new_tank = rebuild_tank(p)
            calculate_terminal_objective(new_tank, time, aquarium_state, bluff_body_state)
        end

        n_fp = n_differentiable_params(tank.fluid)
        n_bp = n_differentiable_params(tank.bluff_body)
        ∂fluid_params     = ∂tank[1:n_fp]
        ∂bluff_body_params = ∂tank[(n_fp + 1):(n_fp + n_bp)]
        ∂swimmer_params    = ∂tank[(n_fp + n_bp + 1):end]

        return (
            ForwardDiff.gradient(x -> calculate_terminal_objective(tank, time, x, bluff_body_state), aquarium_state),
            ForwardDiff.gradient(bb -> calculate_terminal_objective(tank, time, aquarium_state, bb), bluff_body_state),
            ∂fluid_params,
            ∂swimmer_params,
            ∂bluff_body_params,
        )
    end,
    # Initial state Jacobians w.r.t. parameters
    initial_aquarium_state_fluid_properties_jacobian::AbstractArray = zeros(length(aquarium_state_0), 4),
    initial_aquarium_state_swimmer_params_jacobian::AbstractArray = zeros(length(aquarium_state_0), length(collect_differentiable_params(tank.swimmer))),
    initial_aquarium_state_bluff_body_params_jacobian::AbstractArray = zeros(length(aquarium_state_0), length(collect_differentiable_params(tank.bluff_body))),
)

    # Validate solver and preconditioner types
    valid_solver_types = (:pardiso, :mumps, :gmres, :backslash)
    valid_preconditioner_types = (:none, :ilu, :ilu0, :pardiso, :amg,
                                   :approx_schur_ilu,
                                   :approx_schur_ilu0,
                                   :approx_schur_partial_amg,
                                   :approx_schur_full_amg)

    if !(solver_type in valid_solver_types)
        solver_type = :gmres
        @warn "Invalid solver_type specified. Using default GMRES instead."
    end

    if solver_type in (:pardiso, :mumps, :backslash)
        preconditioner_type = :none
        lazy = false
        @warn "No preconditioning or lazy with Pardiso, MUMPS or backslash. Setting preconditioner_type to :none."
    elseif !(preconditioner_type in valid_preconditioner_types)
        preconditioner_type = :ilu
        @warn "Invalid preconditioner_type specified. Using default ILU instead."
    end

    # if performing schur complement preconditioning, do not pivot
    if preconditioner_type in (:approx_schur_ilu,
                               :approx_schur_ilu0,
                               :approx_schur_partial_amg,
                               :approx_schur_full_amg)
        pivot_type = :none
        @warn "No pivoting with Schur complement preconditioning. Setting pivot_type to :none."
    end

    # Extract subsystems
    fluid = tank.fluid
    swimmer = tank.swimmer
    bluff_body = tank.bluff_body

    # Determine knot points
    time_step = tank.time_step
    N = Int(final_time/time_step + 1)

    # Extract number of states
    n_states = tank.n_states

    # Initialize trajectories
    time_traj = Vector(LinRange(0, final_time, N))
    aquarium_state_traj = [copy(aquarium_state_0) for k = 1:N]

    # Generate bluff body state trajectory from params
    bluff_body_state_traj = [calculate_bluff_body_state_from_params(bluff_body, t, bluff_body_state_params) for t in time_traj]

    # Generate control trajectory from control params
    swimmer_control_trajectory = [calculate_control_input_from_params(swimmer, t, swimmer_control_params) for t = time_traj[2:end]]

    # Initialize solution vector for solver
    solution_vector = rand(n_states)

    # Create a sample Jacobian to initialize solver (using random states)
    bluff_body_state_sample = length(bluff_body_state_traj[1]) > 0 ? bluff_body_state_traj[1] : zeros(0)
    swimmer_control_sample = length(swimmer_control_trajectory) > 0 && length(swimmer_control_trajectory[1]) > 0 ? swimmer_control_trajectory[1] : zeros(0)

    kkt_rand, _, _, _, _, _, _ = calculate_aquarium_dynamics_jacobian(
        tank,
        rand(n_states),
        rand(n_states),
        bluff_body_state_sample,
        swimmer_control_sample,
        is_midpoint_state_bluff_body=is_midpoint_bluff_body
    )

    # Create solver
    solver = create_solver(kkt_rand, solution_vector, solver_type;
        n_pardiso_threads=n_pardiso_threads,
        gmres_memory=gmres_memory
    )

    # For GMRES with Pardiso preconditioner, create a separate Pardiso solver
    preconditioner = nothing
    preconditioner_solver = nothing

    if solver_type == :gmres && preconditioner_type == :pardiso
        if !PARDISO_LOADED[]
            error("Pardiso is not available. Please use a different preconditioner_type or install Pardiso on Linux/Windows.")
        end
        preconditioner_solver = create_solver(kkt_rand, solution_vector, :pardiso;
            n_pardiso_threads=n_pardiso_threads
        )
        lazy = true # Force lazy mode when using Pardiso preconditioner
    end

    # Initialize dynamics Jacobian storage if needed
    if compute_swimmer_dynamics_jacobian

        n_control_inputs = isempty(swimmer_control_params) ? 0 : length(swimmer_control_trajectory[1])
        n_swimmer_body_states = swimmer.n_body_states

        dynamics_jacobian_wrt_state_traj = [zeros(n_swimmer_body_states, n_swimmer_body_states) for k = 1:N]
        dynamics_jacobian_wrt_control_traj = [zeros(n_swimmer_body_states, n_control_inputs) for k = 1:N]

        # Initialize first timestep
        # A_1 = I (initial state is given)
        dynamics_jacobian_wrt_state_traj[1] = zeros(n_swimmer_body_states, n_swimmer_body_states)
        for i = 1:n_swimmer_body_states
            dynamics_jacobian_wrt_state_traj[1][i, i] = 1.0
        end

        # B_1 = 0 (initial swimmer state doesn't depend on u_1)
        dynamics_jacobian_wrt_control_traj[1] = zeros(n_swimmer_body_states, n_control_inputs)

        state_dynamics_jacobian_kp1 = zeros(n_swimmer_body_states, n_swimmer_body_states)
        control_dynamics_jacobian_kp1 = zeros(n_swimmer_body_states, n_control_inputs)

    end

    # Initialize objective gradients if needed
    if calculate_objective

        # Initialize objective trajectory and cumulated value
        objective_trajectory = zeros(N)
        objective_value = 0.0

        # Initialize objective gradients trajectories
        objective_gradient_wrt_fluid_properties_traj = [zeros(4) for k = 1:N]
        objective_gradient_wrt_swimmer_params_traj = [zeros(length(collect_differentiable_params(swimmer))) for k = 1:N]
        objective_gradient_wrt_bluff_body_params_traj = [zeros(length(collect_differentiable_params(bluff_body))) for k = 1:N]
        objective_gradient_wrt_control_params_traj = [zeros(length(swimmer_control_params)) for k = 1:N]
        objective_gradient_wrt_bluff_body_state_params_traj = [zeros(length(bluff_body_state_params)) for k = 1:N]

        # Compute control and bluff body trajectory Jacobians
        ∂control_trajectory_∂control_params = [calculate_control_input_from_params_jacobian(swimmer, t, swimmer_control_params) for t = time_traj[2:end]]
        ∂bluff_body_trajectory_∂bluff_body_state_params = [calculate_bluff_body_state_from_params_jacobian(bluff_body, t, bluff_body_state_params)[1] for t in time_traj]
        ∂bluff_body_trajectory_∂bluff_body_params = [calculate_bluff_body_state_from_params_jacobian(bluff_body, t, bluff_body_state_params)[2] for t in time_traj]
        
        # Initialize sensitivity matrices
        ∂aquarium_state_k_∂fluid_properties = copy(initial_aquarium_state_fluid_properties_jacobian)
        ∂aquarium_state_k_∂swimmer_params = copy(initial_aquarium_state_swimmer_params_jacobian)
        ∂aquarium_state_k_∂bluff_body_params = copy(initial_aquarium_state_bluff_body_params_jacobian)
        ∂aquarium_state_k_∂control_params = zeros(n_states, length(swimmer_control_params))
        ∂aquarium_state_k_∂bluff_body_state_params = zeros(n_states, length(bluff_body_state_params))

        # Initialize with stage 1 contribution
        objective_trajectory[1] = calculate_stage_objective(
            tank,
            time_traj[1],
            aquarium_state_traj[1],
            bluff_body_state_traj[1],
            length(swimmer_control_trajectory) > 0 ? swimmer_control_trajectory[1] : zeros(0)
        )

        ∂stage_1_∂x_k, ∂stage_1_∂bb_k, ∂stage_1_∂u_k, ∂stage_1_∂fluid_props, ∂stage_1_∂swimmer_params, ∂stage_1_∂bluff_body_params =
            calculate_stage_objective_gradients(
                tank,
                time_traj[1],
                aquarium_state_traj[1],
                bluff_body_state_traj[1],
                length(swimmer_control_trajectory) > 0 ? swimmer_control_trajectory[1] : zeros(0)
            )

        # Initialize accumulated gradients
        ∂objective_∂fluid_properties_gradient = (∂stage_1_∂x_k' * ∂aquarium_state_k_∂fluid_properties + ∂stage_1_∂fluid_props')[:]
        ∂objective_∂swimmer_params_gradient = (∂stage_1_∂x_k' * ∂aquarium_state_k_∂swimmer_params + ∂stage_1_∂swimmer_params')[:]
        ∂objective_∂bluff_body_params_gradient = (∂stage_1_∂x_k' * ∂aquarium_state_k_∂bluff_body_params + ∂stage_1_∂bluff_body_params')[:]

        if !isempty(swimmer_control_params)
            ∂objective_∂control_params_gradient = (∂stage_1_∂x_k' * ∂aquarium_state_k_∂control_params +
                ∂stage_1_∂u_k' * ∂control_trajectory_∂control_params[1])[:]
        else
            ∂objective_∂control_params_gradient = zeros(0)
        end

        if !isempty(bluff_body_state_params)
            ∂objective_∂bluff_body_state_params_gradient = (∂stage_1_∂x_k' * ∂aquarium_state_k_∂bluff_body_state_params +
                ∂stage_1_∂bb_k' * ∂bluff_body_trajectory_∂bluff_body_state_params[1])[:]
            ∂objective_∂bluff_body_params_gradient .+= (∂stage_1_∂bb_k' * ∂bluff_body_trajectory_∂bluff_body_params[1])[:]
        else
            ∂objective_∂bluff_body_state_params_gradient = zeros(0)
        end

        objective_gradient_wrt_fluid_properties_traj[1] = ∂objective_∂fluid_properties_gradient
        objective_gradient_wrt_swimmer_params_traj[1] = ∂objective_∂swimmer_params_gradient
        objective_gradient_wrt_bluff_body_params_traj[1] = ∂objective_∂bluff_body_params_gradient
        objective_gradient_wrt_control_params_traj[1] = ∂objective_∂control_params_gradient
        objective_gradient_wrt_bluff_body_state_params_traj[1] = ∂objective_∂bluff_body_state_params_gradient

    end

    # Preallocate combined block solve matrices and compute column ranges
    if calculate_objective || compute_swimmer_dynamics_jacobian
        # Calculate column ranges for each component
        current_col = 1
        n_combined_cols = 0

        # Initialize ranges (empty by default)
        swimmer_body_state_col_range = 1:0
        control_dyn_col_range = 1:0
        fluid_props_col_range = 1:0
        swimmer_params_col_range = 1:0
        bluff_body_params_col_range = 1:0
        control_params_col_range = 1:0
        bluff_body_state_params_col_range = 1:0

        # Swimmer dynamics Jacobian columns
        if compute_swimmer_dynamics_jacobian
            n_swimmer_body_states = swimmer.n_body_states

            # State dynamics jacobian columns
            if n_swimmer_body_states > 0
                swimmer_body_state_col_range = current_col:(current_col + n_swimmer_body_states - 1)
                current_col += n_swimmer_body_states
                n_combined_cols += n_swimmer_body_states
            end

            # Control dynamics jacobian columns
            if !isempty(swimmer_control_params)
                n_control_inputs = length(swimmer_control_trajectory[1])
                if n_control_inputs > 0
                    control_dyn_col_range = current_col:(current_col + n_control_inputs - 1)
                    current_col += n_control_inputs
                    n_combined_cols += n_control_inputs
                end
            end
        end

        # Objective gradient columns
        if calculate_objective
            # Fluid properties gradient columns
            if calculate_gradient_wrt_fluid_properties
                n_fluid_props = 4
                if n_fluid_props > 0
                    fluid_props_col_range = current_col:(current_col + n_fluid_props - 1)
                    current_col += n_fluid_props
                    n_combined_cols += n_fluid_props
                end
            end

            # Swimmer params gradient columns
            if calculate_gradient_wrt_swimmer_params
                n_swimmer_params = length(collect_differentiable_params(swimmer))
                if n_swimmer_params > 0
                    swimmer_params_col_range = current_col:(current_col + n_swimmer_params - 1)
                    current_col += n_swimmer_params
                    n_combined_cols += n_swimmer_params
                end
            end

            # Bluff body params gradient columns
            if calculate_gradient_wrt_bluff_body_params
                n_bluff_body_params = length(collect_differentiable_params(bluff_body))
                if n_bluff_body_params > 0
                    bluff_body_params_col_range = current_col:(current_col + n_bluff_body_params - 1)
                    current_col += n_bluff_body_params
                    n_combined_cols += n_bluff_body_params
                end
            end

            # Control params gradient columns
            if calculate_gradient_wrt_control_params
                n_control_params = length(swimmer_control_params)
                if n_control_params > 0
                    control_params_col_range = current_col:(current_col + n_control_params - 1)
                    current_col += n_control_params
                    n_combined_cols += n_control_params
                end
            end

            # Bluff body state params gradient columns
            if calculate_gradient_wrt_bluff_body_state_params
                n_bluff_body_state_params = length(bluff_body_state_params)
                if n_bluff_body_state_params > 0
                    bluff_body_state_params_col_range = current_col:(current_col + n_bluff_body_state_params - 1)
                    current_col += n_bluff_body_state_params
                    n_combined_cols += n_bluff_body_state_params
                end
            end
        end

        # Preallocate matrices
        B_combined = zeros(n_states, n_combined_cols)
        X_combined = zeros(n_states, n_combined_cols)
    end

    if verbose
        print("Setting up aquarium simulation...")
    end

    # Print simulation and solver information
    if verbose
        println("Finished!")
        println("Using solver type: $(solver_type)")
        println("Using preconditioner type: $(preconditioner_type)")
        println("Using pivot type: $(pivot_type)")
        println("Using scaling type: $(scaling_type)")
        println("Lazy preconditioner mode: $(lazy)")
        if solver_type == :pardiso || preconditioner_type == :pardiso
            println("Number of Pardiso threads: $(n_pardiso_threads)")
        end
        if solver_type == :mumps
            println("MUMPS solver initialized")
        end
        if preconditioner_type in (:approx_schur_partial_amg, :approx_schur_full_amg)
            println("AMG smoother type: $(amg_smoother_type)")
        end
    end

    @showprogress enabled=true desc="Simulating aquarium..." for k = 1:N-1

        # Extract bluff body state for current time step
        bluff_body_state_kp1 = bluff_body_state_traj[k+1]

        # Extract swimmer control for current time step
        swimmer_control_k = swimmer_control_trajectory[k]

        # Initialize next state with current state
        newton_iter = 0
        aquarium_state_traj[k+1] = copy(aquarium_state_traj[k])

        if verbose
            println("")
            println("Time step: $(k)")
            println("")
            println("Newton iteration: $(newton_iter)")
            println("")
            print("Constructing KKT system...")
        end

        # Calculate initial residual
        residual = calculate_aquarium_dynamics_residual(
            tank,
            aquarium_state_traj[k+1],
            aquarium_state_traj[k],
            bluff_body_state_kp1,
            swimmer_control_k;
            is_midpoint_state_bluff_body=is_midpoint_bluff_body
        )

        # Calculate initial Jacobian
        kkt_matrix, _, _, _, _, _, _ = calculate_aquarium_dynamics_jacobian(
            tank,
            aquarium_state_traj[k+1],
            aquarium_state_traj[k],
            bluff_body_state_kp1,
            swimmer_control_k;
            is_midpoint_state_bluff_body=is_midpoint_bluff_body
        )

        if verbose
            println(" Finished!")
        end

        # Compute scaling factors once (will be reused in Newton iterations)
        left_scale, right_scale = scale_linear_system!(kkt_matrix, residual; scaling_type=scaling_type, verbose=verbose)

        # Apply dual regularization if specified
        apply_regularization!(kkt_matrix;
            regularization_indices=tank.dual_indices,
            regularization_value=dual_regularization,
            verbose=verbose
        )

        # Apply primal regularization if specified
        apply_regularization!(kkt_matrix;
            regularization_indices=tank.primal_indices,
            regularization_value=-primal_regularization,
            verbose=verbose
        )

        # Apply pivoting if specified
        permutation, inverse_permutation = pivot_linear_system!(kkt_matrix, residual; pivot_type=pivot_type, verbose=verbose)

        # For lazy mode: compute preconditioner/factorization once per timestep
        # For non-lazy mode: preconditioner will be recomputed on each Newton iteration
        schur_dim = tank.n_duals

        preconditioner = calculate_preconditioner(kkt_matrix,
            solution_vector, preconditioner_type;
            preconditioner_solver=preconditioner_solver,
            ilu_drop_tolerance=ilu_drop_tolerance,
            amg_smoother_type=amg_smoother_type,
            verbose=verbose,
            schur_dimension=schur_dim
        )

        # Newton-Raphson iteration
        # Use scaled residual for convergence check
        scaled_residual_norm = maximum(abs.(residual .* left_scale))
        while scaled_residual_norm > newton_tolerance && newton_iter < max_newton_iterations

            newton_iter += 1

            if newton_iter != 1

                if verbose
                    println("")
                    println("Newton iteration: $(newton_iter)")
                    println("")
                    print("Constructing KKT system...")
                end

                # Calculate Jacobian
                kkt_matrix, _, _, _, _, _, _ = calculate_aquarium_dynamics_jacobian(
                    tank,
                    aquarium_state_traj[k+1],
                    aquarium_state_traj[k],
                    bluff_body_state_kp1,
                    swimmer_control_k;
                    is_midpoint_state_bluff_body=is_midpoint_bluff_body
                )

                if verbose
                    println(" Finished!")
                end

                if lazy
                    # Reuse the scaling factors computed before the Newton loop
                    scale_linear_system!(kkt_matrix, residual, left_scale, right_scale; verbose=verbose)

                    # Apply dual regularization if specified
                    apply_regularization!(kkt_matrix;
                        regularization_indices=tank.dual_indices,
                        regularization_value=dual_regularization,
                        verbose=verbose
                    )

                    # Apply primal regularization if specified
                    apply_regularization!(kkt_matrix;
                        regularization_indices=tank.primal_indices,
                        regularization_value=-primal_regularization,
                        verbose=verbose
                    )

                    pivot_linear_system!(kkt_matrix, residual, permutation; verbose=verbose)

                else
                    # Scale linear system
                    left_scale, right_scale = scale_linear_system!(kkt_matrix, residual; scaling_type=scaling_type, verbose=verbose)

                    # Apply dual regularization if specified
                    apply_regularization!(kkt_matrix;
                        regularization_indices=tank.dual_indices,
                        regularization_value=dual_regularization,
                        verbose=verbose
                    )

                    # Apply primal regularization if specified
                    apply_regularization!(kkt_matrix;
                        regularization_indices=tank.primal_indices,
                        regularization_value=-primal_regularization,
                        verbose=verbose
                    )

                    # Apply pivoting if specified
                    permutation, inverse_permutation = pivot_linear_system!(kkt_matrix, residual; pivot_type=pivot_type, verbose=verbose)

                    # Compute preconditioner for non-lazy methods
                    preconditioner = calculate_preconditioner(kkt_matrix,
                        solution_vector, preconditioner_type;
                        preconditioner_solver=preconditioner_solver,
                        ilu_drop_tolerance=ilu_drop_tolerance,
                        amg_smoother_type=amg_smoother_type,
                        verbose=verbose,
                        schur_dimension=schur_dim
                    )
                end

            end

            # Solve linear system
            linear_solve!(solution_vector, kkt_matrix,
                -residual, solver, solver_type;
                preconditioner=preconditioner,
                gmres_tolerance=gmres_tolerance,
                gmres_max_iterations=gmres_max_iterations,
                right_scale=right_scale,
                inverse_permutation=inverse_permutation,
                verbose=verbose
            )

            # Update states
            aquarium_state_traj[k+1] .+= solution_vector

            # Recalculate residual
            residual = calculate_aquarium_dynamics_residual(
                tank,
                aquarium_state_traj[k+1],
                aquarium_state_traj[k],
                bluff_body_state_kp1,
                swimmer_control_k;
                is_midpoint_state_bluff_body=is_midpoint_bluff_body
            )

            # Compute scaled residual norm for convergence check
            scaled_residual_norm = maximum(abs.(residual .* left_scale))

            if verbose
                println("")
                println("\e[1mScaled residual norm: $(scaled_residual_norm)\e[0m")
            end

        end

        if newton_iter >= max_newton_iterations
            @warn "Newton iteration did not converge at time step $(k)"
        end

        if calculate_objective || compute_swimmer_dynamics_jacobian && gradient_method == :forward

            if verbose
                print("Calculating objective gradients using forward mode...")
            end

            _,
            ∂aquarium_dynamics_∂aquarium_state_k,
            ∂aquarium_dynamics_∂swimmer_control_k,
            ∂aquarium_dynamics_∂fluid_properties,
            ∂aquarium_dynamics_∂swimmer_params,
            ∂aquarium_dynamics_∂bluff_body_params,
            ∂aquarium_dynamics_∂bluff_body_state_kp1 = calculate_aquarium_dynamics_jacobian(
                tank,
                aquarium_state_traj[k+1],
                aquarium_state_traj[k],
                bluff_body_state_kp1,
                swimmer_control_k;
                is_midpoint_state_bluff_body=is_midpoint_bluff_body
            )

            # Reuse the scaling factors computed before the Newton loop
            # scale_linear_system_matrix!(kkt_matrix, left_scale, right_scale; verbose=verbose)

            # # Apply dual regularization if specified (using precomputed positions)
            # apply_regularization!(kkt_matrix;
            #     regularization_indices=tank.dual_indices,
            #     regularization_value=dual_regularization,
            #     verbose=verbose
            # )

            # apply_regularization!(kkt_matrix;
            #     regularization_indices=tank.primal_indices,
            #     regularization_value=-primal_regularization,
            #     verbose=verbose
            # )

            # pivot_linear_system_matrix!(kkt_matrix, permutation; verbose=verbose)

            # Fill B_combined with RHS matrices using precomputed column ranges
            if compute_swimmer_dynamics_jacobian
                if length(swimmer_body_state_col_range) > 0
                    # Extract swimmer body state columns from the full aquarium state Jacobian
                    B_combined[:, swimmer_body_state_col_range] .= -Matrix(∂aquarium_dynamics_∂aquarium_state_k[:, tank.swimmer_body_state_indices])
                    scale_rhs_matrix!(view(B_combined, :, swimmer_body_state_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, swimmer_body_state_col_range), permutation)
                end

                if length(control_dyn_col_range) > 0
                    B_combined[:, control_dyn_col_range] .= -Matrix(∂aquarium_dynamics_∂swimmer_control_k)
                    scale_rhs_matrix!(view(B_combined, :, control_dyn_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, control_dyn_col_range), permutation)
                end
            end

            if calculate_objective
                # Implicit function theorem
                gradient_residual_fluid_properties = ∂aquarium_dynamics_∂aquarium_state_k * ∂aquarium_state_k_∂fluid_properties +
                    ∂aquarium_dynamics_∂fluid_properties

                gradient_residual_swimmer_params = ∂aquarium_dynamics_∂aquarium_state_k * ∂aquarium_state_k_∂swimmer_params +
                    ∂aquarium_dynamics_∂swimmer_params

                gradient_residual_bluff_body_params = ∂aquarium_dynamics_∂aquarium_state_k * ∂aquarium_state_k_∂bluff_body_params +
                    ∂aquarium_dynamics_∂bluff_body_params

                gradient_residual_control_params = ∂aquarium_dynamics_∂swimmer_control_k * ∂control_trajectory_∂control_params[k] +
                    ∂aquarium_dynamics_∂aquarium_state_k * ∂aquarium_state_k_∂control_params

                gradient_residual_bluff_body_state_params =
                    ∂aquarium_dynamics_∂bluff_body_state_kp1 * ∂bluff_body_trajectory_∂bluff_body_state_params[k+1] +
                    ∂aquarium_dynamics_∂aquarium_state_k * ∂aquarium_state_k_∂bluff_body_state_params
                gradient_residual_bluff_body_params .+=
                    ∂aquarium_dynamics_∂bluff_body_state_kp1 * ∂bluff_body_trajectory_∂bluff_body_params[k+1]

                if length(fluid_props_col_range) > 0
                    B_combined[:, fluid_props_col_range] .= -gradient_residual_fluid_properties
                    scale_rhs_matrix!(view(B_combined, :, fluid_props_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, fluid_props_col_range), permutation)
                end

                if length(swimmer_params_col_range) > 0
                    B_combined[:, swimmer_params_col_range] .= -gradient_residual_swimmer_params
                    scale_rhs_matrix!(view(B_combined, :, swimmer_params_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, swimmer_params_col_range), permutation)
                end

                if length(bluff_body_params_col_range) > 0
                    B_combined[:, bluff_body_params_col_range] .= -gradient_residual_bluff_body_params
                    scale_rhs_matrix!(view(B_combined, :, bluff_body_params_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, bluff_body_params_col_range), permutation)
                end

                if length(control_params_col_range) > 0
                    B_combined[:, control_params_col_range] .= -gradient_residual_control_params
                    scale_rhs_matrix!(view(B_combined, :, control_params_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, control_params_col_range), permutation)
                end

                if length(bluff_body_state_params_col_range) > 0
                    B_combined[:, bluff_body_state_params_col_range] .= -gradient_residual_bluff_body_state_params
                    scale_rhs_matrix!(view(B_combined, :, bluff_body_state_params_col_range), left_scale)
                    pivot_rhs_matrix!(view(B_combined, :, bluff_body_state_params_col_range), permutation)
                end
            end

            # Perform single block solve for all RHS matrices
            if n_combined_cols > 0
                block_linear_solve!(X_combined, kkt_matrix, B_combined, solver, solver_type;
                    preconditioner=preconditioner,
                    gmres_tolerance=gmres_tolerance,
                    gmres_max_iterations=gmres_max_iterations,
                    gmres_memory=gmres_memory,
                    right_scale=right_scale,
                    inverse_permutation=inverse_permutation,
                    reuse_factorization=true,
                    verbose=false
                )

                # Extract results from combined solution using precomputed ranges
                if compute_swimmer_dynamics_jacobian
                    if length(swimmer_body_state_col_range) > 0
                        # Extract swimmer body state rows from the solution
                        state_dynamics_jacobian_kp1 .= X_combined[tank.swimmer_body_state_indices, swimmer_body_state_col_range]
                        dynamics_jacobian_wrt_state_traj[k+1] = copy(state_dynamics_jacobian_kp1)
                    end

                    if length(control_dyn_col_range) > 0
                        # Extract swimmer body state rows from the solution
                        control_dynamics_jacobian_kp1 .= X_combined[tank.swimmer_body_state_indices, control_dyn_col_range]
                        dynamics_jacobian_wrt_control_traj[k+1] = copy(control_dynamics_jacobian_kp1)
                    end
                end

                if calculate_objective
                    if length(fluid_props_col_range) > 0
                        ∂aquarium_state_k_∂fluid_properties .= X_combined[:, fluid_props_col_range]
                    end

                    if length(swimmer_params_col_range) > 0
                        ∂aquarium_state_k_∂swimmer_params .= X_combined[:, swimmer_params_col_range]
                    end

                    if length(bluff_body_params_col_range) > 0
                        ∂aquarium_state_k_∂bluff_body_params .= X_combined[:, bluff_body_params_col_range]
                    end

                    if length(control_params_col_range) > 0
                        ∂aquarium_state_k_∂control_params .= X_combined[:, control_params_col_range]
                    end

                    if length(bluff_body_state_params_col_range) > 0
                        ∂aquarium_state_k_∂bluff_body_state_params .= X_combined[:, bluff_body_state_params_col_range]
                    end
                end
            end

            if calculate_objective

                # Compute objective gradients at this timestep
                if k < N-1
                    # Stage objective at k+1
                objective_trajectory[k+1] = calculate_stage_objective(
                    tank,
                    time_traj[k+1],
                    aquarium_state_traj[k+1],
                    bluff_body_state_traj[k+1],
                    swimmer_control_trajectory[k+1]
                )

                ∂stage_∂x_kp1, ∂stage_∂bb_kp1, ∂stage_∂u_kp1, ∂stage_∂fluid_props, ∂stage_∂swimmer_params, ∂stage_∂bluff_body_params =
                    calculate_stage_objective_gradients(
                        tank,
                        time_traj[k+1],
                        aquarium_state_traj[k+1],
                        bluff_body_state_traj[k+1],
                        swimmer_control_trajectory[k+1]
                    )

                # Accumulate gradients
                if calculate_gradient_wrt_fluid_properties
                    objective_gradient_wrt_fluid_properties_traj[k+1] = (∂stage_∂x_kp1' * ∂aquarium_state_k_∂fluid_properties + ∂stage_∂fluid_props')[:]
                    ∂objective_∂fluid_properties_gradient += objective_gradient_wrt_fluid_properties_traj[k+1]
                end

                if calculate_gradient_wrt_swimmer_params
                    objective_gradient_wrt_swimmer_params_traj[k+1] = (∂stage_∂x_kp1' * ∂aquarium_state_k_∂swimmer_params + ∂stage_∂swimmer_params')[:]
                    ∂objective_∂swimmer_params_gradient += objective_gradient_wrt_swimmer_params_traj[k+1]
                end

                if calculate_gradient_wrt_bluff_body_params
                    objective_gradient_wrt_bluff_body_params_traj[k+1] = (∂stage_∂x_kp1' * ∂aquarium_state_k_∂bluff_body_params + ∂stage_∂bluff_body_params')[:]
                    ∂objective_∂bluff_body_params_gradient += objective_gradient_wrt_bluff_body_params_traj[k+1]
                end

                if calculate_gradient_wrt_control_params && !isempty(swimmer_control_params)
                    objective_gradient_wrt_control_params_traj[k+1] = (∂stage_∂x_kp1' * ∂aquarium_state_k_∂control_params +
                        ∂stage_∂u_kp1' * ∂control_trajectory_∂control_params[k+1])[:]
                    ∂objective_∂control_params_gradient += objective_gradient_wrt_control_params_traj[k+1]
                end

                if calculate_gradient_wrt_bluff_body_state_params && !isempty(bluff_body_state_params)
                    objective_gradient_wrt_bluff_body_state_params_traj[k+1] = (∂stage_∂x_kp1' * ∂aquarium_state_k_∂bluff_body_state_params +
                        ∂stage_∂bb_kp1' * ∂bluff_body_trajectory_∂bluff_body_state_params[k+1])[:]
                    ∂objective_∂bluff_body_state_params_gradient += objective_gradient_wrt_bluff_body_state_params_traj[k+1]
                end

            else
                # Terminal objective at final time
                objective_trajectory[k+1] = calculate_terminal_objective(
                    tank,
                    time_traj[k+1],
                    aquarium_state_traj[k+1],
                    bluff_body_state_traj[k+1]
                )

                ∂terminal_∂x_final, ∂terminal_∂bb_final, ∂terminal_∂fluid_props, ∂terminal_∂swimmer_params, ∂terminal_∂bluff_body_params =
                    calculate_terminal_objective_gradients(
                        tank,
                        time_traj[k+1],
                        aquarium_state_traj[k+1],
                        bluff_body_state_traj[k+1]
                    )

                # Accumulate terminal gradients
                if calculate_gradient_wrt_fluid_properties
                    ∂objective_∂fluid_properties_gradient += (∂terminal_∂x_final' * ∂aquarium_state_k_∂fluid_properties + ∂terminal_∂fluid_props')[:]
                end

                if calculate_gradient_wrt_swimmer_params
                    ∂objective_∂swimmer_params_gradient += (∂terminal_∂x_final' * ∂aquarium_state_k_∂swimmer_params + ∂terminal_∂swimmer_params')[:]
                end

                if calculate_gradient_wrt_bluff_body_params
                    ∂objective_∂bluff_body_params_gradient += (∂terminal_∂x_final' * ∂aquarium_state_k_∂bluff_body_params + ∂terminal_∂bluff_body_params')[:]
                end

                if calculate_gradient_wrt_control_params && !isempty(swimmer_control_params)
                    ∂objective_∂control_params_gradient += (∂terminal_∂x_final' * ∂aquarium_state_k_∂control_params)[:]
                end

                if calculate_gradient_wrt_bluff_body_state_params && !isempty(bluff_body_state_params)
                    ∂objective_∂bluff_body_state_params_gradient += (∂terminal_∂x_final' * ∂aquarium_state_k_∂bluff_body_state_params +
                        ∂terminal_∂bb_final' * ∂bluff_body_trajectory_∂bluff_body_state_params[k+1])[:]
                end
            end

            if verbose
                println("Finished!")
                println("")
                println("\e[1mObjective value: $(sum(objective_trajectory[1:k+1]))\e[0m")
                println("\e[1mObjective gradient w.r.t. fluid properties: $(∂objective_∂fluid_properties_gradient)\e[0m")
                println("\e[1mObjective gradient w.r.t. swimmer params: $(∂objective_∂swimmer_params_gradient)\e[0m")
                println("\e[1mObjective gradient w.r.t. bluff body params: $(∂objective_∂bluff_body_params_gradient)\e[0m")
                if !isempty(swimmer_control_params)
                    println("\e[1mObjective gradient w.r.t. control params: $(∂objective_∂control_params_gradient)\e[0m")
                end
                if !isempty(bluff_body_state_params)
                    println("\e[1mObjective gradient w.r.t. bluff body state params: $(∂objective_∂bluff_body_state_params_gradient)\e[0m")
                end
                println("")
            end

        end # if calculate_objective

    end # if calculate_objective || compute_swimmer_dynamics_jacobian && gradient_method == :forward

    end

    # Cleanup Pardiso solver if used
    if PARDISO_LOADED[]
        if solver_type == :pardiso
            pardiso_set_phase!(solver, Val(:RELEASE_ALL))
            pardiso_solve!(solver)
        end
        if preconditioner_type == :pardiso
            pardiso_set_phase!(preconditioner_solver, Val(:RELEASE_ALL))
            pardiso_solve!(preconditioner_solver)
        end
    end

    if solver_type == :mumps
        finalize(solver)
        MPI.Finalize()
    end

    # Extract component trajectories
    fluid_state_traj = [extract_fluid_state(tank, aquarium_state_traj[k]) for k = 1:N]

    trajectories = Dict{Symbol, Any}(
        :time_traj => time_traj,
        :aquarium_state_traj => aquarium_state_traj,
        :fluid_state_traj => fluid_state_traj
    )

    swimmer_state_traj = [extract_swimmer_state(tank, aquarium_state_traj[k]) for k = 1:N]
    trajectories[:swimmer_state_traj] = swimmer_state_traj
    trajectories[:control_traj] = swimmer_control_trajectory

    trajectories[:bluff_body_state_traj] = bluff_body_state_traj
    trajectories[:bluff_body_traj_is_midpoint] = is_midpoint_bluff_body

    if calculate_objective
        objective_value = sum(objective_trajectory)

        trajectories[:objective_value] = [objective_value]
        trajectories[:objective_traj] = objective_trajectory
        trajectories[:objective_gradient_wrt_fluid_properties_traj] = objective_gradient_wrt_fluid_properties_traj
        trajectories[:objective_gradient_wrt_swimmer_params_traj] = objective_gradient_wrt_swimmer_params_traj
        trajectories[:objective_gradient_wrt_bluff_body_params_traj] = objective_gradient_wrt_bluff_body_params_traj
        trajectories[:objective_gradient_wrt_control_params_traj] = objective_gradient_wrt_control_params_traj
        trajectories[:objective_gradient_wrt_bluff_body_state_params_traj] = objective_gradient_wrt_bluff_body_state_params_traj
        trajectories[:objective_gradient_wrt_fluid_properties] = ∂objective_∂fluid_properties_gradient
        trajectories[:objective_gradient_wrt_swimmer_params] = ∂objective_∂swimmer_params_gradient
        trajectories[:objective_gradient_wrt_bluff_body_params] = ∂objective_∂bluff_body_params_gradient
        trajectories[:objective_gradient_wrt_control_params] = ∂objective_∂control_params_gradient
        trajectories[:objective_gradient_wrt_bluff_body_state_params] = ∂objective_∂bluff_body_state_params_gradient
    end

    # Store dynamics Jacobians if computed
    if compute_swimmer_dynamics_jacobian
        trajectories[:dynamics_jacobian_wrt_state_traj] = dynamics_jacobian_wrt_state_traj
        trajectories[:dynamics_jacobian_wrt_control_traj] = dynamics_jacobian_wrt_control_traj
    end

    return trajectories

end