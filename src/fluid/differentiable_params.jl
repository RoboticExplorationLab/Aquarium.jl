#############################################################################################
## Fluid collect_differentiable_params / inject_differentiable_params
##
## Mirror of the solid-side `src/solid/differentiable_params.jl` API for the fluid side.
## Parameter layout is canonical and matches the column ordering of the analytic
## `∂stationarity_∂fluid_properties` jacobian:
##
##     [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
##
## `inject_differentiable_params(fluid, p)` rebuilds a `Fluid{T, eltype(p)}` with the new
## parameter values, recomputing the param-derived caches (`cell_mass`,
## `constant_boundary_condition_vector`, `continuity_vector`) from the injected params so a
## `ForwardDiff.Dual` flowing in through `p` propagates cleanly into anything downstream
## that reads those caches.
#############################################################################################

function n_differentiable_params(::Fluid)
    return 4
end

function collect_differentiable_params(fluid::Fluid)
    return Float64[
        fluid.density,
        fluid.dynamic_viscosity,
        fluid.boundary_velocity[1],
        fluid.boundary_velocity[2],
    ]
end

function inject_differentiable_params(fluid::Fluid, params_vec::AbstractVector)
    S = eltype(params_vec)

    density = params_vec[1]
    dynamic_viscosity = params_vec[2]
    boundary_velocity = S[params_vec[3], params_vec[4]]

    fvm_grid = fluid.fvm_grid

    # Recompute param-derived caches
    cell_mass = density * fvm_grid.h_x * fvm_grid.h_y

    constant_boundary_condition_vector = calculate_constant_boundary_condition_vector(
        fvm_grid,
        boundary_velocity,
        fluid.boundary_condition_type,
    )

    continuity_vector = fluid.original_divergence * constant_boundary_condition_vector

    T = typeof(fluid.time_step)

    return Fluid{T, S}(
        density,
        dynamic_viscosity,
        boundary_velocity,
        fluid.time_step,
        fluid.gravity_constant,
        cell_mass,
        fluid.cell_area,
        fluid.external_pressure_gradient,
        fluid.gravitational_acceleration,
        fluid.external_pressure_gradient_force,
        fvm_grid,
        fluid.constant_boundary_condition_matrix,
        constant_boundary_condition_vector,
        fluid.laplacian,
        fluid.original_divergence,
        fluid.divergence,
        continuity_vector,
        fluid.midpoint_operators,
        fluid.boundary_condition_type,
        fluid.n_boundary_conditions,
        fluid.state_indices,
        fluid.dual_indices,
        fluid.velocity_indices,
        fluid.continuity_dual_indices,
        fluid.n_states,
        fluid.n_constraints,
        fluid.n_velocities,
        fluid.n_continuity_constraints,
    )
end


@testitem "Fluid collect_differentiable_params — canonical ordering" begin
    using AquariumClosed
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (8, 8)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )


    p = collect_differentiable_params(fluid)
    @test p isa Vector{Float64}
    @test length(p) == 4
    @test p[1] == density
    @test p[2] == dynamic_viscosity
    @test p[3] == boundary_velocity[1]
    @test p[4] == boundary_velocity[2]
end

@testitem "Fluid inject_differentiable_params — round-trip equality" begin
    using AquariumClosed
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (8, 8)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )


    p0 = collect_differentiable_params(fluid)
    fluid_rt = inject_differentiable_params(fluid, p0)

    @test fluid_rt isa Fluid
    @test fluid_rt.density ≈ fluid.density
    @test fluid_rt.dynamic_viscosity ≈ fluid.dynamic_viscosity
    @test fluid_rt.boundary_velocity ≈ fluid.boundary_velocity

    # Derived caches must be recomputed consistently with the injected params
    @test fluid_rt.cell_mass ≈ fluid.cell_mass
    @test fluid_rt.constant_boundary_condition_vector ≈ fluid.constant_boundary_condition_vector
    @test fluid_rt.continuity_vector ≈ fluid.continuity_vector

    # Grid-level fields must be preserved exactly
    @test fluid_rt.fvm_grid === fluid.fvm_grid || fluid_rt.fvm_grid == fluid.fvm_grid
    @test fluid_rt.time_step == fluid.time_step
    @test fluid_rt.boundary_condition_type == fluid.boundary_condition_type
end

@testitem "Fluid inject_differentiable_params — perturbation updates derived caches" begin
    using AquariumClosed
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (8, 8)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )


    p0 = collect_differentiable_params(fluid)

    # Perturb density — cell_mass should scale linearly
    p_density_perturbed = copy(p0)
    p_density_perturbed[1] = density * 2.0
    fluid_2x = inject_differentiable_params(fluid, p_density_perturbed)
    @test fluid_2x.density ≈ density * 2.0
    @test fluid_2x.cell_mass ≈ fluid.cell_mass * 2.0

    # Perturb boundary velocity — constant_boundary_condition_vector should change
    p_bv_perturbed = copy(p0)
    p_bv_perturbed[3] = boundary_velocity[1] + 0.5
    fluid_bv = inject_differentiable_params(fluid, p_bv_perturbed)
    @test fluid_bv.boundary_velocity[1] ≈ boundary_velocity[1] + 0.5
    @test fluid_bv.constant_boundary_condition_vector != fluid.constant_boundary_condition_vector
end

@testitem "Fluid inject_differentiable_params — Dual propagation" begin
    using AquariumClosed
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (8, 8)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )


    # A scalar closure depending on density through cell_mass.
    # ForwardDiff.gradient must flow duals through inject.
    loss = p -> begin
        f = inject_differentiable_params(fluid, p)
        return f.cell_mass
    end

    p0 = collect_differentiable_params(fluid)
    grad = ForwardDiff.gradient(loss, p0)

    @test length(grad) == 4
    # ∂cell_mass/∂density = h_x * h_y
    @test grad[1] ≈ fluid.fvm_grid.h_x * fluid.fvm_grid.h_y
    # Other slots should be zero
    @test grad[2] == 0.0
    @test grad[3] == 0.0
    @test grad[4] == 0.0
end

@testitem "Fluid inject_differentiable_params — ForwardDiff.gradient vs finite-diff on stationarity loss" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (8, 8)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )


    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)

    loss = p -> begin
        f = inject_differentiable_params(fluid, p)
        res = calculate_fluid_stationarity_residual(f, fluid_state_kp1, fluid_state_k;
            recompute_bc_vector=true)
        return 0.5 * sum(abs2, res)
    end

    p0 = collect_differentiable_params(fluid)

    grad_fd = ForwardDiff.gradient(loss, p0)
    grad_finite = FiniteDiff.finite_difference_gradient(loss, p0)

    @test all(isfinite, grad_fd)
    @test all(isfinite, grad_finite)
    @test grad_fd ≈ grad_finite rtol=1e-5
end
