#############################################################################################
## Standalone time-stepping driver for composition-based passive/actuated systems.
##
## Replacement for the legacy `simulate_solid_system` + `calculate_midpoint_state_trajectory`
## that were deleted with SolidSystem.jl. Uses the implicit midpoint variational
## integrator: at each step, solve the KKT system
##   dynamics_residual(x_{k+1}, x_k, u_k) = 0
## via Newton iteration with `calculate_solid_dynamics_residual` and
## `calculate_solid_dynamics_jacobian`.
##
## Intended for tank-free experiments (e.g., a pendulum simulation with no fluid). For
## fluid-coupled simulations use `simulate_aquarium` instead.
#############################################################################################

function simulate_solid_system(
    system::SolidSystem,
    initial_state::AbstractVector,
    final_time::Real;
    control_trajectory::Union{Nothing, AbstractVector} = nothing,
    max_newton_iterations::Int = 50,
    newton_tolerance::Real = 1e-8,
    verbose::Bool = false,
)
    time_step = system.time_step
    n_time_steps = Int(round(final_time / time_step)) + 1

    # Determine element type: system params, initial state, or control trajectory
    # may carry ForwardDiff.Dual under AD.
    control_eltype = control_trajectory === nothing ? Float64 : eltype(eltype(control_trajectory))
    T = promote_type(Float64, _system_param_type(system), eltype(initial_state), control_eltype)
    state_traj = Vector{Vector{T}}(undef, n_time_steps)
    state_traj[1] = T.(initial_state)
    time_traj = collect(0.0:time_step:final_time)

    for k in 1:(n_time_steps - 1)
        x_k = state_traj[k]
        u_k = if control_trajectory === nothing
            T[]
        else
            control_trajectory[k]
        end

        # Newton's method: solve dynamics_residual(x_{k+1}, x_k, u_k) = 0 for x_{k+1}.
        x_kp1 = copy(x_k)
        converged = false
        for iter in 1:max_newton_iterations
            r = calculate_solid_dynamics_residual(system, x_kp1, x_k, u_k)
            res_norm = sqrt(sum(abs2, ForwardDiff.value.(r)))
            if verbose
                @info "Step $k iter $iter: ‖r‖ = $res_norm"
            end
            if res_norm < newton_tolerance
                converged = true
                break
            end
            J_kp1, _, _, _ = calculate_solid_dynamics_jacobian(system, x_kp1, x_k, u_k)
            δ = J_kp1 \ (-r)
            x_kp1 = x_kp1 .+ δ
        end
        if !converged
            @warn "Newton did not converge at step $k"
        end
        state_traj[k + 1] = x_kp1
    end

    configuration_traj = [x[system.configuration_indices] for x in state_traj]
    velocity_traj = [x[system.velocity_indices] for x in state_traj]

    return Dict(
        :time_traj => time_traj,
        :system_state_traj => state_traj,
        :configuration_traj => configuration_traj,
        :velocity_traj => velocity_traj,
    )
end

# Apply the variational integrator's midpoint rule (q_mid = q - (dt/2) * v) to every
# state in a trajectory. Matches the legacy `calculate_midpoint_state_trajectory` signature.
function calculate_midpoint_state_trajectory(
    system::SolidSystem,
    state_trajectory::AbstractVector,
)
    return [calculate_midpoint_state(system, x) for x in state_trajectory]
end


@testitem "Prescribed mode simulation" begin
    using Aquarium
    using FiniteDiff

    @testset "ActuatedPendulum: constant angle hold" begin
        ap = ActuatedPendulum(0.01;
            bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0],
            Kp=50.0, Kd=5.0, max_torque=100.0,
            actuation_mode=:prescribed)

        init_state = initialize_solid_state(ap, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        θ_desired = 0.3
        n_steps = 100
        control_traj = [[θ_desired] for _ in 1:n_steps]

        result = simulate_solid_system(ap, init_state, n_steps * ap.time_step;
            control_trajectory=control_traj)

        state_traj = result[:system_state_traj]
        @test length(state_traj) == n_steps + 1

        for k in 2:length(state_traj)
            @test abs(state_traj[k][3] - θ_desired) < 1e-8
        end

        # Discrete multiplier ≈ dt * m*g*(L/2)*cos(θ)
        final_state = state_traj[end]
        torques = extract_prescribed_angle_torques(ap, final_state)
        expected_discrete = ap.time_step * 2.0 * 9.81 * 0.5 * cos(θ_desired)
        @test abs(torques[1]) ≈ expected_discrete rtol=0.05
    end

    @testset "ActuatedPendulum: sinusoidal trajectory" begin
        ap = ActuatedPendulum(0.01;
            bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0],
            actuation_mode=:prescribed)

        init_state = initialize_solid_state(ap, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        n_steps = 200
        dt = ap.time_step
        control_traj = [[0.3 * sin(2π * k * dt)] for k in 1:n_steps]

        result = simulate_solid_system(ap, init_state, n_steps * dt;
            control_trajectory=control_traj)
        state_traj = result[:system_state_traj]
        for k in 2:length(state_traj)
            @test abs(state_traj[k][3] - control_traj[k-1][1]) < 1e-8
        end
    end

    @testset "RExEel: prescribed simulation" begin
        n_links = 3
        rex = RExEel(0.01, n_links;
            bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links),
            Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(10.0, n_links-1),
            actuation_mode=:prescribed)

        q_min = [0.5, 0.0, 0.0, 0.0, 0.0]
        config = rex_eel_maximal_from_minimal(rex, q_min, n_links)
        init_state = initialize_solid_state(rex, vcat(config, zeros(3 * n_links)))

        n_steps = 50
        dt = rex.time_step
        n_joints = n_links - 1
        control_traj = [
            [0.2 * sin(2π * k * dt + j * π/4) for j in 1:n_joints]
            for k in 1:n_steps
        ]

        result = simulate_solid_system(rex, init_state, n_steps * dt;
            control_trajectory=control_traj)
        state_traj = result[:system_state_traj]
        @test length(state_traj) == n_steps + 1

        for k in 2:length(state_traj)
            config_k = state_traj[k][rex.configuration_indices]
            residual = calculate_prescribed_angle_constraint_residual(rex, config_k, control_traj[k-1])
            @test all(abs.(residual) .< 1e-8)
        end
    end

    @testset "pd mode regression" begin
        ap_pd = ActuatedPendulum(0.01;
            bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=10.0)
        init_state = initialize_solid_state(ap_pd, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        control_traj = [[0.3, 0.0] for _ in 1:50]
        result = simulate_solid_system(ap_pd, init_state, 50 * ap_pd.time_step;
            control_trajectory=control_traj)
        @test length(result[:system_state_traj]) == 51
    end

    @testset "prescribed simulation gradient via FiniteDiff" begin
        ap = ActuatedPendulum(0.01;
            bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0],
            Kp=50.0, Kd=5.0, max_torque=100.0,
            actuation_mode=:prescribed)

        init_state = initialize_solid_state(ap, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        n_steps = 10

        function final_theta(u_flat)
            ctrl = [[u_flat[k]] for k in 1:n_steps]
            res = simulate_solid_system(ap, init_state, n_steps * ap.time_step;
                control_trajectory=ctrl)
            return res[:system_state_traj][end][3]
        end

        u0 = fill(0.2, n_steps)
        grad_fd = FiniteDiff.finite_difference_gradient(final_theta, u0)
        @test length(grad_fd) == n_steps
        @test all(isfinite, grad_fd)
        @test all(abs.(grad_fd) .< 10.0)   # O(1) gradients, not O(Kp)
    end
end
