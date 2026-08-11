@testitem "Swing-up trajopt: :prescribed converges faster than :pd" begin
    using AquariumClosed
    using ForwardDiff
    using LinearAlgebra

    dt = 0.01
    n_steps = 30
    L = 0.5
    m = 1.0
    moi = (1 / 12) * m * L^2
    g = 9.81

    Kp_motor = 1.0e3
    Kd_motor = 50.0

    n_knots = 5

    function build_pendulum(mode::Symbol)
        ActuatedPendulum(dt;
            bar_length=L, mass=m, moi=moi,
            hinge_position=[0.0, 0.0],
            Kp=Kp_motor, Kd=Kd_motor, max_torque=1.0e6,
            n_boundary_nodes=4,
            gravity=[0.0, -g],
            actuation_mode=mode,
        )
    end

    function interp_knots(knots::AbstractVector, n_steps::Int)
        T = eltype(knots)
        n = length(knots)
        out = Vector{T}(undef, n_steps)
        @inbounds for k in 1:n_steps
            x = (k - 1) / max(n_steps - 1, 1) * (n - 1) + 1
            i = clamp(floor(Int, x), 1, n - 1)
            α = x - i
            out[k] = (1 - α) * knots[i] + α * knots[i + 1]
        end
        return out
    end

    θ_0 = 0.0
    θ_target = π / 4

    function build_cost(mode::Symbol)
        sys = build_pendulum(mode)
        max_config = pendulum_maximal_from_minimal(sys, [θ_0])
        body_state_0 = vcat(max_config, zeros(3))
        state_0 = initialize_solid_state(sys, body_state_0)

        cost = function (knots)
            θ_des_traj = interp_knots(knots, n_steps)
            ctrl = if mode == :pd
                [[θ_des_traj[k], zero(eltype(knots))] for k in 1:n_steps]
            else
                [[θ_des_traj[k]] for k in 1:n_steps]
            end
            res = simulate_solid_system(sys, state_0, n_steps * dt;
                control_trajectory=ctrl, verbose=false)
            θ_final = res[:system_state_traj][end][3]
            return (θ_final - θ_target)^2 + 1e-6 * sum(abs2, knots)
        end
        return cost, sys
    end

    cost_pd, _ = build_cost(:pd)
    cost_pres, sys_pres = build_cost(:prescribed)

    p0 = fill(θ_target / 2, n_knots)

    g_pd = ForwardDiff.gradient(cost_pd, p0)
    g_pres = ForwardDiff.gradient(cost_pres, p0)

    @test all(isfinite, g_pd)
    @test all(isfinite, g_pres)

    function backtracking_gd(cost, p0; n_iters, α0=1.0, ρ=0.5, c1=1e-4, max_ls=40)
        p = copy(p0)
        f = cost(p)
        hist = [f]
        for _ in 1:n_iters
            grad = ForwardDiff.gradient(cost, p)
            d = -grad
            slope = dot(grad, d)
            α = α0
            f_new = f
            for _ in 1:max_ls
                f_new = cost(p .+ α .* d)
                if f_new ≤ f + c1 * α * slope
                    break
                end
                α *= ρ
            end
            p = p .+ α .* d
            f = f_new
            push!(hist, f)
        end
        return p, hist
    end

    n_iters = 12
    _,        hist_pd = backtracking_gd(cost_pd, p0; n_iters=n_iters)
    p_pres, hist_pres = backtracking_gd(cost_pres, p0; n_iters=n_iters)

    # Initial costs differ slightly because PD tracks θ_des with steady-state
    # offset, so we don't compare initial cost directly. We compare *relative*
    # reductions and absolute final cost.
    @test hist_pd[end] < hist_pd[1]
    @test hist_pres[end] < hist_pres[1]
    @test hist_pres[end] < hist_pd[end]

    rel_pres = (hist_pres[1] - hist_pres[end]) / hist_pres[1]
    rel_pd = (hist_pd[1] - hist_pd[end]) / hist_pd[1]
    @test rel_pres > rel_pd

    # Prescribed should reach < 10% of starting cost; PD with line search will
    # take much longer because each step shrinks α to compensate for Kp scaling.
    @test hist_pres[end] < 0.1 * hist_pres[1]

    # Recovered torques from prescribed solution are physically reasonable.
    θ_des_final = interp_knots(p_pres, n_steps)
    ctrl_final = [[θ_des_final[k]] for k in 1:n_steps]
    max_config_chk = pendulum_maximal_from_minimal(sys_pres, [θ_0])
    state_0_chk = initialize_solid_state(sys_pres, vcat(max_config_chk, zeros(3)))
    res_final = simulate_solid_system(sys_pres, state_0_chk, n_steps * dt;
        control_trajectory=ctrl_final, verbose=false)

    discrete_torques = [extract_prescribed_angle_torques(sys_pres, s)[1]
                        for s in res_final[:system_state_traj][2:end]]
    physical_torques = discrete_torques ./ dt   # multiplier is dt-scaled
    @test all(isfinite, physical_torques)

    # Steady-state torques (drop first few transient steps where the impulsive
    # initial-acceleration component dominates) should be order-of-magnitude
    # m*g*(L/2) ≈ 2.5 Nm plus inertia contributions.
    steady_torques = physical_torques[6:end]
    @test maximum(abs, steady_torques) < 30.0

    # Even including transients, torques should be finite and not explosive
    # (Lagrange multipliers are well-defined here, not blowing up).
    @test maximum(abs, physical_torques) < 1000.0
end
