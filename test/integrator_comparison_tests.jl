@testitem "Passive pendulum integrator comparison" begin
    using Aquarium
    using ForwardDiff
    using LinearAlgebra

    function ref_pendulum_dynamics(x; m, g, R, I_hinge, k, c, θ_eq)
        θ, θ_dot = x
        τ_gravity = -(m * g * R) * cos(θ)
        τ_spring = -k * (θ - θ_eq)
        τ_damping = -c * θ_dot
        return [θ_dot, (τ_gravity + τ_spring + τ_damping) / I_hinge]
    end

    function ref_simulate(x0, dt, n_steps; m, g, R, I_hinge, k, c, θ_eq)
        traj = [copy(x0)]
        for _ in 1:n_steps
            xk = traj[end]
            xkp1 = copy(xk)
            for _ in 1:20
                xm = (xkp1 + xk) / 2
                r = xkp1 - xk - dt * ref_pendulum_dynamics(xm; m=m, g=g, R=R, I_hinge=I_hinge, k=k, c=c, θ_eq=θ_eq)
                J = ForwardDiff.jacobian(
                    x -> x - xk - dt * ref_pendulum_dynamics((x + xk) / 2; m=m, g=g, R=R, I_hinge=I_hinge, k=k, c=c, θ_eq=θ_eq), xkp1)
                xkp1 -= J \ r
                norm(r) < 1e-12 && break
            end
            push!(traj, xkp1)
        end
        return traj
    end

    dt = 0.01
    n_steps = 20
    L = 0.5
    m = 5.0
    moi = (1 / 12) * m * L^2
    g = 9.81
    k = 5.0
    c = 1.0

    pendulum = Pendulum(dt;
        bar_length=L, mass=m, moi=moi,
        hinge_position=[0.0, 0.0],
        stiffness=k, damping=c,
        n_boundary_nodes=4,
        gravity=[0.0, -g],
    )

    θ_0 = deg2rad(-45)
    max_config = pendulum_maximal_from_minimal(pendulum, [θ_0])
    initial_body_state = vcat(max_config, zeros(3))
    state_0 = initialize_solid_state(pendulum, initial_body_state)

    trajectories = simulate_solid_system(pendulum, state_0, n_steps * dt; verbose=false)

    R = L / 2
    I_hinge = moi + m * R^2
    ref_traj = ref_simulate([θ_0, 0.0], dt, n_steps; m=m, g=g, R=R, I_hinge=I_hinge, k=k, c=c, θ_eq=0.0)

    θ_aquarium = [trajectories[:system_state_traj][i][3] for i in 1:n_steps+1]
    θ_reference = [ref_traj[i][1] for i in 1:n_steps+1]

    @test maximum(abs.(θ_aquarium .- θ_reference)) < 0.05
end

@testitem "Actuated pendulum integrator comparison" begin
    using Aquarium
    using ForwardDiff
    using LinearAlgebra

    function ref_actuated_dynamics(x, control; m, g, R, I_hinge, Kp, Kd)
        θ, θ_dot = x
        θ_des, ω_des = control
        τ_gravity = -(m * g * R) * cos(θ)
        τ_motor = Kp * (θ_des - θ) + Kd * (ω_des - θ_dot)
        return [θ_dot, (τ_gravity + τ_motor) / I_hinge]
    end

    function ref_simulate_actuated(x0, dt, n_steps, control; m, g, R, I_hinge, Kp, Kd)
        traj = [copy(x0)]
        for _ in 1:n_steps
            xk = traj[end]
            xkp1 = copy(xk)
            for _ in 1:20
                xm = (xkp1 + xk) / 2
                r = xkp1 - xk - dt * ref_actuated_dynamics(xm, control; m=m, g=g, R=R, I_hinge=I_hinge, Kp=Kp, Kd=Kd)
                J = ForwardDiff.jacobian(
                    x -> x - xk - dt * ref_actuated_dynamics((x + xk) / 2, control;
                        m=m, g=g, R=R, I_hinge=I_hinge, Kp=Kp, Kd=Kd), xkp1)
                xkp1 -= J \ r
                norm(r) < 1e-12 && break
            end
            push!(traj, xkp1)
        end
        return traj
    end

    dt = 0.01
    n_steps = 20
    L = 0.5
    m = 5.0
    moi = (1 / 12) * m * L^2
    g = 9.81
    Kp = 20.0
    Kd = 5.0
    θ_des = deg2rad(-45)

    pendulum = ActuatedPendulum(dt;
        bar_length=L, mass=m, moi=moi,
        hinge_position=[0.0, 0.0],
        Kp=Kp, Kd=Kd, max_torque=1000.0,
        n_boundary_nodes=4,
        gravity=[0.0, -g],
    )

    θ_0 = deg2rad(0)
    max_config = pendulum_maximal_from_minimal(pendulum, [θ_0])
    initial_body_state = vcat(max_config, zeros(3))
    state_0 = initialize_solid_state(pendulum, initial_body_state)

    control_trajectory = [[θ_des, 0.0] for _ in 1:n_steps]
    trajectories = simulate_solid_system(pendulum, state_0, n_steps * dt;
        control_trajectory=control_trajectory, verbose=false)

    R = L / 2
    I_hinge = moi + m * R^2
    ref_traj = ref_simulate_actuated([θ_0, 0.0], dt, n_steps, [θ_des, 0.0];
        m=m, g=g, R=R, I_hinge=I_hinge, Kp=Kp, Kd=Kd)

    θ_aquarium = [trajectories[:system_state_traj][i][3] for i in 1:n_steps+1]
    θ_reference = [ref_traj[i][1] for i in 1:n_steps+1]

    @test maximum(abs.(θ_aquarium .- θ_reference)) < 0.05
end

@testitem "Simulation gradient via ForwardDiff vs finite differences" begin
    using Aquarium
    using ForwardDiff
    using FiniteDiff

    dt = 0.01
    n_steps = 5
    final_time = n_steps * dt

    system = Pendulum(dt;
        bar_length=0.5, mass=5.0, moi=(1/12)*5.0*0.25,
        hinge_position=[0.0, 0.0],
        stiffness=5.0, damping=1.0,
        n_boundary_nodes=4,
        gravity=[0.0, -9.81],
    )

    θ_0 = deg2rad(-45)
    mc = pendulum_maximal_from_minimal(system, [θ_0])
    initial_body_state = vcat(mc, zeros(3))
    state_0 = initialize_solid_state(system, initial_body_state)

    function objective(params)
        sys = inject_differentiable_params(system, params)
        traj = simulate_solid_system(sys, state_0, final_time; verbose=false)
        return sum(s -> sum(abs2, s), traj[:system_state_traj])
    end

    p0 = collect_differentiable_params(system)

    grad_ad = ForwardDiff.gradient(objective, p0)
    grad_fd = FiniteDiff.finite_difference_gradient(objective, p0)

    @test length(grad_ad) == length(p0)
    @test all(isfinite, grad_ad)
    @test grad_ad ≈ grad_fd rtol=1e-4
end
