include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Pardiso
using Colors
using JLD2
using Test
using PGFPlotsX

vis_dir = visualization_dir("pendulum")

#############################################################################################
## Helper functions for simulating pendulum with implicit-midpoint time integrator
#############################################################################################

# kinetic energy in terms of just angular velocity
function test_kinetic_energy(ω; moi=1.0, L=1.0)
    0.5*moi*ω^2
end

# potential energy in terms of just angle
function test_potential_energy(θ; m, L, g)
    # Account for Aquarium convention: θ measured from vertical (hanging down = 0)
    # PE = 0 at θ=0 (hanging down), increases as pendulum rises
    m*g*L*sin(θ)
end

function test_total_energy(θ, θ_dot; m, moi, L, g, k=0.0)
    test_kinetic_energy(θ_dot; moi=moi, L=L) + test_potential_energy(θ; m=m, L=L, g=g) + (1/2)*k*θ^2
end

function test_pendulum_dynamics(x; m=1.0, g=9.81, L=1.0, moi=0.1, k=0.0, c=0.0, θ_eq=-deg2rad(90))

    θ = x[1]
    θ_dot = x[2]

    # Account for Aquarium convention: θ measured from vertical (hanging down = 0)
    # Standard pendulum dynamics use horizontal reference, so add 90 deg offset
    # Gravity torque
    τ_gravity = -(m*g*L)*sin(θ + deg2rad(90))
    
    # Spring torque: τ_spring = -k * (θ - θ_eq)
    τ_spring = -k * (θ - θ_eq)
    
    # Damping torque: τ_damping = -c * θ_dot
    τ_damping = -c * θ_dot
    
    # Total angular acceleration
    θ_ddot = (τ_gravity + τ_spring + τ_damping) / moi

    return [θ_dot, θ_ddot]
end

function implicit_midpoint_residual(xkp1, xk, time_step; m=1.0, g=9.81, L=1.0, moi=0.1, k=0.0, c=0.0, θ_eq=-deg2rad(90))

    xm = (xkp1 + xk)/2

    return xkp1 - xk - time_step.*test_pendulum_dynamics(xm; m=m, g=g, L=L, moi=moi, k=k, c=c, θ_eq=θ_eq)

end

function implicit_midpoint_jacobian(xkp1, xk, time_step; m=1.0, g=9.81, L=1.0, moi=0.1, k=0.0, c=0.0, θ_eq=-deg2rad(90))

    return ForwardDiff.jacobian(x -> implicit_midpoint_residual(x, xk, time_step; g=g, L=L, m=m, moi=moi, k=k, c=c, θ_eq=θ_eq), xkp1)

end

function test_simulate_pendulum(x0, time_step, final_time; m=1.0, g=9.81, L=1.0, moi=0.1, k=0.0, c=0.0, θ_eq=-deg2rad(90))

    N = Int(final_time/time_step) + 1

    # Initialize trajectory
    time_traj = Vector(LinRange(0, final_time, N))
    test_state_traj = [copy(x0) for k = 1:N]

    for i in 1:N-1

        x_kp1 = copy(test_state_traj[i])
        x_k = test_state_traj[i]

        residual = implicit_midpoint_residual(x_kp1, x_k, time_step; m=m, g=g, L=L, moi=moi, k=k, c=c, θ_eq=θ_eq)
        max_iter = 0

        # Newton's method
        while maximum(abs.(residual)) > 1e-6 && max_iter < 10

            # compute residual
            ∂residual_∂x_kp1 = implicit_midpoint_jacobian(x_kp1, x_k, time_step; m=m, g=g, L=L, moi=moi, k=k, c=c, θ_eq=θ_eq)

            x_kp1 .-= ∂residual_∂x_kp1 \ residual

            residual = implicit_midpoint_residual(x_kp1, x_k, time_step; m=m, g=g, L=L, moi=moi, k=k, c=c, θ_eq=θ_eq)

            max_iter += 1

        end

        test_state_traj[i+1] = x_kp1

    end

    return time_traj, test_state_traj

end

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define pendulum
#############################################################################################

# time properties
time_step = 0.01
final_time = 10.0
N = Int(final_time/time_step) + 1

# pendulum properties
pendulum_length = 0.5
mass = 5.0
moi = (1/12) * mass * pendulum_length^2
gravity_constant = 9.81

# spring and damping properties (equilibrium at θ=0, hanging straight down)
spring_stiffness = 5.0      # N⋅m/rad
damping_coefficient = 1.0   # N⋅m⋅s/rad

# boundary properties
n_boundary_nodes = 5

# hinge position
hinge_position = [0.0, 0.0]

pendulum = Pendulum(time_step;
    bar_length = pendulum_length,
    mass = mass,
    moi = moi,
    hinge_position = hinge_position,
    stiffness = spring_stiffness,
    damping = damping_coefficient,
    n_boundary_nodes = n_boundary_nodes,
    gravity = [0.0, -gravity_constant],
)

#############################################################################################
## Define initial pendulum state
#############################################################################################

# pendulum state
θ_0 = deg2rad(-45)
ω_0 = 0.0

max_config = pendulum_maximal_from_minimal(pendulum, [θ_0])
body = pendulum.bodies[1]
center_minus_hinge = max_config[1:2] .- hinge_position
max_velocity = [-ω_0 * center_minus_hinge[2], ω_0 * center_minus_hinge[1], ω_0]

initial_body_state = vcat(max_config, max_velocity)
full_system_state_0 = initialize_solid_state(pendulum, initial_body_state)

# test hinge constraint satisfied
@test calculate_system_constraint_residual(
    pendulum,
    full_system_state_0[pendulum.configuration_indices],
) ≈ zeros(pendulum.n_constraints) atol=1e-12

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

trajectories = simulate_solid_system(pendulum,
    full_system_state_0,
    final_time; verbose=false
)
time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
system_state_traj = trajectories[:system_state_traj]

#############################################################################################
## Simulate with implicit midpoint with regular pendulum dynamics
#############################################################################################

test_moi = (1/3)*mass*pendulum_length^2 # mass moment of inertia for rod about end
x0 = [θ_0, 0.0]  # Use same angle convention as Aquarium
_, test_state_traj = test_simulate_pendulum(x0,
    time_step, final_time;
    m=mass,
    g=gravity_constant,
    L=pendulum_length/2,
    moi=test_moi,
    k=spring_stiffness,
    c=damping_coefficient,
    θ_eq=0.0
)

#############################################################################################
## Plot trajectories for comparison
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel = "Time (s)", ylabel = "State",
    xlim=(0, final_time), ylim=(-5, 6),
    use_data_aspect=false
)
lines!(ax, time_traj, [x[3] for x in configuration_traj], label="Aquarium θ")
lines!(ax, time_traj, [x[3] for x in velocity_traj], label="Aquarium ω")
lines!(ax, time_traj, [x[1] for x in test_state_traj], label="Implicit Midpoint θ")
lines!(ax, time_traj, [x[2] for x in test_state_traj], label="Implicit Midpoint ω")
axislegend(ax,
    backgroundcolor=:transparent,
    labelcolor=:white,
    framecolor=:white,
    orientation=:horizontal,
    nbanks=1
)
display(fig)

#############################################################################################
## Plot energies for comparison
#############################################################################################

energy_traj = [calculate_total_energy(pendulum, state) for state in system_state_traj] .-
    calculate_total_energy(pendulum, system_state_traj[1])
test_energy_traj = [test_total_energy(x[1], x[2];
    m=mass, moi=test_moi,
    L=pendulum_length/2, g=gravity_constant, k=spring_stiffness) for x in test_state_traj] .-
    test_total_energy(x0[1], x0[2];
    m=mass, moi=test_moi,
    L=pendulum_length/2, g=gravity_constant, k=spring_stiffness)

fig2, ax2 = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel = "Time (s)", ylabel = "Δ Energy",
    xlim=(0, final_time),
    ylim=(-20.0, 1.0),
    use_data_aspect=false
)
lines!(ax2, time_traj, energy_traj, label="Aquarium ΔE")
lines!(ax2, time_traj, test_energy_traj, label="Implicit Midpoint ΔE")
axislegend(ax2,
    backgroundcolor=:transparent,
    labelcolor=:white,
    framecolor=:white,
    orientation=:horizontal,
    nbanks=1
)
display(fig2)