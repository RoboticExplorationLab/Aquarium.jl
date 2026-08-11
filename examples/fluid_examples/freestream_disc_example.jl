include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.CairoMakie
using Colors
using JLD2
using LsqFit

vis_dir = visualization_dir("freestream_disc")

#############################################################################################
## Plot parameters
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define simulation parameters
#############################################################################################

# time step
time_step = 0.001
final_time = 2.5
N = Int(final_time/time_step + 1)

# fluid properties
fluid_density = 1.0 # g/cm³
dynamic_viscosity = 1.0 # g/(cm*s)

# freestream velocity (flow from left to right)
# freestream_velocity_x = 40.0
freestream_velocity_x = 100.0 # <- try this for higher Reynolds number (100)
freestream_velocity_y = 0.0

# fluid grid
length_x = 20.0
length_y = 20.0

num_cells_x = 200
num_cells_y = 200

# boundary conditions - freestream allows fluid to flow in and out
boundary_condition_type = :freestream

# disc properties
disc_density = fluid_density  # neutrally buoyant
disc_diameter = length_x/20
disc_radius = disc_diameter / 2
disc_mass = disc_density * π * disc_radius^2
disc_moi = 0.5 * disc_mass * disc_radius^2
n_boundary_nodes = Int(floor(pi*disc_diameter / (length_x / num_cells_x)))

# bluff body position (centered in y, offset in x to avoid inlet)
bluff_body_x = length_x / 2
bluff_body_y = length_y / 2

# calculate Reynolds number
reynolds_number = (fluid_density * freestream_velocity_x * disc_diameter) / dynamic_viscosity

#############################################################################################
## Create fluid environment
#############################################################################################

fluid_env = Fluid(
    time_step;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = [freestream_velocity_x, freestream_velocity_y],
    grid_size = (num_cells_x, num_cells_y),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
)

#############################################################################################
## Create disc (disc) as a bluff body
#############################################################################################

bluff_body = FreeDisc(time_step;
    radius=disc_radius,
    mass=disc_mass,
    moi=disc_moi,
    n_boundary_nodes=n_boundary_nodes,
    ib_method=:weak_form,
    gravity=[0.0, 0.0],
)

#############################################################################################
## Create AquariumTank with disc as bluff body (stationary)
#############################################################################################

tank = AquariumTank_only_bluff_body(fluid_env, bluff_body)

println("\nAquariumTank created:")
println("  Fluid states: ", tank.n_fluid_states)
println("  disc states: ", tank.n_bluff_body_states)
println("  No-slip constraints: ", tank.n_no_slip_constraints)
println("  Total aquarium states: ", tank.n_states)

#############################################################################################
## Define disc state (stationary at center)
#############################################################################################

# Create stationary disc state
# Configuration: [x, y, θ], Velocity: [vx, vy, ω] (all in body frame)
bluff_body_configuration = [bluff_body_x, bluff_body_y, 0.0]
bluff_body_velocity = [0.0, 0.0, 0.0]
bluff_body_state = vcat(bluff_body_configuration, bluff_body_velocity)

#############################################################################################
## Initialize aquarium state
#############################################################################################

# Initialize fluid with freestream velocity
fluid_initial_velocity_x = freestream_velocity_x * ones(fluid_env.fvm_grid.n_vx)
fluid_initial_velocity_y = freestream_velocity_y * ones(fluid_env.fvm_grid.n_vy)
fluid_initial_velocity = vcat(fluid_initial_velocity_x, fluid_initial_velocity_y)

aquarium_state_0 = initialize_aquarium_state(
    tank,
    fluid_initial_velocity
)

#############################################################################################
## Simulate aquarium dynamics
#############################################################################################

trajectories = simulate_aquarium(
    tank,
    aquarium_state_0,
    final_time,
    bluff_body_state;
    is_midpoint_bluff_body=false,
    pivot_type=:metis,
    scaling_type=:ruiz,
    solver_type=:gmres,
    preconditioner_type=:ilu,
    lazy=false,
    ilu_drop_tolerance=1e-2,
    gmres_tolerance=1e-8,
    newton_tolerance=1e-6,
    dual_regularization=1e-8,
    max_newton_iterations=10,
    gmres_memory=100,
    gmres_max_iterations=1000,
    verbose=true
)

println("\nSimulation complete!")

# Save simulation data
save_file = data_file("freestream_disc_$(floor(Int, reynolds_number))re.jld2")
maybe_jldsave(save_file; tank, trajectories)
SAVE_DATA && println("\nResults saved to: ", save_file)

#############################################################################################
## Load sim results
#############################################################################################

# tank and trajectories are already in memory from the simulation above.

fluid_env = tank.fluid
bluff_body = tank.bluff_body

time_traj = trajectories[:time_traj]
aquarium_state_traj = trajectories[:aquarium_state_traj]
fluid_velocity_traj = trajectories[:fluid_state_traj]
bluff_body_state_traj = trajectories[:bluff_body_state_traj]
bluff_body_state = bluff_body_state_traj[1]

fluid_velocity_traj = [extract_fluid_velocity(tank, aquarium_state_traj[i]) for i in 1:N]
fluid_pressure_traj = [-extract_fluid_dual(tank, aquarium_state_traj[i]) ./ time_step for i in 1:N]

#############################################################################################
## Calculate lift and drag coefficients on bluff body
#############################################################################################

function calculate_bluff_body_no_slip_forces(
    tank::AquariumTank,
    aquarium_state::Vector{Float64},
    bluff_body_state::Vector{Float64};
)

    fluid = tank.fluid
    bluff_body = tank.bluff_body

    fluid_velocity = extract_fluid_velocity(tank, aquarium_state)
    bluff_body_no_slip_dual = extract_bluff_body_no_slip_dual(tank, aquarium_state)

    # Calculate fluid forces on bluff body
    bluff_body_no_slip_impulses =
        calculate_no_slip_constraint_vjp(
            fluid,
            bluff_body,
            fluid_velocity,
            bluff_body_state[bluff_body.configuration_indices],
            bluff_body_state[bluff_body.velocity_indices],
            bluff_body_no_slip_dual;
            is_midpoint_state=false
        )[2][bluff_body.velocity_indices]

    # Extract lift and drag forces
    drag_force = -bluff_body_no_slip_impulses[1]/tank.time_step
    lift_force = -bluff_body_no_slip_impulses[2]/tank.time_step
    torque = -bluff_body_no_slip_impulses[3]/tank.time_step

    return drag_force, lift_force, torque
end

drag_force_traj = [calculate_bluff_body_no_slip_forces(tank, aquarium_state_traj[i], bluff_body_state_traj[i])[1] for i in 1:N]
lift_force_traj = [calculate_bluff_body_no_slip_forces(tank, aquarium_state_traj[i], bluff_body_state_traj[i])[2] for i in 1:N]

# Calculate lift and drag coefficients
drag_coeff_traj = [
    (2 * drag_force_traj[i]) / (fluid_density * freestream_velocity_x^2 * disc_diameter)
    for i in 1:N
]
lift_coeff_traj = [
    (2 * lift_force_traj[i]) / (fluid_density * freestream_velocity_x^2 * disc_diameter)
    for i in 1:N
]

# Plot lift and drag coefficients over time
fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "Time (s)", ylabel = "Coefficient",
    ylim = (-1.0, 2.0),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=false
)
lines!(ax, time_traj, drag_coeff_traj, label="Drag Coefficient", linewidth=2, color=:red)
lines!(ax, time_traj, lift_coeff_traj, label="Lift Coefficient", linewidth=2, color=:blue)
axislegend(ax, backgroundcolor=:transparent, labelcolor=:white, framecolor=:white)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_lift_drag_coefficients_$(floor(Int, reynolds_number))re.png"), fig)

#############################################################################################
## Calculate Strouhal number using curve fitting
#############################################################################################

function calculate_strouhal_number(
    time_traj::Vector{Float64},
    drag_coeff_traj::Vector{Float64},
    freestream_velocity::Float64,
    characteristic_length::Float64;
    fraction::Float64=0.1  # Use last 10% of the trajectory
)
    # Extract the last fraction of the trajectory for analysis
    n_total = length(time_traj)
    n_start = max(1, floor(Int, n_total * (1 - fraction)))

    # Get time and drag coefficient data for the last portion
    time_segment = time_traj[n_start:end]
    drag_segment = drag_coeff_traj[n_start:end]

    # Shift time to start at zero for fitting
    time_fit = time_segment .- time_segment[1]

    # Define sinusoidal model: y(t) = amplitude * sin(2π * frequency * t + phase) + bias
    # Parameterization: p = [amplitude, frequency, phase, bias]
    @. model(t, p) = p[1] * sin(2π * p[2] * t + p[3]) + p[4]

    # Initial parameter guess
    bias_guess = sum(drag_segment) / length(drag_segment)
    amplitude_guess = (maximum(drag_segment) - minimum(drag_segment)) / 2

    # Estimate frequency from zero crossings or simple periodicity
    drag_centered = drag_segment .- bias_guess
    zero_crossings = 0
    for i in 1:(length(drag_centered)-1)
        if drag_centered[i] * drag_centered[i+1] < 0
            zero_crossings += 1
        end
    end
    period_estimate = 2 * (time_fit[end] - time_fit[1]) / max(zero_crossings, 1)
    frequency_guess = 1.0 / period_estimate

    phase_guess = 0.0

    p0 = [amplitude_guess, frequency_guess, phase_guess, bias_guess]

    # Fit the model
    fit = curve_fit(model, time_fit, drag_segment, p0)

    # Extract fitted parameters
    amplitude = abs(fit.param[1])  # Take absolute value for amplitude
    dominant_frequency = abs(fit.param[2])  # Frequency should be positive
    phase = fit.param[3]
    bias = fit.param[4]

    # Calculate Strouhal number: St = f * L / U
    strouhal_number = dominant_frequency * characteristic_length / freestream_velocity

    # Generate fitted curve for visualization
    fitted_curve = model(time_fit, fit.param)

    return strouhal_number, dominant_frequency, amplitude, bias, phase, time_fit, fitted_curve
end

# Calculate frequency, bias, amplitude, and Strouhal number from lift coefficient oscillations using curve fitting
strouhal_number, shedding_frequency, lift_amplitude, lift_bias, _, time_fit, lift_fitted_curve = calculate_strouhal_number(
    time_traj,
    lift_coeff_traj,
    freestream_velocity_x,
    disc_diameter
)

# Calculate frequency, bias, and amplitude from drag coefficient oscillations using curve fitting
_, drag_shedding_frequency, drag_amplitude, drag_bias, _, _, drag_fitted_curve = calculate_strouhal_number(
    time_traj,
    drag_coeff_traj,
    freestream_velocity_x,
    disc_diameter
)

println("\nStrouhal Number Analysis (Curve Fitting):")
println("  Reynolds Number: ", round(reynolds_number, digits=2))
println("  Dominant Frequency: ", round(shedding_frequency, digits=4), " Hz")
println("  Strouhal Number: ", round(strouhal_number, digits=4))
println("  Lift Amplitude: ", round(lift_amplitude, digits=4))
println("  Lift Bias (Mean): ", round(lift_bias, digits=4))
println("  Drag Amplitude: ", round(drag_amplitude, digits=4))
println("  Drag Bias (Mean): ", round(drag_bias, digits=4))

# Plot fitted curve comparison
n_total = length(time_traj)
n_start = max(1, floor(Int, n_total * 0.9))  # Last 10% of data
time_segment = time_traj[n_start:end]
lift_segment = lift_coeff_traj[n_start:end]
drag_segment = drag_coeff_traj[n_start:end]

fig_fit, ax_fit = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    ylim = (-0.75, 2.5),
    xlabel = "Time (s)", ylabel = "Lift Coefficient",
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=false
)
lines!(ax_fit, time_segment, lift_segment, linewidth=2, color=:blue, label="Lift Data")
lines!(ax_fit, time_segment, drag_segment, linewidth=2, color=:green, label="Drag Data")
lines!(ax_fit, time_segment[1] .+ time_fit, lift_fitted_curve, linewidth=2, color=:red,
       linestyle=:dash, label="Lift Fitted (f=$(round(shedding_frequency, digits=3)) Hz)")
lines!(ax_fit, time_segment[1] .+ time_fit, drag_fitted_curve, linewidth=2, color=:orange,
       linestyle=:dash, label="Drag Fitted (f=$(round(drag_shedding_frequency, digits=3)) Hz)")
axislegend(ax_fit, backgroundcolor=:transparent, labelcolor=:white, framecolor=:white)
display(fig_fit)
maybe_save(joinpath(vis_dir, "freestream_disc_curve_fit_$(floor(Int, reynolds_number))re.png"), fig_fit)

#############################################################################################
## Plot streamlines
#############################################################################################

println("\nGenerating visualizations...")

xlim = (length_x*0.25, length_x)
ylim = (length_y*0.25, length_y*0.75)

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = xlim, ylim = ylim,
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_streamlines!(fig, ax,
    fluid_env,
    bluff_body, nothing,
    fluid_velocity_traj[end],
    bluff_body_state_traj[end], [];
    density=50
)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_streamlines_$(floor(Int, reynolds_number))re.png"), fig)

save_path = joinpath(vis_dir, "freestream_disc_streamlines_animation_$(floor(Int, reynolds_number))re.mp4")
animate_if_enabled(animate_streamlines, fig, ax,
    fluid_env,
    bluff_body, nothing,
    time_traj,
    fluid_velocity_traj,
    bluff_body_state_traj, [[]],
    save_path;
    density=100,
    framerate=20,
    timescale=4.0,
)

#############################################################################################
## Plot vorticity field
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = xlim, ylim = ylim,
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_vorticity_field!(fig, ax,
    fluid_env,
    bluff_body, nothing,
    fluid_velocity_traj[end],
    bluff_body_state_traj[end], [];
    density=20,
    min_threshold=-500.0,
    max_threshold=500.0
)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_vorticity_$(floor(Int, reynolds_number))re.png"), fig)

save_path = joinpath(vis_dir, "freestream_disc_vorticity_animation_$(floor(Int, reynolds_number))re.mp4")
animate_if_enabled(animate_vorticity_field, fig, ax,
    fluid_env,
    bluff_body, nothing,
    time_traj,
    fluid_velocity_traj,
    bluff_body_state_traj, [[]],
    save_path;
    density=10,
    min_threshold=-500.0,
    max_threshold=500.0,
    framerate=20,
    timescale=4.0,
)

#############################################################################################
## Plot pressure field
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = xlim, ylim = ylim,
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_pressure_field!(fig, ax,
    fluid_env,
    bluff_body, nothing,
    fluid_pressure_traj[end],
    bluff_body_state, [];
)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_pressure_animation_$(floor(Int, reynolds_number))re.png"), fig)

save_path = joinpath(vis_dir, "freestream_disc_pressure_animation_$(floor(Int, reynolds_number))re.mp4")
animate_if_enabled(animate_pressure_field, fig, ax,
    fluid_env,
    bluff_body, nothing,
    time_traj,
    fluid_pressure_traj,
    bluff_body_state_traj, [[]],
    save_path;
    framerate=20,
    timescale=4.0,
)

#############################################################################################
## Plot energy over time
#############################################################################################

# calculate total energy
fluid_energy_traj = [calculate_total_energy(fluid_env, fluid_velocity_traj[i]) for i in 1:N]

# plot energy
fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "Time (s)", ylabel = "Total Energy (erg)",
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=false
)
lines!(ax, time_traj, fluid_energy_traj, label="Fluid Energy", linewidth=2)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_energy_$(floor(Int, reynolds_number))re.png"), fig)

#############################################################################################
## Plot fluid pressure norms over time
#############################################################################################

# calculate pressure norms
fluid_pressure_norm_traj = [norm(fluid_pressure_traj[i]) for i in 1:N]

# plot pressure norms
fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "Time (s)", ylabel = "Pressure Norm",
    resolution=resolution,
    ylim=(0.0, norm(fluid_pressure_norm_traj[end])*1.1),
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=false
)
lines!(ax, time_traj, fluid_pressure_norm_traj, label="Pressure Norm", linewidth=2)
display(fig)
maybe_save(joinpath(vis_dir, "freestream_disc_pressure_norm_$(floor(Int, reynolds_number))re.png"), fig)

println("\n" * "="^80)
println("Simulation and visualization complete!")
SAVE_FIGURES && println("Results saved to: ", vis_dir)
println("="^80)
