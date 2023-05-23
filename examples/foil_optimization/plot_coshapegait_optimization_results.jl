import Pkg
Pkg.activate(joinpath(@__DIR__,"..", ".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Makie
using Makie.GeometryBasics
using CairoMakie
using Pardiso
using JLD2
using Test
using ProgressMeter
using Interpolations
using BenchmarkTools
using PGFPlotsX
using Colors

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

include("foil_utils.jl")

function plot_boundary_dashed!(model::DiamondFoil, x::AbstractVector; color=:black, linewidth=5)

    x_b = boundary_state(model, x)
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:2*model.nodes]
    append!(x, x[1])
    append!(y, y[1])

    lines!(Point2f[(x[i], y[i]) for i in eachindex(x)], color=color,
        linestyle = :dash, linewidth = linewidth
    )
    
end

##############################
## load data
##############################

data = load(joinpath(DATADIR, "diamondfoil_example_coshapegait_optimization_history.jld2"))

data_iter0 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data039406355354754474065113716110150820015707963267948966.jld2"))
data_iter1 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data02562100734711923504960249310267874604989214475798240415707963267948966.jld2"))
data_iter2 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data0104719755119659770678943973856207100072967546400610211559811552315187.jld2"))
data_iter3 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977064599838595430202383176968786936415559575292318586.jld2"))
data_iter4 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data0104719755119659770976705741652262202715603365503878615298250173556194.jld2"))
data_iter5 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220386518836301494615045257988223608.jld2"))
data_iter6 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220553048466649106512406708013556937.jld2"))
data_iter7 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data0104719755119659770976705741652262204682306463208113513035127582318369.jld2"))
data_iter8 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220473392689440893912959359708947635.jld2"))
data_iter9 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220473430159854020912944801458426298.jld2"))
data_iter10 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220474298570254960712951787416920755.jld2"))
data_iter11 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220474785479368885612956367727919897.jld2"))
data_iter12 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data01047197551196597709767057416522622047479725301610241295773494341454.jld2"))
data_iter13 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data01047197551196597709767057416522622047477813885597087129579253668498.jld2"))
data_iter14 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data0104719755119659770976705741652262204747844756437340512957971419299101.jld2"))

dt = data["dt"]
t0 = data["t0"]
tf = data["tf"]

fluid = data["fluid"]
normalized_fluid = data["normalized_fluid"]

reward_vec = -fluid.ref_u^2 .* fluid.ref_L .* [data_iter0["net_fx"], data_iter1["net_fx"], data_iter2["net_fx"], data_iter3["net_fx"],
    data_iter4["net_fx"], data_iter5["net_fx"], data_iter6["net_fx"], data_iter7["net_fx"],
    data_iter8["net_fx"], data_iter9["net_fx"], data_iter10["net_fx"], data_iter11["net_fx"],
    data_iter12["net_fx"], data_iter13["net_fx"], data_iter14["net_fx"]
]

α = data["α"]
ρ_b = data["ρ_b"]
c = data["c"]
leading_ratio = data["leading_ratio"]
cm_x = data["cm_x"]
cm_y = data["cm_y"]
nodes = data["nodes"]

θ_initial = data["θ_initial"]
θ_initial_normalized = data["θ_initial_normalized"]
net_thrust_initial = data["net_thrust_initial"]

θ_optimal = data["θ_optimal"]
θ_optimal_normalized = data["θ_optimal_normalized"]
net_thrust_optimal = data["net_thrust_optimal"]

θ_iter1 = data["θ_iter1"]
θ_iter1_normalized = data_iter0["θ_normalized"]
net_thrust_iter1 = data_iter0["net_fx"]

θ_iter5 = data["θ_iter5"]
θ_iter5_normalized = data["θ_iter5_normalized"]
net_thrust_iter5 = data["net_thrust_iter5"]

a_initial = θ_initial[1]
A_initial = θ_initial[2]
B_initial = θ_initial[3]
ϕ_pitch_initial = θ_initial[4]
Bd_initial = round(Int, rad2deg(B_initial))
ϕd_pitch_initial = round(Int, rad2deg(ϕ_pitch_initial))

a_iter1 = θ_iter1[1]
A_iter1 = θ_iter1[2]
B_iter1 = θ_iter1[3]
ϕ_pitch_iter1 = θ_iter1[4]
Bd_iter1 = round(Int, rad2deg(B_iter1))
ϕd_pitch_iter1 = round(Int, rad2deg(ϕ_pitch_iter1))

a_iter5 = θ_iter5[1]
A_iter5 = θ_iter5[2]
B_iter5 = θ_iter5[3]
ϕ_pitch_iter5 = θ_iter5[4]
Bd_iter5 = round(Int, rad2deg(B_iter5))
ϕd_pitch_iter5 = round(Int, rad2deg(ϕ_pitch_iter5))

a_optimal = θ_optimal[1]
A_optimal = θ_optimal[2]
B_optimal = θ_optimal[3]
ϕ_pitch_optimal = θ_optimal[4]
Bd_optimal = round(Int, rad2deg(B_optimal))
ϕd_pitch_optimal = round(Int, rad2deg(ϕ_pitch_optimal))

##############################
## plotting constants
##############################

logocolors = Colors.JULIA_LOGO_COLORS

color_initial = RGBA(logocolors[2].r, logocolors[2].g, logocolors[2].b, 0.25)
color_iter1 = RGBA(logocolors[4].r, logocolors[4].g, logocolors[4].b, 0.5)
color_iter5 = RGBA(colorant"orange".r, colorant"orange".g, colorant"orange".b, 0.75)
color_optimal = colorant"black";

color_initial_solid = logocolors[2];
color_iter1_solid = logocolors[4];
color_iter5_solid = colorant"orange";
color_optimal_solid = colorant"black";

density = 150.0
levels = 100
level_perc = 0.2
x_lim = [0.5, 2.75]
y_lim = [0.75, 2.25]
colormap_streamlines = :blues
colormap_vorticity = :berlin
background_color = :black
obj_color = :black
plot_fontsize=25
animation_fontsize=40
framerate=20
timescale=10.0
lineopts = @pgf {no_marks, "very thick"}

########################################
## define additional variables
########################################

t0 = 0.0
tf = 1.0
T = 0:dt:tf

ref_L = fluid.ref_L
ref_u = fluid.ref_u

###############################################
## Create boundaries
###############################################

boundary_initial = Aquarium.DiamondFoil(ρ_b, a_initial, c; leading_ratio = leading_ratio, nodes=nodes)
boundary_iter1 = Aquarium.DiamondFoil(ρ_b, a_iter1, c; leading_ratio = leading_ratio, nodes=nodes)
boundary_iter5 = Aquarium.DiamondFoil(ρ_b, a_iter5, c; leading_ratio = leading_ratio, nodes=nodes)
boundary_optimal = Aquarium.DiamondFoil(ρ_b, a_optimal, c; leading_ratio = leading_ratio, nodes=nodes)

###############################################
## Create oscillating motion
###############################################

X_initial = oscillating_motion_x(boundary_initial, [cm_x, cm_y, 0.0], θ_initial[2:end], α, T)
X_initial_normalized = [Aquarium.normalize(boundary_initial, X_initial[i], ref_L, ref_u) for i in eachindex(X_initial)]

X_iter1 = oscillating_motion_x(boundary_iter1, [cm_x, cm_y, 0.0], θ_iter1[2:end], α, T)
X_iter1_normalized = [Aquarium.normalize(boundary_iter1, X_iter1[i], ref_L, ref_u) for i in eachindex(X_iter1)]

X_iter5 = oscillating_motion_x(boundary_iter5, [cm_x, cm_y, 0.0], θ_iter5[2:end], α, T)
X_iter5_normalized = [Aquarium.normalize(boundary_iter5, X_iter5[i], ref_L, ref_u) for i in eachindex(X_iter5)]

X_optimal = oscillating_motion_x(boundary_optimal, [cm_x, cm_y, 0.0], θ_optimal[2:end], α, T)
X_optimal_normalized = [Aquarium.normalize(boundary_optimal, X_optimal[i], ref_L, ref_u) for i in eachindex(X_optimal)]

##############################
## Animate boundary motions
##############################

mkpath(joinpath(VISDIR, "diamondfoil", "coshapegait_optimization"))

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization", "oscillating_diamondfoil_coshapegait_optimization_initial.mp4")
Aquarium.animate_boundary(boundary_initial, anime_file, T, X_initial; x_lim=x_lim, y_lim=y_lim,
    color=color_initial_solid, lengthscale=1.5, framerate=framerate, timescale=timescale, show_vel=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization", "oscillating_diamondfoil_coshapegait_optimization_iter1.mp4")
Aquarium.animate_boundary(boundary_iter1, anime_file, T, X_iter1; x_lim=x_lim, y_lim=y_lim,
    color=color_iter1_solid, lengthscale=1.5, framerate=framerate, timescale=timescale, show_vel=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization", "oscillating_diamondfoil_coshapegait_optimization_iter5.mp4")
Aquarium.animate_boundary(boundary_iter5, anime_file, T, X_iter5; x_lim=x_lim, y_lim=y_lim,
    color=color_iter5_solid, lengthscale=1.5, framerate=framerate, timescale=timescale, show_vel=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization", "oscillating_diamondfoil_coshapegait_optimization_optimal.mp4")
Aquarium.animate_boundary(boundary_optimal, anime_file, T, X_optimal; x_lim=x_lim, y_lim=y_lim,
    color=color_optimal_solid, lengthscale=1.5, framerate=framerate, timescale=timescale, show_vel=false
)

###############################################
## normalize
###############################################

α_normalized = α * (ref_L / ref_u)

normalized_boundary_initial = Aquarium.normalize(boundary_initial, ref_L)
normalized_boundary_iter1 = Aquarium.normalize(boundary_iter1, ref_L)
normalized_boundary_iter5 = Aquarium.normalize(boundary_iter5, ref_L)
normalized_boundary_optimal = Aquarium.normalize(boundary_optimal, ref_L)

##############################
## simulate
##############################

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary_initial)
t_hist_initial, x_hist_initial, u_hist_initial, p_hist_initial, f_b_hist_initial = simulate(
    normalized_fluid, normalized_boundary_initial, uk, pk, fk_b, X_initial_normalized;
    λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=false
)

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary_iter1)
t_hist_iter1, x_hist_iter1, u_hist_iter1, p_hist_iter1, f_b_hist_iter1 = simulate(
    normalized_fluid, normalized_boundary_iter1, uk, pk, fk_b, X_iter1_normalized;
    λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=false
)

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary_iter5)
t_hist_iter5, x_hist_iter5, u_hist_iter5, p_hist_iter5, f_b_hist_iter5 = simulate(
    normalized_fluid, normalized_boundary_iter5, uk, pk, fk_b, X_iter5_normalized;
    λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=false
)

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary_optimal)
t_hist_optimal, x_hist_optimal, u_hist_optimal, p_hist_optimal, f_b_hist_optimal = simulate(
    normalized_fluid, normalized_boundary_optimal, uk, pk, fk_b, X_optimal_normalized;
    λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=false
)

##############################
## save data
##############################

save_file = joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_initial.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary_initial, ref_L, ref_u, t_hist_initial, x_hist_initial, u_hist_initial, p_hist_initial, f_b_hist_initial)

save_file = joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_iter1.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary_iter1, ref_L, ref_u, t_hist_iter1, x_hist_iter1, u_hist_iter1, p_hist_iter1, f_b_hist_iter1)

save_file = joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_iter5.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary_iter5, ref_L, ref_u, t_hist_iter5, x_hist_iter5, u_hist_iter5, p_hist_iter5, f_b_hist_iter5)

save_file = joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_optimal.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary_optimal, ref_L, ref_u, t_hist_optimal, x_hist_optimal, u_hist_optimal, p_hist_optimal, f_b_hist_optimal)

##############################
## load data
##############################

data_initial = load(joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_initial.jld2"))
t_hist_initial = data_initial["t_hist_initial"]
x_hist_initial = data_initial["x_hist_initial"]
u_hist_initial = data_initial["u_hist_initial"]
p_hist_initial = data_initial["p_hist_initial"]
f_b_hist_initial = data_initial["f_b_hist_initial"]
x_initial = x_hist_initial[end]
uk_initial = u_hist_initial[end]

data_iter1 = load(joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_iter1.jld2"))
t_hist_iter1 = data_iter1["t_hist_iter1"]
x_hist_iter1 = data_iter1["x_hist_iter1"]
u_hist_iter1 = data_iter1["u_hist_iter1"]
p_hist_iter1 = data_iter1["p_hist_iter1"]
f_b_hist_iter1 = data_iter1["f_b_hist_iter1"]
x_iter1 = x_hist_iter1[end]
uk_iter1 = u_hist_iter1[end]

data_iter5 = load(joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_iter5.jld2"))
t_hist_iter5 = data_iter5["t_hist_iter5"]
x_hist_iter5 = data_iter5["x_hist_iter5"]
u_hist_iter5 = data_iter5["u_hist_iter5"]
p_hist_iter5 = data_iter5["p_hist_iter5"]
f_b_hist_iter5 = data_iter5["f_b_hist_iter5"]
x_iter5 = x_hist_iter5[end]
uk_iter5 = u_hist_iter5[end]

data_optimal = load(joinpath(DATADIR, "oscillating_diamondfoil_coshapegait_optimization_optimal.jld2"))
t_hist_optimal = data_optimal["t_hist_optimal"]
x_hist_optimal = data_optimal["x_hist_optimal"]
u_hist_optimal = data_optimal["u_hist_optimal"]
p_hist_optimal = data_optimal["p_hist_optimal"]
f_b_hist_optimal = data_optimal["f_b_hist_optimal"]
x_optimal = x_hist_optimal[end]
uk_optimal = u_hist_optimal[end]

##############################
## calculate forces
##############################

net_fx_initial = ones(length(f_b_hist_initial))

for i in eachindex(f_b_hist_initial)
    fi_b_initial = f_b_hist_initial[i]
    normalized_net_fx_initial = fi_b_initial[1:normalized_boundary_initial.nodes]    
    net_fx_initial[i] = -sum(normalized_net_fx_initial) .* fluid.ref_u^2 .* fluid.ref_L
end

net_fx_iter1 = ones(length(f_b_hist_iter1))

for i in eachindex(f_b_hist_iter1)
    fi_b_iter1 = f_b_hist_iter1[i]
    normalized_net_fx_iter1 = fi_b_iter1[1:normalized_boundary_iter1.nodes]    
    net_fx_iter1[i] = -sum(normalized_net_fx_iter1) .* fluid.ref_u^2 .* fluid.ref_L
end

net_fx_iter5 = ones(length(f_b_hist_iter5))

for i in eachindex(f_b_hist_iter5)
    fi_b_iter5 = f_b_hist_iter5[i]
    normalized_net_fx_iter5 = fi_b_iter5[1:normalized_boundary_iter5.nodes]    
    net_fx_iter5[i] = -sum(normalized_net_fx_iter5) .* fluid.ref_u^2 .* fluid.ref_L
end

net_fx_optimal = ones(length(f_b_hist_optimal))

for i in eachindex(f_b_hist_optimal)
    fi_b_optimal = f_b_hist_optimal[i]
    normalized_net_fx_optimal = fi_b_optimal[1:normalized_boundary_optimal.nodes]    
    net_fx_optimal[i] = -sum(normalized_net_fx_optimal) .* fluid.ref_u^2 .* fluid.ref_L
end

##############################
## Plot Forces
##############################

set_theme!(font = "Times New Roman", fontsize=plot_fontsize)

n = minimum([length(t_hist_initial), length(t_hist_iter1), length(t_hist_iter5), length(t_hist_optimal)])
n = n÷2
t_hist_plot = dt.*(0:n-1)

fig, ax, sp = lines(t_hist_plot[5:n], net_fx_initial[5:n],
    color=color_initial_solid, axis = (xlabel = "Time (s)", ylabel = "Thrust (N/m)"),
    label = "Initial"
)
lines!(t_hist_plot[5:n], net_fx_iter1[5:n], color=color_iter1_solid, label = "Iteration 1")
lines!(t_hist_plot[5:n], net_fx_iter5[5:n], color=color_iter5_solid, label = "Iteration 5")
lines!(t_hist_plot[5:n], net_fx_optimal[5:n], color=color_optimal_solid, label = "Converged (Iteration 14)")

axislegend(ax, position = :lt)
xlims!(ax, [t_hist_plot[5], t_hist_plot[n]])
ylims!(ax, [-25, 30])

display(fig)

mkpath(joinpath(VISDIR, "diamondfoil", "coshapegait_optimization"))
save_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "thrust_history.png")
save(save_file, fig, pt_per_unit = 2)

thrust_time_history = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Thrust (N/m)",
        legend_pos = "north west",
        legend_columns=2,
        legend_cell_align="left",
        xmin = t_hist_plot[5],
        xmax = t_hist_plot[n],
        ymin = -25,
        ymax = 32,
        
    },
    PlotInc({lineopts..., color=color_initial_solid}, Coordinates(t_hist_plot[5:n], net_fx_initial[5:n])),
    PlotInc({lineopts..., color=color_iter1_solid}, Coordinates(t_hist_plot[5:n], net_fx_iter1[5:n])),
    PlotInc({lineopts..., color=color_iter5_solid}, Coordinates(t_hist_plot[5:n], net_fx_iter5[5:n])),
    PlotInc({lineopts..., color=color_optimal_solid}, Coordinates(t_hist_plot[5:n], net_fx_optimal[5:n])),

    PGFPlotsX.Legend(["Initial", "Iteration 1", "Iteration 5", "Converged (Iteration 14)"])
)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "BFGS_thrust_time_history.tikz");
pgfsave(filename, thrust_time_history, include_preamble=false);

##############################
## Plot Reward
##############################

set_theme!(font = "Times New Roman", fontsize=plot_fontsize)

bfgs_iter = 0:14

fig, ax, sp = lines(bfgs_iter, reward_vec,
    color=:blue, axis = (xlabel = "BFGS Iteration", ylabel = "Thrust (N/m)")
)
ylims!(ax, [-16, 7])

display(fig)

bfgs_iteration_history = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "BFGS Iteration",
        ylabel = "Thrust (N/m)",
        xmin = -1,
        xmax = 15,
        ymin = -16,
        ymax = 7,
        
    },
    PlotInc({lineopts..., color="blue"}, Coordinates(bfgs_iter, reward_vec))
)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "BFGS_reward_iteration_history.tikz")
pgfsave(filename, bfgs_iteration_history, include_preamble=false)

##############################
## plot shape morphology
##############################

set_theme!(font = "Times New Roman", fontsize=plot_fontsize, Axis = (
        background_color = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))

fig, ax = plot_boundary(boundary_optimal, Aquarium.unnormalize(
    boundary_optimal, x_optimal, ref_L, ref_u))

plot_boundary!(boundary_initial, Aquarium.unnormalize(
    boundary_initial, x_initial, ref_L, ref_u);
    color=color_initial
)
plot_boundary!(boundary_iter1, Aquarium.unnormalize(
    boundary_initial, x_iter1, ref_L, ref_u);
    color=color_iter1
)
plot_boundary!(boundary_iter5, Aquarium.unnormalize(
    boundary_initial, x_iter5, ref_L, ref_u);
    color=color_iter5
)
plot_boundary!(boundary_optimal, Aquarium.unnormalize(
    boundary_optimal, x_optimal, ref_L, ref_u)
)
# plot_boundary_dashed!(boundary_initial, Aquarium.unnormalize(
#     boundary_initial, x_initial, ref_L, ref_u);
#     color=logocolors[2], linewidth=3
# )
# plot_boundary_dashed!(boundary_iter1, Aquarium.unnormalize(
#     boundary_iter1, x_iter1, ref_L, ref_u);
#     color=logocolors[4], linewidth=3
# )
# plot_boundary_dashed!(boundary_iter5, Aquarium.unnormalize(
#     boundary_iter5, x_iter5, ref_L, ref_u);
#     color="orange", linewidth=3
# )
plot_boundary!(boundary_optimal, Aquarium.unnormalize(
    boundary_optimal, x_optimal, ref_L, ref_u)
)
display(fig)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "shape_gait_morphology.png")
save(filename, fig, px_per_unit = 10)

uk = uk_optimal
x = x_optimal
normalized_boundary = normalized_boundary_optimal

uk_avg = average(normalized_fluid, uk)
scene = plot_vorticity(normalized_fluid, normalized_boundary_optimal, x, uk_avg;
    levels=levels, x_lim=[0.75, 2.5], y_lim=[0.9, 2.1], level_perc=level_perc,
    colormap=colormap_vorticity, obj_color = color_optimal, fontsize=plot_fontsize)
display(scene)

filename = joinpath(VISDIR, "paper", "optimal_shape_gait_vorticity.png")
save(filename, scene, px_per_unit = 10)

##############################
## plot simulation
##############################

# uk = uk_initial
# x = x_initial
# normalized_boundary = normalized_boundary_initial
# plot_color = color_initial_solid

# uk = uk_iter1
# x = x_iter1
# normalized_boundary = normalized_boundary_iter1
# plot_color = color_iter1_solid

# uk = uk_iter5
# x = x_iter5
# normalized_boundary = normalized_boundary_iter5
# plot_color = color_iter5_solid

uk = uk_optimal
x = x_optimal
normalized_boundary = normalized_boundary_optimal
plot_color =:white

uk_avg = average(normalized_fluid, uk)
scene = plot_streamlines(normalized_fluid, normalized_boundary_optimal, x, uk_avg;
    density=density, x_lim=x_lim, y_lim=y_lim, background_color=background_color,
    colormap=colormap_streamlines, obj_color=plot_color, fontsize=plot_fontsize)
display(scene)

scene = plot_vorticity(normalized_fluid, normalized_boundary_optimal, x, uk_avg;
    levels=levels, x_lim=x_lim, y_lim=y_lim, level_perc=0.2, colormap=colormap_vorticity,
    obj_color=plot_color, fontsize=plot_fontsize)
display(scene)

##############################
## create animation
##############################

mkpath(joinpath(VISDIR, "diamondfoil", "coshapegait_optimization"))

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_initial_streamlines.mp4")
animate_streamlines(normalized_fluid, normalized_boundary_initial, t_hist_initial, x_hist_initial, u_hist_initial, anime_file;
    density=density, x_lim=x_lim, y_lim=y_lim, background_color=background_color, obj_color = color_initial_solid,
    framerate=framerate, timescale=timescale, display_live=false, fontsize=animation_fontsize, resolution=(1600, 1200)
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_initial_vorticity.mp4"
)

animate_vorticity(normalized_fluid, normalized_boundary_initial, t_hist_initial, x_hist_initial, u_hist_initial, anime_file;
    levels=levels, x_lim=x_lim, y_lim=y_lim, level_perc=level_perc, colormap=colormap_vorticity, fontsize=animation_fontsize,
    obj_color = color_initial_solid, framerate=framerate, timescale=timescale, resolution=(1600, 1200), display_live=false
)

####################################################################

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_iter1_streamlines.mp4"
)

animate_streamlines(normalized_fluid, normalized_boundary_iter1, t_hist_iter1, x_hist_iter1, u_hist_iter1, anime_file;
    density=density, x_lim=x_lim, y_lim=y_lim, background_color=background_color, obj_color = color_iter1_solid,
    framerate=framerate, timescale=timescale, fontsize=animation_fontsize, resolution=(1600, 1200), display_live=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_iter1_vorticity.mp4"
)

animate_vorticity(normalized_fluid, normalized_boundary_iter1, t_hist_iter1, x_hist_iter1, u_hist_iter1, anime_file;
    levels=levels, x_lim=x_lim, y_lim=y_lim, level_perc=level_perc, colormap=colormap_vorticity, fontsize=animation_fontsize,
    obj_color = color_iter1_solid, framerate=framerate, timescale=timescale, resolution=(1600, 1200), display_live=false
)

####################################################################

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_iter5_streamlines.mp4"
)

animate_streamlines(normalized_fluid, normalized_boundary_iter5, t_hist_iter5, x_hist_iter5, u_hist_iter5, anime_file;
    density=density, x_lim=x_lim, y_lim=y_lim, background_color=background_color, obj_color = color_iter5_solid,
    framerate=framerate, timescale=timescale, fontsize=animation_fontsize, resolution=(1600, 1200), display_live=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_iter5_vorticity.mp4"
)

animate_vorticity(normalized_fluid, normalized_boundary_iter5, t_hist_iter5, x_hist_iter5, u_hist_iter5, anime_file;
    levels=levels, x_lim=x_lim, y_lim=y_lim, level_perc=level_perc, colormap=colormap_vorticity, fontsize=animation_fontsize,
    obj_color = color_iter5_solid, framerate=framerate, timescale=timescale, resolution=(1600, 1200), display_live=false
)

####################################################################

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_optimal_streamlines.mp4"
)

animate_streamlines(normalized_fluid, normalized_boundary_optimal, t_hist_optimal, x_hist_optimal, u_hist_optimal, anime_file;
    density=density, x_lim=x_lim, y_lim=y_lim, background_color=background_color, obj_color = color_optimal_solid,
    framerate=framerate, timescale=timescale, fontsize=animation_fontsize, resolution=(1600, 1200), display_live=false
)

anime_file = joinpath(VISDIR, "diamondfoil", "coshapegait_optimization",
    "oscillating_diamondfoil_coshapegait_optimization_optimal_vorticity.mp4"
)

animate_vorticity(normalized_fluid, normalized_boundary_optimal, t_hist_optimal, x_hist_optimal, u_hist_optimal, anime_file;
    levels=levels, x_lim=x_lim, y_lim=y_lim, level_perc=level_perc, colormap=colormap_vorticity, fontsize=animation_fontsize,
    obj_color = color_optimal_solid, framerate=framerate, timescale=timescale, resolution=(1600, 1200), display_live=false
)