import Pkg
Pkg.activate(joinpath(@__DIR__,"..", ".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Makie
using Makie.GeometryBasics
using CairoMakie
using DelimitedFiles
using PGFPlotsX
using Colors

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

include("tail_utils.jl")

##############################
## plotting constants
##############################

x_lim=[0.1664, 0.3]
y_lim=[0.26, 0.34]
colormap = :blues
obj_color = :white
background_color = :black
fontsize=40
framerate=30
timescale=2.0
lineopts = @pgf {no_marks, "very thick"}
resolution = (2000, 1200)
logocolors = Colors.JULIA_LOGO_COLORS

##############################
## import tail data
##############################

# data = load(joinpath(DATADIR, "eth_fish_3hz_force.jld2"))
# F_emp = data["F"]
# T_emp = 0:(1/60):(1/60 * (length(F_emp)-1))

##############################
## define fluid variables
##############################

# time step
dt = 0.01
t0 = 0.0
tf = 2.5
T = 0:dt:tf

# fluid properties
ρ = 997.0 # kg/m^3
μ = 8.9e-4 # Pa*s

# fluid grid
L_x = 0.6
L_y = 0.6

ne_x = 300
ne_y = 300

# boundary conditions
u_∞ = 0.0

U_west_bc = SA[u_∞, 0.0]
U_east_bc = SA[u_∞, 0.0]
U_north_bc = SA[u_∞, 0.0]
U_south_bc = SA[u_∞, 0.0]

# outflow
outflow = SA[false, true, false, false]

ref_L = 0.04022
ref_u = 0.005

##############################
## create multilink model
##############################

ρ_s = 1.0
x1 = [0.1664, 0.3]
nominal_ds = L_y / ne_y

boundary_model = SRLFishTail2D(ρ_s, x1, nominal_ds)
X = interpolate_tail_data(12, dt; fit=:cubic, display_plot=false)

##############################
## create animation
##############################

mkpath(joinpath(VISDIR, "srlfishtail_hardware_validation"))
anime_file = joinpath(VISDIR, "srlfishtail_hardware_validation",
    "srlfishtail_animation_dark.mp4"
)
X_animation = interpolate_tail_data(12, 1/120; fit=:cubic, display_plot=false)
Aquarium.animate_boundary(boundary_model, anime_file, 0.0:1/120:1/120*(length(X_animation)-1), X_animation;
    x_lim=x_lim, y_lim=y_lim, show_vel=false, timescale=2, framerate=60, resolution=resolution, obj_color=obj_color,
    background_color=background_color, linewidth=5, fontsize=fontsize
)

##############################
## make FSI model
##############################

fluid_model = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false
)

###############################################
## normalize
###############################################

normalized_fluid = Aquarium.normalize(fluid_model)
normalized_boundary = Aquarium.normalize(boundary_model, ref_L)

X = map((x) -> Aquarium.normalize(boundary_model, x, ref_L, ref_u), X)

##############################
## simulate
##############################

t = 0.0
uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(normalized_fluid, normalized_boundary, uk, pk, fk_b, X;
     λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=false
)

##############################
## save data
##############################

save_file = joinpath(DATADIR, "SRLFishTail2D_hardware_validation_0to2p5_sec.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary, ref_L, ref_u, t_hist, x_hist, u_hist, p_hist, f_b_hist)

##############################
## load data
##############################

data = load(joinpath(DATADIR, "SRLFishTail2D_hardware_validation_0to2p5_sec.jld2"))
t_hist = data["t_hist"]
x_hist = data["x_hist"]
u_hist = data["u_hist"]
p_hist = data["p_hist"]
f_b_hist = data["f_b_hist"]

t = 1.45
xk = x_hist[t_hist .== t]
uk = u_hist[t_hist .== t]

##############################
## create animation
##############################

mkpath(joinpath(VISDIR, "srlfishtail_hardware_validation"))
anime_file = joinpath(VISDIR, "srlfishtail_hardware_validation",
    "srlfishtail_animation_velocityfield.mp4"
)

animate_velocityfield(normalized_fluid, normalized_boundary,
    t_hist, x_hist, u_hist, anime_file;
    density = 0.4, x_lim=x_lim, y_lim=y_lim,
    framerate=50, timescale=2.0, lengthscale=0.0025,
    arrowcolor=logocolors[3], obj_color=:black,
    display_live=false, normalize_arrow=true,
    fontsize=fontsize, resolution=resolution   
)

##############################
## plot fish tail at 1.45 sec
##############################

boundary_1D = Aquarium.SRLFishTail1D(ρ_s, x1, nominal_ds)
x_b_1d = Aquarium.boundary_state(boundary_1D, xk[1])[1:end÷2]

x_1D = x_b_1d[1:boundary_1D.nodes]
y_1D = x_b_1d[boundary_1D.nodes+1:end]

x_1D_joints = x_1D[vcat(boundary_1D.joints, boundary_1D.nodes)]
y_1D_joints = y_1D[vcat(boundary_1D.joints, boundary_1D.nodes)]

model = boundary_model
x_b = boundary_state(model, xk[1])[1:end÷2]
x = x_b[1:model.joints[end]]
y = x_b[model.nodes+1:model.nodes+model.joints[end]]

p = model.point_order

sorted_x = x[p]
sorted_y = y[p]

x_fin = x_b[model.joints[end]:model.nodes]
y_fin = x_b[model.nodes+model.joints[end]:end]

set_theme!(font = "Times New Roman", fontsize=30)
fig = plot_velocityfield(normalized_fluid, normalized_boundary, xk[1], average(normalized_fluid, uk[1]);
    x_lim=x_lim, y_lim=y_lim, lengthscale=0.0025, arrowcolor=logocolors[3], density = 0.4, obj_color=colorant"grey",
    normalize_arrow=true
)
lines!(x_fin, y_fin, color=:black, linewidth=5, grid=false)
lines!(x_1D, y_1D, color=:black, linewidth=5, grid=false)
scatter!(x_1D_joints[1:end-2], y_1D_joints[1:end-2], color=:black, markersize=17)

display(fig)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "SRLFishTail2D_simulation_1p45_sec.png")
save(filename, fig, px_per_unit = 10)

##############################
## calculate net thrust
##############################

net_Fx_hist = ones(length(f_b_hist))

for i in eachindex(f_b_hist)
    fi_b = f_b_hist[i]
    normalized_Fx = fi_b[1:normalized_boundary.nodes]
    net_Fx_hist[i] = -sum(normalized_Fx .* ref_u^2 .* ref_L)
end

##############################
## plot forces
##############################

start_t = 0.5
net_Fx_hist_normalized = net_Fx_hist ./ maximum(abs.(net_Fx_hist[10:end]))
# F_emp_trunc = F_emp[1:length(T_emp)] ./ maximum(F_emp)

set_theme!(font = "Times New Roman", fontsize=25)
fig, ax, sp = lines(t_hist[t_hist .> start_t][1:end-1], net_Fx_hist_normalized[t_hist .> start_t][2:end], color=:black,
    axis = (xlabel = "Time (s)", ylabel = "Thrust"), label = "Aquarium"
)
# lines!(T_emp[T_emp .> start_t], F_emp_trunc[T_emp .> start_t], color=:red, label = "Empirical")
axislegend(ax, position = :lt)

xlims!(ax, (minimum(t_hist[t_hist .> start_t]), maximum(t_hist[t_hist .> start_t])))
ylims!(ax, (-1.1, 1.1))
display(fig)

hardware_val = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Thrust",
        legend_pos = "north west",
        xmin = 0.5,
        xmax = 2.5,
        ymin = -1.3,
        ymax =1.3,
        
    },
    PlotInc({lineopts..., color="black"}, Coordinates(t_hist[t_hist .> start_t][1:end-1], net_Fx_hist_normalized[t_hist .> start_t][2:end])),
    # PlotInc({lineopts..., color="red"}, Coordinates(T_emp[T_emp .> start_t], F_emp_trunc[T_emp .> start_t])),

    PGFPlotsX.Legend(["Aquarium", "Empirical"])
)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "hardware_validation.tikz")
pgfsave(filename, hardware_val, include_preamble=false)