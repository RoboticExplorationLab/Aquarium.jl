import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

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

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

##############################
## plotting constants
##############################

density = 150.0
levels = 20
level_perc = 0.2
x_lim=[0.5, 3.75]
y_lim=[1.5, 2.5]
colormap_streamlines = :blues
colormap_vorticity = :bwr
background_color = :white
obj_color = :black
fontsize=50
framerate=30
timescale=10
streamline_linewidth = 4
resolution=(2000, 800)

##############################
## define fluid variables
##############################

# time step
dt = 0.001
tf = 2.5

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.1 # Pa*s

# fluid grid
L_x = 4.0
L_y = 4.0

ne_x = 500
ne_y = 500

# boundary conditions
u_∞ = 20.0
u_∞ = 50.0 # <- uncomment to simulate vortex shedding

u_west_bc = SA[u_∞, 0.0]
u_east_bc = SA[u_∞, 0.0]
u_north_bc = SA[u_∞, 0.0]
u_south_bc = SA[u_∞, 0.0]

# outflow
outflow = SA[false, true, false, false]

########################################
## define Immersed boundary variables
########################################

# boundary solid density
ρ_b = 1.0 # kg/m^3

# geometric properties
D = 0.2
cm_x = 1.0
cm_y = L_y/2
nodes=79

# normalization references
ref_L = D
ref_u = u_∞

##############################
## make models
##############################

# create boundary
boundary = Aquarium.Cylinder(ρ_b, D; nodes=nodes)

# make FSIModel
fluid = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, u_west_bc,
    u_east_bc, u_north_bc, u_south_bc, outflow; normalize=false
)
Re = Int(fluid.Re)

###############################################
## normalize
###############################################

normalized_fluid = Aquarium.normalize(fluid)
normalized_boundary = Aquarium.normalize(boundary, ref_L)

x = Aquarium.normalize(boundary, SA[cm_x, cm_y, 0., 0., 0., 0.], ref_L, ref_u)

##############################
## simulate
##############################

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(normalized_fluid, normalized_boundary, uk, pk, fk_b, [x];
     λ1=1e-6, tol=1e-6, tf=tf, alg=:pardiso, verbose=true)

##############################
## save data
##############################

save_file = joinpath(DATADIR, "freestream_cylinder_Re$Re"*"_0to2p5_sec.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary, ref_L, ref_u, t_hist, x_hist, u_hist, p_hist, f_b_hist)

##############################
## load data
##############################

data = load(joinpath(DATADIR, "freestream_cylinder_Re$Re"*"_0to2p5_sec.jld2"))
t_hist = data["t_hist"]
x_hist = data["x_hist"]
u_hist = data["u_hist"]
p_hist = data["p_hist"]
f_b_hist = data["f_b_hist"]
x = x_hist[end]
uk = u_hist[end]

##############################
## calculate forces
##############################

CD = ones(length(f_b_hist))
CL = ones(length(f_b_hist))

for i in eachindex(f_b_hist)
    F = f_b_hist[i]

    normalized_Fx = F[1:normalized_boundary.nodes]
    normalized_Fy = F[normalized_boundary.nodes+1:end]
    
    CD[i] = 2 .* sum(normalized_Fx) ./ normalized_fluid.ρ
    CL[i] = 2 .* sum(normalized_Fy) ./ normalized_fluid.ρ

end

##############################
## Calculate steady state values
##############################

# Re = 40
CD_steady_state = minimum(CD[end])

# Re = 100

CD_upper = maximum(CD[end-50:end])
CD_lower = minimum(CD[end-50:end])
CD_avg = (CD_upper + CD_lower)/2

CL_upper = maximum(CL[end-50:end])
CL_lower = minimum(CL[end-50:end])
CL_avg = (CL_upper + CL_lower)/2

T_analysis = t_hist[end-50:end];
t1 = T_analysis[end-25:end][CL[end-25:end] .== maximum(CL[end-25:end])][1];
t2 = T_analysis[end-50:end-25][CL[end-50:end-25] .== maximum(CL[end-50:end-25])][1];

St = 1/(t1-t2)*ref_L/ref_u

##############################
## plot forces
##############################

t_start = 0.5
t_end = 1.5

t_hist_plot = t_hist[t_hist .>= t_start]
CD_hist_plot = CD[t_hist .>= t_start]
CD_hist_plot = CD_hist_plot[t_hist_plot .<= t_end]
CL_hist_plot = CL[t_hist .>= t_start]
CL_hist_plot = CL_hist_plot[t_hist_plot .<= t_end]
t_hist_plot = t_hist_plot[t_hist_plot .<= t_end]

fig, ax, sp = lines(t_hist_plot, CD_hist_plot, color=:blue, axis = (xlabel = "Time (s)", ylabel = "Drag/Lift Coefficient"),
    label = "Drag Coefficient")
lines!(t_hist_plot, CL_hist_plot, color=:red, label = "Lift Coefficient")
axislegend(ax, position = :lt)

display(fig)

##############################
## plot
##############################

uk_avg = average(normalized_fluid, uk)
scene = plot_streamlines(normalized_fluid, normalized_boundary, x, uk_avg;
    density=density, linewidth=streamline_linewidth, x_lim=x_lim, y_lim=y_lim,
    colormap=colormap_streamlines, fontsize=fontsize, obj_color=obj_color,
    resolution=resolution, background_color=background_color,
)
display(scene)

scene = plot_vorticity(normalized_fluid, normalized_boundary, x, uk_avg;
    levels=levels, level_perc=level_perc, x_lim=x_lim, y_lim=y_lim,
    colormap=colormap_vorticity, obj_color=obj_color, fontsize=fontsize,
    resolution=resolution
)
display(scene)

mkpath(joinpath(VISDIR, "paper"))
filename = joinpath(VISDIR, "paper", "freestream_cylinder_Re$Re.png")
save(filename, scene, px_per_unit = 4)

##############################
## create animation
##############################

mkpath(joinpath(VISDIR, "freestream_cylinder", "Re$Re"))

anime_file = joinpath(VISDIR, "freestream_cylinder", "Re$Re",
    "freestream_cylinder_Re$Re"*"_streamlines.mp4")

animate_streamlines(normalized_fluid, normalized_boundary, t_hist, x_hist, u_hist, anime_file;
    density=density, linewidth=streamline_linewidth, x_lim=x_lim, y_lim=y_lim, colormap=colormap_streamlines,
    obj_color=obj_color, fontsize=fontsize, framerate=framerate, timescale=timescale, display_live=false,
    resolution=resolution
)

anime_file = joinpath(VISDIR, "freestream_cylinder", "Re$Re",
    "freestream_cylinder_Re$Re"*"_vorticity.mp4")

animate_vorticity(normalized_fluid, normalized_boundary, t_hist, x_hist, u_hist, anime_file;
    levels=levels, level_perc=level_perc, x_lim=x_lim, y_lim=y_lim, colormap=colormap_vorticity,
    obj_color=obj_color, fontsize=fontsize, framerate=framerate, timescale=timescale, display_live=false,
    resolution=resolution
)