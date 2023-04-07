import Pkg
Pkg.activate(joinpath(@__DIR__,"..", ".."))

using Aquarium
using TrajectoryOptimization
using LinearAlgebra
using StaticArrays
using RobotDynamics
using Rotations
using JLD2
using CairoMakie
using Makie
using Makie.GeometryBasics
using ForwardDiff, FiniteDiff
using ProgressMeter
using CurveFit
using Interpolations

function interpolate_tail_data(nstates, dt; fit=:cubic, display_plot=true)

    data_path = joinpath(DATADIR, "eth_fish_3hz_states.csv")
    X_θ_data = readdlm(data_path, ',', Float64)
    T_data = 0:1/60:(1/60)*(size(X_θ_data, 1)-1)
    X_θ_data = X_θ_data[eachindex(T_data), :]

    X_ω_data = deepcopy(X_θ_data)
    X_ω_data[2:end-1, :] = reduce(hcat, [(X_θ_data[i+1, :] - X_θ_data[i-1, :]) ./ (2/60) for i in 2:size(X_ω_data, 1)-1])'
    X_ω_data[end, :] = (X_θ_data[end, :] - X_θ_data[end-1, :]) ./ (1/60)

    T = T_data[1]:dt:T_data[end]
    N = size(X_θ_data, 1)

    X_θ_hist = zeros(length(T), nstates)
    X_ω_hist = zeros(length(T), nstates)

    for i in 1:size(X_θ_data, 2)

        ω = zeros(length(T))
        
        θ_data = X_θ_data[:, i]

        if fit == :quadratic
        
            θ_interpolate =scale(interpolate(θ_data, BSpline(Quadratic(Natural(OnGrid())))), T_data)

        elseif fit == :cubic

            θ_interpolate =scale(interpolate(θ_data, BSpline(Cubic(Natural(OnGrid())))), T_data)

        end

        θ = θ_interpolate(T)
        ω = map(x -> Interpolations.gradient(θ_interpolate, x)[1], T)

        X_θ_hist[:, i] .= θ
        X_ω_hist[:, i] .= ω

        if display_plot

            ω_data = X_ω_data[:, i]

            fig, ax = lines(T_data, θ_data, label = "Exp Data θ$i")
            lines!(T, θ, label = "Curve Fit θ$i")
            axislegend(ax, position = :lt)

            display(fig)

            fig, ax = lines(T_data, ω_data, label = "Exp Data ω$i")
            lines!(T, ω, label = "Curve Fit ω$i")
            axislegend(ax, position = :lt)

            display(fig)

        end
        
    end

    X_θ = [X_θ_hist[i, :] for i in 1:size(X_θ_hist, 1)]
    X_ω = [X_ω_hist[i, :] for i in 1:size(X_ω_hist, 1)]

    X = [vcat(X_θ[i], X_ω[i]) for i in eachindex(X_θ)]

    return X

end

function curvefit(nstates, dt; n=10, tf=4, fit=:poly_fit, display_plot=true)

    data_path = joinpath(DATADIR, "eth_fish_3hz_states.csv")
    X_θ_data = readdlm(data_path, ',', Float64)
    T_data = 0:1/60:tf
    X_θ_data = X_θ_data[eachindex(T_data), :]

    X_ω_data = deepcopy(X_θ_data)
    X_ω_data[2:end-1, :] = reduce(hcat, [(X_θ_data[i+1, :] - X_θ_data[i-1, :]) ./ (2/60) for i in 2:size(X_ω_data, 1)-1])'
    X_ω_data[end, :] = (X_θ_data[end, :] - X_θ_data[end-1, :]) ./ (1/60)

    T = T_data[1]:dt:T_data[end]
    N = size(X_θ_data, 1)
    P = T_data[end] - T_data[1]

    X_θ_hist = zeros(length(T), nstates)
    X_ω_hist = zeros(length(T), nstates)

    for i in 1:size(X_θ_data, 2)

        θ = zeros(length(T))
        ω = zeros(length(T))
        
        θ_data = X_θ_data[:, i]

        if fit == :FFT
        
            Fθ = fft(θ_data)[1:N÷2]
            ak =  2/N * real.(Fθ)
            ak[1] = ak[1]/2
            bk = -2/N * imag.(Fθ)

            for i in 1:n
                θ .+= ak[i] .* cos.(2π*(i-1)/P .* T) .+ bk[i] .* sin.(2π*(i-1)/P .* T)
                ω .+= -(ak[i]*2π*(i-1)/P) .* sin.(2π*(i-1)/P .* T) .+ (bk[i]*2π*(i-1)/P) .* cos.(2π*(i-1)/P .* T)
            end

        elseif fit == :Polynomial

            ak = poly_fit(T_data, θ_data, n)

            for i in eachindex(ak)

                θ .+= ak[i].*T.^(i-1)
                ω .+= ((i-1)*ak[i]).*T.^(i-2)

            end

        end

        if display_plot

            ω_data = X_ω_data[:, i]

            fig, ax = lines(T_data, θ_data, label = "Exp Data θ$i")
            lines!(T, θ, label = "Curve Fit θ$i")
            axislegend(ax, position = :lt)

            display(fig)

            fig, ax = lines(T_data, ω_data, label = "Exp Data ω$i")
            lines!(T, ω, label = "Curve Fit ω$i")
            axislegend(ax, position = :lt)

            display(fig)

        end

        X_θ_hist[:, i] .= θ
        X_ω_hist[:, i] .= ω

    end

    X_θ = [X_θ_hist[i, :] for i in 1:size(X_θ_hist, 1)]
    X_ω = [X_ω_hist[i, :] for i in 1:size(X_ω_hist, 1)]

    X = [vcat(X_θ[i], X_ω[i]) for i in eachindex(X_θ)]

    return X

end

function const_curve_flapping(nlinks, freq, max_θ; ts=0.0, dt=0.01, tf=5.0)

    N = Int(floor((tf-ts)/dt) + 1)
    x = [zeros(Int(2*nlinks)) for _ in 1:N]
    θ = [zeros(nlinks) for _ in 1:N]
    ω = [zeros(nlinks) for _ in 1:N]
    T = ts:dt:tf

    for i in 1:N

        t = T[i]

        θ[i] .= max_θ*sin((2*pi*freq)*t)
        ω[i] .= 2*pi*freq*max_θ*cos((2*pi*freq)*t)

        x[i] = vcat(θ[i], ω[i])

    end

    return x, θ, ω

end

function const_curve_flapping_max_ω(nlinks, freq, max_ω; ts=0.0, dt=0.01, tf=5.0)

    N = Int(floor((tf-ts)/dt) + 1)
    x = [zeros(Int(2*nlinks)) for _ in 1:N]
    θ = [zeros(nlinks) for _ in 1:N]
    ω = [zeros(nlinks) for _ in 1:N]
    T = ts:dt:tf

    for i in 1:N

        t = T[i]

        θ[i] .= (max_ω/(2*pi*freq))*sin((2*pi*freq)*t)
        ω[i] .= max_ω*cos((2*pi*freq)*t)

        x[i] = vcat(θ[i], ω[i])

    end

    return x, θ, ω

end

function jointwidth_cubic_spline(coeff, jointlocations, startwidth)

    a = 10 .* coeff
    b = -(3/2)*a*jointlocations[end]

    d = startwidth/2

    joint_half_width = a.*jointlocations.^3 + b.*jointlocations.^2 .+ d
    jointwidths = 2 .* joint_half_width

    return jointwidths
end

# function animate_tail(model::SRLFishTail1D, anime_file, X; x_lim=(0.0, nothing),
#     y_lim=(0.0, nothing), color=:black, linewidth=5, lengthscale=0.5,
#     framerate=60, show_vel=true)

#     x = X[1]
#     state_b = boundary_state(model, x)
#     x_b = state_b[1:end÷2]
#     u_b = state_b[end÷2+1:end]

#     x = x_b[1:model.nodes]
#     y = x_b[model.nodes+1:end]
#     u = u_b[1:model.nodes]
#     v = u_b[model.nodes+1:end]

#     total_frames = length(X)
#     p = Progress(total_frames, 1, "Creating animation...")

#     fig, ax = plot_boundary(model, X[1], linewidth=linewidth)

#     if show_vel
#         arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
#     end

#     ax.aspect = DataAspect()
#     xlims!(ax, x_lim)
#     ylims!(ax, y_lim)

#     record(fig, anime_file, 1:total_frames, framerate=framerate) do i

#         empty!(ax)
#         x = X[i]

#         state_b = boundary_state(model, x)
#         x_b = state_b[1:end÷2]
#         u_b = state_b[end÷2+1:end]
        
#         x = x_b[1:model.nodes]
#         y = x_b[model.nodes+1:end]
#         u = u_b[1:model.nodes]
#         v = u_b[model.nodes+1:end]

#         plot_boundary!(model, X[i], linewidth=linewidth)
        
#         if show_vel
#             arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
#         end

#         ax.aspect = DataAspect()
#         xlims!(ax, x_lim)
#         ylims!(ax, y_lim)

#         next!(p)

#     end

# end

# function animate_tail(model::SRLFishTail2D, anime_file, X; x_lim=(0.0, nothing),
#     y_lim=(0.0, nothing), linewidth=5, color=:black, framerate=60, lengthscale=0.5, show_vel=true)

#     x = X[1]
    
#     state_b = boundary_state(model, x)
#     x_b = state_b[1:end÷2]
#     u_b = state_b[end÷2+1:end]

#     x = x_b[1:model.nodes]
#     y = x_b[model.nodes+1:end]
#     u = u_b[1:model.nodes]
#     v = u_b[model.nodes+1:end]

#     total_frames = length(X)
#     p = Progress(total_frames, 1, "Creating animation...")
    
#     fig, ax = plot_boundary(model, X[1], linewidth=linewidth)

#     if show_vel
#         arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
#     end

#     ax.aspect = DataAspect()
#     xlims!(ax, x_lim)
#     ylims!(ax, y_lim)

#     record(fig, anime_file, 1:total_frames, framerate=framerate) do i

#         empty!(ax)
#         x = X[i]

#         state_b = boundary_state(model, x)
#         x_b = state_b[1:end÷2]
#         u_b = state_b[end÷2+1:end]
        
#         x = x_b[1:model.nodes]
#         y = x_b[model.nodes+1:end]
#         u = u_b[1:model.nodes]
#         v = u_b[model.nodes+1:end]

#         plot_boundary!(model, X[i], linewidth=linewidth)

#         if show_vel
#             arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
#         end

#         ax.aspect = DataAspect()
#         xlims!(ax, x_lim)
#         ylims!(ax, y_lim)

#         next!(p)

#     end

# end