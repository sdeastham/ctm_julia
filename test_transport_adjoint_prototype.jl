include(joinpath(@__DIR__, "ctm_v1_core.jl"))

using .CTMV1Core
using LinearAlgebra
using Random
using Dates

const JAC_STEP = 1.0e-6
const DIR_TOL = 2.0e-3
const TEST_VERSION = "v3"

struct TransportCase
    grid::CTMV1Core.Grid2D
    lat_field::Array{Float64,2}
    lon_field::Array{Float64,2}
    u::Array{Float64,2}
    v::Array{Float64,2}
    dt_sec::Float64
    t_utc::DateTime
    receptor_lat::Float64
    receptor_lon::Float64
end

function build_case()
    grid = CTMV1Core.make_grid(
        dx_km=80.0,
        dy_km=80.0,
        xmin_km=-160.0,
        xmax_km=160.0,
        ymin_km=-160.0,
        ymax_km=160.0,
        lon0_deg=-2.0,
        lat0_deg=54.0,
    )
    lon_field, lat_field = CTMV1Core.build_lonlat_fields(grid)

    u = fill(4.0, grid.nx, grid.ny)
    v = fill(-1.5, grid.nx, grid.ny)

    return TransportCase(
        grid,
        lat_field,
        lon_field,
        u,
        v,
        120.0,
        DateTime(2015, 1, 1, 12, 0, 0),
        51.498926,
        -0.174777,
    )
end

function initial_state(case::TransportCase)
    state = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    c0 = CTMV1Core.default_background_vector()
    for j in 1:case.grid.ny
        for i in 1:case.grid.nx
            @inbounds state[i, j, :] .= c0
            state[i, j, CTMV1Core.SIDX[:O3]] *= 1.0 + 0.05 * sin(0.7 * i - 0.4 * j)
        end
    end
    return state
end

function transport_map_o3(x0_o3_vec, case::TransportCase)
    nx, ny = case.grid.nx, case.grid.ny

    state = initial_state(case)
    o3 = CTMV1Core.SIDX[:O3]
    @views state[:, :, o3] .= reshape(x0_o3_vec, nx, ny)

    CTMV1Core.apply_transport_step!(
        state,
        case.u,
        case.v,
        case.grid,
        case.dt_sec,
        case.t_utc,
        case.lat_field,
        case.lon_field;
        limiter=:vanleer,
        bc_mode=:zero,
    )

    return vec(@view(state[:, :, o3]))
end

function forward_objective(alpha_vec, case::TransportCase)
    nx, ny = case.grid.nx, case.grid.ny
    state0 = initial_state(case)
    o3 = CTMV1Core.SIDX[:O3]
    x0 = vec(@view(state0[:, :, o3])) .+ alpha_vec
    x1 = transport_map_o3(x0, case)

    return CTMV1Core.bilinear_receptor_value(
        reshape(x1, nx, ny),
        case.lat_field,
        case.lon_field,
        case.receptor_lat,
        case.receptor_lon,
    )
end

function reverse_wind_gradient(alpha_vec, case::TransportCase)
    nx, ny = case.grid.nx, case.grid.ny
    state_adj = zeros(Float64, nx, ny, CTMV1Core.NSPEC)

    CTMV1Core.accumulate_objective_pullback!(
        state_adj,
        :O3,
        1.0,
        case.lat_field,
        case.lon_field,
        case.receptor_lat,
        case.receptor_lon,
    )

    # Heuristic often suggested for transport: one step with reversed winds.
    CTMV1Core.apply_transport_step!(
        state_adj,
        .-case.u,
        .-case.v,
        case.grid,
        case.dt_sec,
        case.t_utc,
        case.lat_field,
        case.lon_field;
        limiter=:vanleer,
        bc_mode=:zero,
    )

    return vec(@view(state_adj[:, :, CTMV1Core.SIDX[:O3]]))
end

function objective_pullback_vector(case::TransportCase)
    lambda_state = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    CTMV1Core.accumulate_objective_pullback!(
        lambda_state,
        :O3,
        1.0,
        case.lat_field,
        case.lon_field,
        case.receptor_lat,
        case.receptor_lon,
    )
    return vec(@view(lambda_state[:, :, CTMV1Core.SIDX[:O3]]))
end

function discrete_transport_adjoint_gradient(alpha_vec, case::TransportCase; h=JAC_STEP)
    nctl = length(alpha_vec)
    state0 = initial_state(case)
    o3 = CTMV1Core.SIDX[:O3]
    x0 = vec(@view(state0[:, :, o3])) .+ alpha_vec
    lambda1 = objective_pullback_vector(case)

    g = zeros(Float64, nctl)
    for i in 1:nctl
        xp = copy(x0)
        xm = copy(x0)
        xp[i] += h
        xm[i] -= h
        yp = transport_map_o3(xp, case)
        ym = transport_map_o3(xm, case)
        col_i = (yp .- ym) ./ (2.0 * h)
        g[i] = dot(col_i, lambda1)
    end
    return g
end

function main()
    case = build_case()
    nctl = case.grid.nx * case.grid.ny
    alpha0 = zeros(Float64, nctl)

    state0 = initial_state(case)
    o3 = CTMV1Core.SIDX[:O3]
    x0 = vec(@view(state0[:, :, o3])) .+ alpha0
    lambda1 = objective_pullback_vector(case)

    state_prev = copy(state0)
    @views state_prev[:, :, o3] .= reshape(x0, case.grid.nx, case.grid.ny)
    lambda_next = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    @views lambda_next[:, :, o3] .= reshape(lambda1, case.grid.nx, case.grid.ny)
    lambda_prev = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    CTMV1Core.transport_pullback_step!(
        lambda_prev,
        lambda_next,
        state_prev,
        case.u,
        case.v,
        case.grid,
        case.dt_sec,
        case.t_utc,
        case.lat_field,
        case.lon_field;
        limiter=:vanleer,
        bc_mode=:zero,
    )
    grad_analytic = vec(@view(lambda_prev[:, :, o3]))
    grad_discrete = discrete_transport_adjoint_gradient(alpha0, case)

    rng = MersenneTwister(11)
    dir = randn(rng, nctl)
    dir ./= norm(dir)

    h = 1.0e-6
    jp = forward_objective(alpha0 .+ h .* dir, case)
    jm = forward_objective(alpha0 .- h .* dir, case)
    fd_dir = (jp - jm) / (2.0 * h)
    analytic_dir = dot(grad_analytic, dir)
    analytic_rel = abs(analytic_dir - fd_dir) / max(abs(analytic_dir), abs(fd_dir), eps())
    discrete_dir = dot(grad_discrete, dir)
    discrete_rel = abs(discrete_dir - fd_dir) / max(abs(discrete_dir), abs(fd_dir), eps())

    analytic_vs_discrete = norm(grad_analytic - grad_discrete) / max(norm(grad_discrete), eps())

    grad_rw = reverse_wind_gradient(alpha0, case)
    rw_dir = dot(grad_rw, dir)
    rw_rel_vs_discrete = abs(rw_dir - discrete_dir) / max(abs(rw_dir), abs(discrete_dir), eps())

    out = joinpath(@__DIR__, "transport_adjoint_prototype_v2.txt")
    open(out, "w") do io
        println(io, "TEST_VERSION = ", TEST_VERSION)
        println(io, "NCTRL = ", nctl)
        println(io, "FD_DIR = ", fd_dir)
        println(io, "ANALYTIC_ADJ_DIR = ", analytic_dir)
        println(io, "ANALYTIC_ADJ_RELERR = ", analytic_rel)
        println(io, "DISCRETE_ADJ_DIR = ", discrete_dir)
        println(io, "DISCRETE_ADJ_RELERR = ", discrete_rel)
        println(io, "DISCRETE_ADJ_TOL = ", DIR_TOL)
        println(io, "ANALYTIC_VS_DISCRETE_RELNORM = ", analytic_vs_discrete)
        println(io, "REVERSE_WIND_DIR = ", rw_dir)
        println(io, "REVERSE_WIND_RELERR_VS_DISCRETE = ", rw_rel_vs_discrete)
        println(io, "PASS = ", (discrete_rel <= DIR_TOL) && (analytic_rel <= DIR_TOL))
    end

    println("WROTE = ", out)

    if discrete_rel > DIR_TOL || analytic_rel > DIR_TOL
        error("Transport adjoint check failed: discrete_rel=$(discrete_rel), analytic_rel=$(analytic_rel), tol=$(DIR_TOL)")
    end

    println("PASS: transport analytic and discrete adjoint directional checks")
end

main()
