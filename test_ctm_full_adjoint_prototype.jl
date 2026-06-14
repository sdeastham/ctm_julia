include(joinpath(@__DIR__, "ctm_v1_core.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))

using .CTMV1Core
using .CanteraMechanismTypes
using .CanteraMechanismCodegen
using LinearAlgebra
using Random
using Dates

const REL_TOL = 2.0e-3
const NSTEPS = 6

struct FullAdjointCase
    grid::CTMV1Core.Grid2D
    lat_field::Array{Float64,2}
    lon_field::Array{Float64,2}
    baseline_no::Array{Float64,2}
    u::Array{Float64,2}
    v::Array{Float64,2}
    temp_k::Float64
    dt_sec::Float64
    t0::DateTime
    receptor_lat::Float64
    receptor_lon::Float64
    mech::CanteraMechanismTypes.Mechanism
    idx_new::Dict{Symbol,Int}
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

    baseline_no = zeros(Float64, grid.nx, grid.ny)
    for j in 1:grid.ny
        for i in 1:grid.nx
            baseline_no[i, j] = 4.0e-9 * (1.0 + 0.06 * i - 0.04 * j)
        end
    end

    u = fill(4.0, grid.nx, grid.ny)
    v = fill(-1.5, grid.nx, grid.ny)

    mech = CanteraMechanismTypes.parse_cantera_yaml(joinpath(@__DIR__, "cantera_mechanism_example.yaml"))
    idx_new = CanteraMechanismCodegen.species_index_map(mech)

    return FullAdjointCase(
        grid,
        lat_field,
        lon_field,
        baseline_no,
        u,
        v,
        293.0,
        120.0,
        DateTime(2015, 1, 1, 12, 0, 0),
        51.498926,
        -0.174777,
        mech,
        idx_new,
    )
end

function initial_state(case::FullAdjointCase)
    state = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    c0 = CTMV1Core.default_background_vector()
    for j in 1:case.grid.ny
        for i in 1:case.grid.nx
            @inbounds state[i, j, :] .= c0
            state[i, j, CTMV1Core.SIDX[:NO]] *= 1.0 + 0.04 * sin(0.8 * i + 0.3 * j)
            state[i, j, CTMV1Core.SIDX[:NO2]] *= 1.0 + 0.03 * cos(0.5 * i - 0.4 * j)
            state[i, j, CTMV1Core.SIDX[:O3]] *= 1.0 + 0.03 * sin(0.3 * i - 0.2 * j)
        end
    end
    return state
end

function old_to_new_cell!(cnew, cold, idx_new)
    fill!(cnew, 0.0)
    for (sp, iold) in CTMV1Core.SIDX
        if haskey(idx_new, sp)
            cnew[idx_new[sp]] = cold[iold]
        end
    end
    return nothing
end

function new_to_old_cell!(cold, cnew, idx_new)
    fill!(cold, 0.0)
    for (sp, iold) in CTMV1Core.SIDX
        if haskey(idx_new, sp)
            cold[iold] = cnew[idx_new[sp]]
        end
    end
    return nothing
end

function apply_emissions_step!(state, alpha_k, case::FullAdjointCase)
    no_idx = CTMV1Core.SIDX[:NO]
    conv = 1.0 / (CTMV1Core.MW_NO_KG_PER_MOL * CTMV1Core.DEFAULT_MIXING_DEPTH_M)
    for j in 1:case.grid.ny
        for i in 1:case.grid.nx
            state[i, j, no_idx] += case.dt_sec * conv * case.baseline_no[i, j] * alpha_k[i, j]
        end
    end
    return nothing
end

function chemistry_step_with_mask!(state_out, mask_out, state_in, hour_of_day, case::FullAdjointCase)
    nx, ny = case.grid.nx, case.grid.ny

    cnew = zeros(Float64, length(case.mech.species))
    dcnew = similar(cnew)
    cold = zeros(Float64, CTMV1Core.NSPEC)
    dcold = zeros(Float64, CTMV1Core.NSPEC)

    for j in 1:ny
        for i in 1:nx
            @views cold .= state_in[i, j, :]
            old_to_new_cell!(cnew, cold, case.idx_new)
            CanteraMechanismCodegen.chemistry_rhs!(dcnew, cnew, case.temp_k, hour_of_day, case.mech)
            new_to_old_cell!(dcold, dcnew, case.idx_new)

            @inbounds for s in 1:CTMV1Core.NSPEC
                z = state_in[i, j, s] + case.dt_sec * dcold[s]
                mask_out[i, j, s] = z > 0.0
                state_out[i, j, s] = max(z, 0.0)
            end
        end
    end
    return nothing
end

function forward_trajectory(alpha_vec, case::FullAdjointCase)
    nx, ny = case.grid.nx, case.grid.ny
    alpha = reshape(alpha_vec, nx, ny, NSTEPS)

    state_prev = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]
    state_tran = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]
    state_emis = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]
    state_next = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]
    chem_mask = [zeros(Bool, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]

    curr = initial_state(case)
    for k in 1:NSTEPS
        state_prev[k] .= curr

        state_tran[k] .= curr
        t_k = case.t0 + Millisecond(round(Int, 1000 * case.dt_sec * (k - 1)))
        CTMV1Core.apply_transport_step!(
            state_tran[k],
            case.u,
            case.v,
            case.grid,
            case.dt_sec,
            t_k,
            case.lat_field,
            case.lon_field;
            limiter=:vanleer,
            bc_mode=:zero,
        )

        state_emis[k] .= state_tran[k]
        apply_emissions_step!(state_emis[k], @view(alpha[:, :, k]), case)

        hour_of_day = Dates.hour(t_k) + Dates.minute(t_k) / 60 + Dates.second(t_k) / 3600
        chemistry_step_with_mask!(state_next[k], chem_mask[k], state_emis[k], hour_of_day, case)

        curr .= state_next[k]
    end

    return state_prev, state_tran, state_emis, state_next, chem_mask
end

function forward_objective(alpha_vec, case::FullAdjointCase)
    _, _, _, state_next, _ = forward_trajectory(alpha_vec, case)
    obj = 0.0
    for k in 1:NSTEPS
        obj += CTMV1Core.bilinear_receptor_value(
            @view(state_next[k][:, :, CTMV1Core.SIDX[:O3]]),
            case.lat_field,
            case.lon_field,
            case.receptor_lat,
            case.receptor_lon,
        )
    end
    return obj / NSTEPS
end

function adjoint_gradient(case::FullAdjointCase, alpha_vec)
    nx, ny = case.grid.nx, case.grid.ny
    state_prev, _, state_emis, _, chem_mask = forward_trajectory(alpha_vec, case)

    grad_alpha = zeros(Float64, nx, ny, NSTEPS)
    obj_weight = 1.0 / NSTEPS

    lambda_next = zeros(Float64, nx, ny, CTMV1Core.NSPEC)
    lambda_emis = zeros(Float64, nx, ny, CTMV1Core.NSPEC)
    lambda_tran = zeros(Float64, nx, ny, CTMV1Core.NSPEC)
    lambda_prev = zeros(Float64, nx, ny, CTMV1Core.NSPEC)

    cnew = zeros(Float64, length(case.mech.species))
    lamb_new = zeros(Float64, length(case.mech.species))
    vjp_new = zeros(Float64, length(case.mech.species))
    lamb_old = zeros(Float64, CTMV1Core.NSPEC)
    lamb_masked = zeros(Float64, CTMV1Core.NSPEC)
    vjp_old = zeros(Float64, CTMV1Core.NSPEC)

    no_idx = CTMV1Core.SIDX[:NO]
    conv = 1.0 / (CTMV1Core.MW_NO_KG_PER_MOL * CTMV1Core.DEFAULT_MIXING_DEPTH_M)

    for k in NSTEPS:-1:1
        CTMV1Core.accumulate_objective_pullback!(
            lambda_next,
            :O3,
            obj_weight,
            case.lat_field,
            case.lon_field,
            case.receptor_lat,
            case.receptor_lon,
        )

        fill!(lambda_emis, 0.0)
        t_k = case.t0 + Millisecond(round(Int, 1000 * case.dt_sec * (k - 1)))
        hour_of_day = Dates.hour(t_k) + Dates.minute(t_k) / 60 + Dates.second(t_k) / 3600

        for j in 1:ny
            for i in 1:nx
                @inbounds for s in 1:CTMV1Core.NSPEC
                    lamb_masked[s] = chem_mask[k][i, j, s] ? lambda_next[i, j, s] : 0.0
                end

                old_to_new_cell!(cnew, @view(state_emis[k][i, j, :]), case.idx_new)
                old_to_new_cell!(lamb_new, lamb_masked, case.idx_new)
                CanteraMechanismCodegen.chemistry_vjp!(vjp_new, cnew, lamb_new, case.temp_k, hour_of_day, case.mech)
                new_to_old_cell!(vjp_old, vjp_new, case.idx_new)

                @inbounds for s in 1:CTMV1Core.NSPEC
                    lambda_emis[i, j, s] = lamb_masked[s] + case.dt_sec * vjp_old[s]
                end
            end
        end

        lambda_tran .= lambda_emis
        for j in 1:ny
            for i in 1:nx
                grad_alpha[i, j, k] = lambda_tran[i, j, no_idx] * case.dt_sec * conv * case.baseline_no[i, j]
            end
        end

        fill!(lambda_prev, 0.0)
        CTMV1Core.transport_pullback_step!(
            lambda_prev,
            lambda_tran,
            state_prev[k],
            case.u,
            case.v,
            case.grid,
            case.dt_sec,
            t_k,
            case.lat_field,
            case.lon_field;
            limiter=:vanleer,
            bc_mode=:zero,
        )

        lambda_next .= lambda_prev
    end

    return vec(grad_alpha)
end

function main()
    case = build_case()
    nctl = case.grid.nx * case.grid.ny * NSTEPS
    alpha0 = ones(Float64, nctl)

    g_adj = adjoint_gradient(case, alpha0)

    rng = MersenneTwister(17)
    dir = randn(rng, nctl)
    dir ./= norm(dir)

    h = 1.0e-6
    jp = forward_objective(alpha0 .+ h .* dir, case)
    jm = forward_objective(alpha0 .- h .* dir, case)
    fd_dir = (jp - jm) / (2.0 * h)
    adj_dir = dot(g_adj, dir)
    rel = abs(adj_dir - fd_dir) / max(abs(fd_dir), abs(adj_dir), eps())

    out = joinpath(@__DIR__, "ctm_full_adjoint_prototype.txt")
    open(out, "w") do io
        println(io, "NCTRL = ", nctl)
        println(io, "NSTEPS = ", NSTEPS)
        println(io, "FD_DIR = ", fd_dir)
        println(io, "ADJ_DIR = ", adj_dir)
        println(io, "RELERR = ", rel)
        println(io, "REL_TOL = ", REL_TOL)
        println(io, "PASS = ", rel <= REL_TOL)
    end

    println("WROTE = ", out)

    if rel > REL_TOL
        error("Full CTM adjoint prototype failed: rel=$(rel), tol=$(REL_TOL)")
    end

    println("PASS: full CTM forward/adjoint prototype directional check")
end

main()
