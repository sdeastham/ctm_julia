include(joinpath(@__DIR__, "ctm_v1_core.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))

using .CTMV1Core
using .CanteraMechanismTypes
using .CanteraMechanismCodegen
using LinearAlgebra
using Random

const REL_TOL = 1.0e-6
const NSTEPS = 8

struct PrototypeCase
    grid::CTMV1Core.Grid2D
    lat_field::Array{Float64,2}
    lon_field::Array{Float64,2}
    baseline_no::Array{Float64,2}
    dt_sec::Float64
    temp_k::Float64
    hour_of_day::Float64
    receptor_lat::Float64
    receptor_lon::Float64
    mech::CanteraMechanismTypes.Mechanism
    idx_new::Dict{Symbol,Int}
end

function build_case()
    grid = CTMV1Core.make_grid(
        dx_km=100.0,
        dy_km=100.0,
        xmin_km=-150.0,
        xmax_km=150.0,
        ymin_km=-150.0,
        ymax_km=150.0,
        lon0_deg=-2.0,
        lat0_deg=54.0,
    )
    lon_field, lat_field = CTMV1Core.build_lonlat_fields(grid)

    baseline_no = zeros(Float64, grid.nx, grid.ny)
    for j in 1:grid.ny
        for i in 1:grid.nx
            baseline_no[i, j] = 5.0e-9 * (1.0 + 0.1 * i - 0.05 * j)
        end
    end

    mech = CanteraMechanismTypes.parse_cantera_yaml(joinpath(@__DIR__, "cantera_mechanism_example.yaml"))
    idx_new = CanteraMechanismCodegen.species_index_map(mech)

    return PrototypeCase(
        grid,
        lat_field,
        lon_field,
        baseline_no,
        300.0,
        293.0,
        12.0,
        51.498926,
        -0.174777,
        mech,
        idx_new,
    )
end

function initial_state(case::PrototypeCase)
    state = zeros(Float64, case.grid.nx, case.grid.ny, CTMV1Core.NSPEC)
    c0 = CTMV1Core.default_background_vector()
    for j in 1:case.grid.ny
        for i in 1:case.grid.nx
            @inbounds state[i, j, :] .= c0
            state[i, j, CTMV1Core.SIDX[:NO]] *= 1.0 + 0.03 * sin(0.8 * i + 0.3 * j)
            state[i, j, CTMV1Core.SIDX[:NO2]] *= 1.0 + 0.03 * cos(0.4 * i - 0.5 * j)
            state[i, j, CTMV1Core.SIDX[:O3]] *= 1.0 + 0.02 * sin(0.3 * i - 0.2 * j)
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

function apply_emissions_alpha!(state, alpha, case::PrototypeCase)
    nx, ny = case.grid.nx, case.grid.ny
    no_idx = CTMV1Core.SIDX[:NO]
    conv_emis = 1.0 / (CTMV1Core.MW_NO_KG_PER_MOL * CTMV1Core.DEFAULT_MIXING_DEPTH_M)
    for j in 1:ny
        for i in 1:nx
            state[i, j, no_idx] += case.dt_sec * conv_emis * case.baseline_no[i, j] * alpha[i, j]
        end
    end
    return nothing
end

function chemistry_step_with_mask!(state_out, mask_out, state_in, case::PrototypeCase)
    nx, ny = case.grid.nx, case.grid.ny

    cnew = zeros(Float64, length(case.mech.species))
    dcnew = similar(cnew)
    cold = zeros(Float64, CTMV1Core.NSPEC)
    dcold = zeros(Float64, CTMV1Core.NSPEC)

    for j in 1:ny
        for i in 1:nx
            @views cold .= state_in[i, j, :]
            old_to_new_cell!(cnew, cold, case.idx_new)
            CanteraMechanismCodegen.chemistry_rhs!(dcnew, cnew, case.temp_k, case.hour_of_day, case.mech)
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

function forward_trajectory(alpha_vec, case::PrototypeCase)
    nx, ny = case.grid.nx, case.grid.ny
    alpha = reshape(alpha_vec, nx, ny, NSTEPS)

    states = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:(NSTEPS + 1)]
    emis_states = [zeros(Float64, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]
    masks = [zeros(Bool, nx, ny, CTMV1Core.NSPEC) for _ in 1:NSTEPS]

    states[1] .= initial_state(case)

    for k in 1:NSTEPS
        emis_states[k] .= states[k]
        apply_emissions_alpha!(emis_states[k], @view(alpha[:, :, k]), case)
        chemistry_step_with_mask!(states[k + 1], masks[k], emis_states[k], case)
    end

    return states, emis_states, masks
end

function forward_objective(alpha_vec, case::PrototypeCase)
    states, _, _ = forward_trajectory(alpha_vec, case)
    obj = 0.0
    for k in 1:NSTEPS
        obj += CTMV1Core.bilinear_receptor_value(
            @view(states[k + 1][:, :, CTMV1Core.SIDX[:O3]]),
            case.lat_field,
            case.lon_field,
            case.receptor_lat,
            case.receptor_lon,
        )
    end
    return obj / NSTEPS
end

function adjoint_gradient(case::PrototypeCase, alpha_vec)
    nx, ny = case.grid.nx, case.grid.ny
    states, emis_states, masks = forward_trajectory(alpha_vec, case)

    obj_weight = 1.0 / NSTEPS
    lambda_curr = zeros(Float64, nx, ny, CTMV1Core.NSPEC)
    lambda_prev = similar(lambda_curr)

    cnew = zeros(Float64, length(case.mech.species))
    no_idx = CTMV1Core.SIDX[:NO]
    conv_emis = 1.0 / (CTMV1Core.MW_NO_KG_PER_MOL * CTMV1Core.DEFAULT_MIXING_DEPTH_M)
    grad_alpha = zeros(Float64, nx, ny, NSTEPS)

    lamb_new = zeros(Float64, length(case.mech.species))
    vjp_new = zeros(Float64, length(case.mech.species))
    lamb_old = zeros(Float64, CTMV1Core.NSPEC)
    lamb_masked = zeros(Float64, CTMV1Core.NSPEC)
    vjp_old = zeros(Float64, CTMV1Core.NSPEC)

    for k in NSTEPS:-1:1
        CTMV1Core.accumulate_objective_pullback!(
            lambda_curr,
            :O3,
            obj_weight,
            case.lat_field,
            case.lon_field,
            case.receptor_lat,
            case.receptor_lon,
        )

        fill!(lambda_prev, 0.0)
        for j in 1:ny
            for i in 1:nx
                @inbounds for s in 1:CTMV1Core.NSPEC
                    lamb_masked[s] = masks[k][i, j, s] ? lambda_curr[i, j, s] : 0.0
                end

                old_to_new_cell!(cnew, @view(emis_states[k][i, j, :]), case.idx_new)
                old_to_new_cell!(lamb_new, lamb_masked, case.idx_new)
                CanteraMechanismCodegen.chemistry_vjp!(vjp_new, cnew, lamb_new, case.temp_k, case.hour_of_day, case.mech)
                new_to_old_cell!(vjp_old, vjp_new, case.idx_new)

                @inbounds for s in 1:CTMV1Core.NSPEC
                    lambda_prev[i, j, s] = lamb_masked[s] + case.dt_sec * vjp_old[s]
                end

                grad_alpha[i, j, k] = lambda_prev[i, j, no_idx] * case.dt_sec * conv_emis * case.baseline_no[i, j]
            end
        end
        lambda_curr, lambda_prev = lambda_prev, lambda_curr
    end

    return vec(grad_alpha)
end

function main()
    case = build_case()
    nctl = case.grid.nx * case.grid.ny * NSTEPS
    alpha0 = ones(Float64, nctl)

    g_adj = adjoint_gradient(case, alpha0)

    rng = MersenneTwister(7)
    dir = randn(rng, nctl)
    dir ./= norm(dir)

    h = 1.0e-6
    jp = forward_objective(alpha0 .+ h .* dir, case)
    jm = forward_objective(alpha0 .- h .* dir, case)
    fd_dir = (jp - jm) / (2.0 * h)
    adj_dir = dot(g_adj, dir)
    rel = abs(adj_dir - fd_dir) / max(abs(fd_dir), abs(adj_dir), eps())

    out = joinpath(@__DIR__, "ctm_chemistry_vjp_prototype.txt")
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
        error("CTM chemistry VJP prototype failed: rel=$(rel), tol=$(REL_TOL)")
    end

    println("PASS: CTM chemistry VJP prototype directional check")
end

main()
