module CanteraMechanismCodegen

import Main.CanteraMechanismTypes
using Dates
using ForwardDiff

export species_index_map, reaction_progress_rate, chemistry_rhs!, chemistry_rhs,
    chemistry_jacobian, chemistry_jvp, chemistry_vjp, chemistry_vjp!,
    generate_reaction_report

const AVOGADRO = 6.02214076e23

"""
Build a species -> index map from a parsed mechanism.
"""
function species_index_map(mech::CanteraMechanismTypes.Mechanism)
    return Dict(sp.name => i for (i, sp) in enumerate(mech.species))
end

function _photolysis_scale(hour_of_day)
    # Smooth diurnal scaling for photolysis in the mechanism generator.
    hour = Float64(hour_of_day)
    return max(0.0, cos(pi * (hour - 12.0) / 12.0))
end

function _photolysis_scale_legacy(dt_utc::DateTime, lat_deg, lon_deg)
    n = dayofyear(Date(dt_utc))
    hour = Dates.hour(dt_utc) + Dates.minute(dt_utc) / 60 + Dates.second(dt_utc) / 3600

    γ = 2pi / 365.0 * (n - 1 + (hour - 12) / 24)
    δ = 0.006918 - 0.399912 * cos(γ) + 0.070257 * sin(γ) - 0.006758 * cos(2γ) +
        0.000907 * sin(2γ) - 0.002697 * cos(3γ) + 0.00148 * sin(3γ)
    eot_min = 229.18 * (0.000075 + 0.001868 * cos(γ) - 0.032077 * sin(γ) - 0.014615 * cos(2γ) - 0.040849 * sin(2γ))

    time_offset_min = eot_min + 4.0 * lon_deg
    tst_min = hour * 60.0 + time_offset_min
    ha_deg = tst_min / 4.0 - 180.0

    lat = deg2rad(lat_deg)
    ha = deg2rad(ha_deg)
    μ = sin(lat) * sin(δ) + cos(lat) * cos(δ) * cos(ha)
    return max(0.0, μ)
end

function _reaction_rate_constant(rate::CanteraMechanismTypes.RateLaw, temp_k::Real, photolysis_scale)
    if rate isa CanteraMechanismTypes.ArrheniusRate
        return CanteraMechanismTypes.reaction_rate(rate, temp_k)
    elseif rate isa CanteraMechanismTypes.PhotolysisRate
        return rate.j_noon * photolysis_scale
    else
        error("Unsupported rate law type: $(typeof(rate))")
    end
end

function reaction_progress_rate(reaction::CanteraMechanismTypes.ReactionDef, c, temp_k::Real, photolysis_scale, idx::Dict{Symbol,Int})
    k = _reaction_rate_constant(reaction.rate, temp_k, photolysis_scale)
    rate = k
    for (sp, nu) in reaction.kinetic_reactants
        i = get(idx, sp, 0)
        i == 0 && continue
        rate *= c[i]^nu
    end
    return rate
end

function _accumulate!(dc, idx::Dict{Symbol,Int}, stoich::Dict{Symbol,Int}, nu_rate)
    for (sp, nu) in stoich
        i = get(idx, sp, 0)
        i == 0 && continue
        dc[i] += nu * nu_rate
    end
    return nothing
end

"""
Compute chemistry tendency from a parsed mechanism.

Inputs:
- dc: output tendency array, same length/order as mech.species
- c: concentration vector in the same ordering
- temp_k: temperature
- hour_of_day: used for the smooth photolysis scale
"""
function chemistry_rhs!(dc, c, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    fill!(dc, zero(eltype(dc)))
    idx = species_index_map(mech)
    conv = AVOGADRO / 1e6

    # Match legacy chemistry internals: compute rates in molecule/cm^3 space.
    y = clamp.(max.(c, 0.0) .* conv, 0.0, 1.0e16)
    dy = zeros(eltype(dc), length(mech.species))

    photolysis_scale = _photolysis_scale(hour_of_day)
    for rxn in mech.reactions
        prog = reaction_progress_rate(rxn, y, temp_k, photolysis_scale, idx)
        _accumulate!(dy, idx, rxn.reactants, -prog)
        _accumulate!(dy, idx, rxn.products, +prog)
    end

    dc .= dy ./ conv
    return nothing
end

"""
Chemistry RHS using legacy-like solar geometry for photolysis scaling.
"""
function chemistry_rhs!(dc, c, temp_k::Real, dt_utc::DateTime, lat_deg, lon_deg, mech::CanteraMechanismTypes.Mechanism)
    fill!(dc, zero(eltype(dc)))
    idx = species_index_map(mech)
    conv = AVOGADRO / 1e6

    y = clamp.(max.(c, 0.0) .* conv, 0.0, 1.0e16)
    dy = zeros(eltype(dc), length(mech.species))

    photolysis_scale = _photolysis_scale_legacy(dt_utc, lat_deg, lon_deg)
    for rxn in mech.reactions
        prog = reaction_progress_rate(rxn, y, temp_k, photolysis_scale, idx)
        _accumulate!(dy, idx, rxn.reactants, -prog)
        _accumulate!(dy, idx, rxn.products, +prog)
    end

    dc .= dy ./ conv
    return nothing
end

"""
Pure chemistry RHS that returns a fresh tendency vector.

This wrapper is convenient for ForwardDiff-based Jacobian generation.
"""
function chemistry_rhs(c, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    dc = zeros(eltype(c), length(mech.species))
    chemistry_rhs!(dc, c, temp_k, hour_of_day, mech)
    return dc
end

"""
Jacobian of the generated chemistry RHS with respect to state c.
"""
function chemistry_jacobian(c, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    f = x -> chemistry_rhs(x, temp_k, hour_of_day, mech)
    return ForwardDiff.jacobian(f, c)
end

"""
Jacobian-vector product for the generated chemistry RHS.
"""
function chemistry_jvp(c, v, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    J = chemistry_jacobian(c, temp_k, hour_of_day, mech)
    return J * v
end

"""
Vector-Jacobian product (reverse-mode primitive): J' * lambda.
"""
function chemistry_vjp(c, lambda, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    J = chemistry_jacobian(c, temp_k, hour_of_day, mech)
    return transpose(J) * lambda
end

"""
In-place Vector-Jacobian product (reverse-mode primitive): out = J' * lambda.
"""
function chemistry_vjp!(out, c, lambda, temp_k::Real, hour_of_day, mech::CanteraMechanismTypes.Mechanism)
    out .= chemistry_vjp(c, lambda, temp_k, hour_of_day, mech)
    return nothing
end

function generate_reaction_report(mech::CanteraMechanismTypes.Mechanism; maxlines::Int=20)
    idx = species_index_map(mech)
    lines = String[]
    push!(lines, "Mechanism reaction report")
    push!(lines, "species_count=$(length(mech.species))")
    push!(lines, "reaction_count=$(length(mech.reactions))")
    nshow = min(length(mech.reactions), maxlines)
    for rxn in mech.reactions[1:nshow]
        push!(lines, string(rxn.label, " | ", rxn.reactants, " => ", rxn.products, " | ", typeof(rxn.rate)))
    end
    return join(lines, "\n")
end

end # module
