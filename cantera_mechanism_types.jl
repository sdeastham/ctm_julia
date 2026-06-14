module CanteraMechanismTypes

export Mechanism, SpeciesDef, ReactionDef, ArrheniusRate, PhotolysisRate,
    parse_cantera_yaml, reaction_rate

using YAML

"""
Minimal internal representation for a Cantera-style chemical mechanism.

This is a scaffold for a future parser/code generator. It is intentionally
small and mechanism-agnostic so we can plug it into the CTM chemistry layer.
"""

struct SpeciesDef
    name::Symbol
    composition::Dict{Symbol,Int}
    charge::Int
end

abstract type RateLaw end

struct ArrheniusRate <: RateLaw
    A::Float64
    b::Float64
    Ea_over_R_K::Float64
end

struct PhotolysisRate <: RateLaw
    j_noon::Float64
end

struct ReactionDef
    reactants::Dict{Symbol,Int}
    products::Dict{Symbol,Int}
    kinetic_reactants::Dict{Symbol,Int}
    rate::RateLaw
    reversible::Bool
    label::Symbol
end

struct Mechanism
    species::Vector{SpeciesDef}
    reactions::Vector{ReactionDef}
end

_get(d, key, default=nothing) = haskey(d, key) ? d[key] : default

function _as_symbol(x)
    x isa Symbol && return x
    x isa AbstractString && return Symbol(strip(x))
    return Symbol(string(x))
end

function _parse_term(term::AbstractString)
    s = strip(term)
    isempty(s) && error("Empty species term in reaction equation")

    parts = split(s)
    if length(parts) == 1
        return 1, _as_symbol(parts[1])
    elseif length(parts) == 2
        coeff = parse(Int, parts[1])
        return coeff, _as_symbol(parts[2])
    else
        error("Unsupported reaction term '$term'. Expected 'A' or '2 A'.")
    end
end

function _parse_side(side::AbstractString)
    side = strip(side)
    isempty(side) && return Dict{Symbol,Int}()

    stoich = Dict{Symbol,Int}()
    for term in split(side, "+")
        coeff, sp = _parse_term(term)
        stoich[sp] = get(stoich, sp, 0) + coeff
    end
    return stoich
end

function _parse_stoich_dict(d)
    stoich = Dict{Symbol,Int}()
    for (k, v) in d
        stoich[_as_symbol(k)] = Int(v)
    end
    return stoich
end

function _parse_equation(eq::AbstractString)
    if occursin("=>", eq)
        lhs, rhs = split(eq, "=>"; limit=2)
        reversible = false
    elseif occursin("<=>", eq)
        lhs, rhs = split(eq, "<=>"; limit=2)
        reversible = true
    else
        error("Reaction equation must contain '=>' or '<=>': $eq")
    end

    return _parse_side(lhs), _parse_side(rhs), reversible
end

function _parse_species(spec_entry)
    if spec_entry isa AbstractString
        return SpeciesDef(_as_symbol(spec_entry), Dict{Symbol,Int}(), 0)
    elseif spec_entry isa AbstractDict
        name = _as_symbol(_get(spec_entry, "name", _get(spec_entry, :name)))
        comp_raw = _get(spec_entry, "composition", _get(spec_entry, :composition, Dict()))
        comp = Dict{Symbol,Int}()
        for (k, v) in comp_raw
            comp[_as_symbol(k)] = Int(v)
        end
        charge = Int(_get(spec_entry, "charge", _get(spec_entry, :charge, 0)))
        return SpeciesDef(name, comp, charge)
    else
        error("Unsupported species entry format: $(typeof(spec_entry))")
    end
end

function _parse_rate(rate_raw, reaction_entry)
    reaction_type = _get(reaction_entry, "type", _get(reaction_entry, :type, "arrhenius"))
    rate_dict = rate_raw isa AbstractDict ? rate_raw : Dict{Any,Any}()
    A = Float64(_get(rate_dict, "A", _get(rate_dict, :A, 0.0)))
    b = Float64(_get(rate_dict, "b", _get(rate_dict, :b, 0.0)))
    Ea = Float64(_get(rate_dict, "Ea", _get(rate_dict, :Ea, 0.0)))

    if lowercase(string(reaction_type)) == "photolysis"
        return PhotolysisRate(A)
    end

    return ArrheniusRate(A, b, Ea)
end

"""
Parse a Cantera YAML mechanism using the YAML.jl package.

Supported subset:
- top-level `species` or `phases[1].species`
- top-level `reactions`
- `equation`, `rate-constant`, `label`, `type`
"""
function parse_cantera_yaml(path::AbstractString)
    doc = YAML.load_file(path)

    species_raw = _get(doc, "species", _get(doc, :species, nothing))
    if species_raw === nothing
        phases = _get(doc, "phases", _get(doc, :phases, nothing))
        phases === nothing && error("No species or phases section found in $path")
        phase1 = phases[1]
        species_raw = _get(phase1, "species", _get(phase1, :species, nothing))
    end
    species_raw === nothing && error("No species list found in $path")

    reactions_raw = _get(doc, "reactions", _get(doc, :reactions, nothing))
    reactions_raw === nothing && error("No reactions list found in $path")

    species = SpeciesDef[]
    for sp in species_raw
        push!(species, _parse_species(sp))
    end

    reactions = ReactionDef[]
    for rxn in reactions_raw
        eq = string(_get(rxn, "equation", _get(rxn, :equation)))
        lhs, rhs, reversible = _parse_equation(eq)
        rate_raw = _get(rxn, "rate-constant", _get(rxn, Symbol("rate-constant"), Dict()))
        rate = _parse_rate(rate_raw, rxn)
        kinetic_reactants_raw = _get(rxn, "kinetic-reactants", _get(rxn, Symbol("kinetic-reactants"), nothing))
        kinetic_reactants = kinetic_reactants_raw === nothing ? lhs : _parse_stoich_dict(kinetic_reactants_raw)
        label = _as_symbol(_get(rxn, "label", _get(rxn, :label, "reaction_$(length(reactions)+1)")))
        push!(reactions, ReactionDef(lhs, rhs, kinetic_reactants, rate, reversible, label))
    end

    return Mechanism(species, reactions)
end

"""
Backward-compatible alias for the parser entry point.
"""
function parse_cantera_yaml_stub(path::AbstractString)
    return parse_cantera_yaml(path)
end

"""
Return the scalar rate constant for a reaction given temperature.

In this v1 CTM alignment path, `Ea_over_R_K` is interpreted in the same
legacy convention as the existing chemistry code, i.e. k = A*(T/300)^b*exp(-Ea/T).
For photolysis reactions, temperature is ignored in the current scaffold.
"""
function reaction_rate(rate::ArrheniusRate, temp_k::Real)
    return rate.A * (temp_k / 300.0)^rate.b * exp(-rate.Ea_over_R_K / temp_k)
end

reaction_rate(rate::PhotolysisRate, temp_k::Real) = rate.j_noon

end # module
