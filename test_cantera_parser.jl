include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
using .CanteraMechanismTypes

path = joinpath(@__DIR__, "cantera_mechanism_example.yaml")
mech = CanteraMechanismTypes.parse_cantera_yaml(path)

out = joinpath(@__DIR__, "cantera_parser_smoke.txt")
open(out, "w") do io
	println(io, "SPECIES_COUNT = ", length(mech.species))
	println(io, "REACTION_COUNT = ", length(mech.reactions))
	println(io, "FIRST_SPECIES = ", mech.species[1].name)
	println(io, "FIRST_REACTION = ", mech.reactions[1].label, " | ", mech.reactions[1].reactants, " => ", mech.reactions[1].products)
	println(io, "FIRST_RATE_TYPE = ", typeof(mech.reactions[1].rate))
end

println("WROTE = ", out)
