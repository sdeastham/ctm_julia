include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))
using .CanteraMechanismTypes
using .CanteraMechanismCodegen

mech = CanteraMechanismTypes.parse_cantera_yaml(joinpath(@__DIR__, "cantera_mechanism_example.yaml"))
idx = CanteraMechanismCodegen.species_index_map(mech)

c = zeros(Float64, length(mech.species))
c[idx[:NO]] = 3.0e-7
c[idx[:NO2]] = 5.0e-7
c[idx[:O3]] = 1.1e-6
c[idx[:OH]] = 1.5e-12
c[idx[:HO2]] = 8.0e-10
c[idx[:VOC]] = 8.0e-7
c[idx[:RO2]] = 1.5e-10
c[idx[:HCHO]] = 8.0e-8
c[idx[:CO]] = 6.0e-6
c[idx[:HNO3]] = 8.0e-9

# Use a daytime hour so photolysis reactions are active.
dc = zeros(Float64, length(mech.species))
CanteraMechanismCodegen.chemistry_rhs!(dc, c, 293.0, 12.0, mech)

out = joinpath(@__DIR__, "cantera_codegen_smoke.txt")
open(out, "w") do io
    println(io, CanteraMechanismCodegen.generate_reaction_report(mech; maxlines=5))
    println(io, "\nDC_SUMMARY")
    println(io, "NO=", dc[idx[:NO]])
    println(io, "NO2=", dc[idx[:NO2]])
    println(io, "O3=", dc[idx[:O3]])
    println(io, "OH=", dc[idx[:OH]])
    println(io, "HO2=", dc[idx[:HO2]])
    println(io, "finite=", all(isfinite, dc))
end

println("WROTE = ", out)
