include(joinpath(@__DIR__, "ctm_v1_core.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))
using .CTMV1Core
using .CanteraMechanismTypes
using .CanteraMechanismCodegen
using Dates
using LinearAlgebra

const REL_TOL = 1.0e-12
const ABS_TOL = 1.0e-24

mech = CanteraMechanismTypes.parse_cantera_yaml(joinpath(@__DIR__, "cantera_mechanism_example.yaml"))
idx_new = CanteraMechanismCodegen.species_index_map(mech)
idx_old = CTMV1Core.SIDX

# Shared chemistry state in old species ordering.
c_old = CTMV1Core.default_background_vector()
c_old[idx_old[:NO]] = 3.0e-7
c_old[idx_old[:NO2]] = 5.0e-7
c_old[idx_old[:O3]] = 1.1e-6
c_old[idx_old[:OH]] = 1.5e-12
c_old[idx_old[:HO2]] = 8.0e-10
c_old[idx_old[:VOC]] = 8.0e-7
c_old[idx_old[:RO2]] = 1.5e-10
c_old[idx_old[:HCHO]] = 8.0e-8
c_old[idx_old[:CO]] = 6.0e-6
c_old[idx_old[:HNO3]] = 8.0e-9

# Legacy tendency.
dc_old = zeros(Float64, length(c_old))
CTMV1Core.chemistry_tendency_cell!(dc_old, c_old, DateTime(2015, 1, 1, 12, 0, 0), 293.0, 54.0, -2.0)

# New generated tendency.
c_new = zeros(Float64, length(mech.species))
for (sp, iold) in idx_old
    if haskey(idx_new, sp)
        c_new[idx_new[sp]] = c_old[iold]
    end
end

dc_new = zeros(Float64, length(mech.species))
CanteraMechanismCodegen.chemistry_rhs!(dc_new, c_new, 293.0, DateTime(2015, 1, 1, 12, 0, 0), 54.0, -2.0, mech)

shared = collect(intersect(Set(keys(idx_old)), Set(keys(idx_new))))
sort!(shared; by=string)
old_vec = [dc_old[idx_old[s]] for s in shared]
new_vec = [dc_new[idx_new[s]] for s in shared]
diff = new_vec .- old_vec

rel = norm(diff) / max(norm(old_vec), eps())
max_abs = maximum(abs.(diff))

out = joinpath(@__DIR__, "cantera_vs_old_chemistry_check.txt")
open(out, "w") do io
    println(io, "Shared species count = ", length(shared))
    println(io, "Relative L2 mismatch = ", rel)
    println(io, "Max abs mismatch = ", max_abs)
    println(io, "species,old,new,diff")
    for k in eachindex(shared)
        println(io, string(shared[k], ",", old_vec[k], ",", new_vec[k], ",", diff[k]))
    end
end

println("WROTE = " * out)

if !(rel <= REL_TOL && max_abs <= ABS_TOL)
    error("Old-vs-new chemistry mismatch too large: rel=$(rel) (tol=$(REL_TOL)), max_abs=$(max_abs) (tol=$(ABS_TOL))")
end

println("PASS: old-vs-new chemistry parity within tolerance")
