include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))
using .CanteraMechanismTypes
using .CanteraMechanismCodegen
using LinearAlgebra

const FD_STEP = 1.0e-7
const REL_TOL = 1.0e-8

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
v = zeros(Float64, length(mech.species))
v[idx[:NO]] = 1.0
v[idx[:O3]] = -0.5

J = CanteraMechanismCodegen.chemistry_jacobian(c, 293.0, 12.0, mech)
jvp = CanteraMechanismCodegen.chemistry_jvp(c, v, 293.0, 12.0, mech)
fd = (CanteraMechanismCodegen.chemistry_rhs(c .+ FD_STEP .* v, 293.0, 12.0, mech) .-
    CanteraMechanismCodegen.chemistry_rhs(c .- FD_STEP .* v, 293.0, 12.0, mech)) ./ (2.0 * FD_STEP)
rel = norm(jvp - fd) / max(norm(fd), eps())
isfinite_j = all(isfinite, J)
pass = (rel <= REL_TOL) && isfinite_j

out = joinpath(@__DIR__, "cantera_jacobian_smoke.txt")
open(out, "w") do io
    println(io, "J_SIZE = ", size(J))
    println(io, "JVP = ", join(jvp, ","))
    println(io, "FD = ", join(fd, ","))
    println(io, "FD_STEP = ", FD_STEP)
    println(io, "JVP_FD_RELERR = ", rel)
    println(io, "REL_TOL = ", REL_TOL)
    println(io, "FINITE = ", isfinite_j)
    println(io, "PASS = ", pass)
end

println("WROTE = ", out)

if !pass
    error("Jacobian/JVP smoke check failed: rel=$(rel) (tol=$(REL_TOL)), finite=$(isfinite_j)")
end

println("PASS: Jacobian/JVP check within tolerance")
