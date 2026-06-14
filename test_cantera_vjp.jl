include(joinpath(@__DIR__, "cantera_mechanism_types.jl"))
include(joinpath(@__DIR__, "cantera_mechanism_codegen.jl"))
using .CanteraMechanismTypes
using .CanteraMechanismCodegen
using LinearAlgebra

const DOT_TOL = 1.0e-11

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
v[idx[:OH]] = 0.25

lambda = zeros(Float64, length(mech.species))
lambda[idx[:NO2]] = 1.0
lambda[idx[:HO2]] = -0.2
lambda[idx[:HNO3]] = 0.4

jvp = CanteraMechanismCodegen.chemistry_jvp(c, v, 293.0, 12.0, mech)
vjp = CanteraMechanismCodegen.chemistry_vjp(c, lambda, 293.0, 12.0, mech)

left = dot(lambda, jvp)
right = dot(vjp, v)
rel = abs(left - right) / max(abs(left), abs(right), eps())

out = joinpath(@__DIR__, "cantera_vjp_smoke.txt")
open(out, "w") do io
    println(io, "DOT_LEFT = ", left)
    println(io, "DOT_RIGHT = ", right)
    println(io, "DOT_RELERR = ", rel)
    println(io, "PASS = ", rel <= DOT_TOL)
end

println("WROTE = ", out)

if rel > DOT_TOL
    error("VJP dot-product check failed: rel=$(rel), tol=$(DOT_TOL)")
end

println("PASS: chemistry VJP dot-product identity within tolerance")
