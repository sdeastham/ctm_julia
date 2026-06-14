const TEST_SCRIPTS = [
    joinpath(@__DIR__, "test_cantera_parser.jl"),
    joinpath(@__DIR__, "test_cantera_codegen.jl"),
    joinpath(@__DIR__, "test_cantera_jacobian.jl"),
    joinpath(@__DIR__, "test_cantera_vs_old_chemistry.jl"),
    joinpath(@__DIR__, "test_cantera_vjp.jl"),
    joinpath(@__DIR__, "test_ctm_chemistry_vjp_prototype.jl"),
    joinpath(@__DIR__, "test_transport_adjoint_prototype.jl"),
    joinpath(@__DIR__, "test_ctm_full_adjoint_prototype.jl"),
]

function main()
    npass = 0
    nfail = 0
    failed = String[]

    for script in TEST_SCRIPTS
        println("\n=== RUN ", script, " ===")
        try
            include(script)
            npass += 1
            println("=== PASS ", script, " ===")
        catch err
            nfail += 1
            push!(failed, script)
            println("=== FAIL ", script, " ===")
            showerror(stdout, err, catch_backtrace())
            println()
        end
    end

    println("\nSUMMARY: pass=", npass, " fail=", nfail)
    if nfail > 0
        println("FAILED SCRIPTS:")
        for f in failed
            println(" - ", f)
        end
        error("run_cantera_tests.jl failed")
    end
end

main()
