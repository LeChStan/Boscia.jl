using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
# using Statistics
using LinearAlgebra
using Distributions
using CSV
using DataFrames
import MathOptInterface
const MOI = MathOptInterface
include("../example_utilities.jl")

# MIPLIB instances
# Objective function: Minimize the distance to randomely picked vertices

# Possible files
# 22433               https://miplib.zib.de/instance_details_22433.html
# n5-3                https://miplib.zib.de/instance_details_n5-3.html
# neos5               https://miplib.zib.de/instance_details_neos5.html
# pg                  https://miplib.zib.de/instance_details_pg.html
# pg5_34              https://miplib.zib.de/instance_details_pg5_34.html
# ran14x18-disj-8     https://miplib.zib.de/instance_details_ran14x18-disj-8.html
# timtab1             https://miplib.zib.de/instance_details_timtab1.html   (Takes LONG!)


# To see debug statements
#ENV["JULIA_DEBUG"] = "Boscia"

#example = "n5-3"

function build_example(example, num_v)
    file_name = string(example, ".mps")
    src = MOI.FileFormats.Model(filename=file_name)
    MOI.read_from_file(src, joinpath(@__DIR__, string("mps-files/", file_name)))

    o = SCIP.Optimizer()
    MOI.copy_to(o, src)
    MOI.set(o, MOI.Silent(), true)
    n = MOI.get(o, MOI.NumberOfVariables())
    lmo = FrankWolfe.MathOptLMO(o)

    # Disable Presolving
    MOI.set(o, MOI.RawOptimizerAttribute("presolving/maxrounds"), 0)

    #trick to push the optimum towards the interior
    vs = [FrankWolfe.compute_extreme_point(lmo, randn(n)) for _ in 1:num_v]
    # done to avoid one vertex being systematically selected
    unique!(vs)

    @assert !isempty(vs)
    b_mps = randn(n)
    max_norm = maximum(norm.(vs))

    function f(x)
        r = dot(b_mps, x)
        for v in vs
            r += 1 / (2 * max_norm) * norm(x - v)^2
        end
        return r
    end

    function grad!(storage, x)
        mul!(storage, length(vs) / max_norm * I, x)
        storage .+= b_mps
        for v in vs
            @. storage -= 1 / max_norm * v
        end
    end

    return lmo, f, grad!
end

function mip_boscia(mode, example, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit = 1800)

    o = SCIP.Optimizer()

    num_v = 0
    if example == "neos5"
        num_v = 5
    elseif example == "pg"
        num_v = 5
    elseif example == "22433"
        num_v = 20
    elseif example == "pg5_34"
        num_v = 5
    elseif example == "ran14x18-disj-8"
        num_v = 5
    elseif example == "n5-3"
        num_v = 100
    elseif example == "timtab1"
        num_v = 3
    end

    lmo, f, grad! = build_example(example, num_v)
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = example
    file_name = "mip_results"
    # Save result
    dimension = num_v
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result

end

#mip_boscia("strong_branching", "n5-3", 2, "na", "na", 1, 1e-6, 180)


