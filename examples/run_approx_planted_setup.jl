include("approx_planted_point.jl")

################# Read ARGS passed #######################
mode = parse(String, ARGS[1])
dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
alternative = parse(String, ARGS[4])
decision_function = parse(String, ARGS[5])
iterations_until_stable =  parse(Int64, ARGS[6])
μ = parse(Float64, ARGS[7])
set = parse(String, ARGS[8])
time_limit = 1800
@show seed, dimension

################# Run example #######################
try 
    approx_planted_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ, set, time_limit)
catch e 
    println(e)
    file = "approx_planted_" * str(seed) * "_" * str(dimension) 
    open(file * ".txt","a") do io
        println(io, e)
    end
end