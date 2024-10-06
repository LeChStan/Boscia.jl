include("low_dim_in_high_dim.jl")

################# Read ARGS passed #######################
mode = parse(String, ARGS[1])
low_dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
alternative = parse(String, ARGS[4])
decision_function = parse(String, ARGS[5])
iterations_until_stable =  parse(Int64, ARGS[6])
μ = parse(Float64, ARGS[7])
high_dimension = parse(Int64, ARGS[8])
time_limit = 1800
@show seed, dimension

################# Run example #######################
try 
    low_dim_high_dim_boscia(mode, low_dimension, high_dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit)
catch e 
    println(e)
    file = "low_dim_high_dim_" * str(seed) * "_" * str(dimension) * "_" * mode 
    open(file * ".txt","a") do io
        println(io, e)
    end
end