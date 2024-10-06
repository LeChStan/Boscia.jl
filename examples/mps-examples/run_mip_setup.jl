include("mip-examples.jl")

################# Read ARGS passed #######################
mode = parse(String, ARGS[1])
example = parse(String, ARGS[2])
seed = parse(Int64, ARGS[3])
alternative = parse(String, ARGS[4])
decision_function = parse(String, ARGS[5])
iterations_until_stable =  parse(Int64, ARGS[6])
μ = parse(Float64, ARGS[7])
time_limit = 1800
@show seed, dimension

################# Run example #######################
try 
    mip_boscia(mode, example, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit)
catch e 
    println(e)
    file = "mip" * str(seed) * "_" * example * "_" * mode 
    open(file * ".txt","a") do io
        println(io, e)
    end
end