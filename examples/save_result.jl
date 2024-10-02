using DataFrames, CSV


function save_results(
    result::Dict{Symbol, Any},
    settings::String,
    example_name::String,
    seed,
    file_name::String,
    over_write::Bool
    )
    
    seed = string(seed)
    l1 = []# save all keys with one entry only
    l2 = []# save all vector results of length equal to that of result[:list_ub]
    l3 = []# save all vector results of length equal to that of lmo_calls_per_layer
    for key in keys(result)
        if string(key) in ["dual_bound","dual_gap","heu_lmo_calls","lmo_calls","number_nodes","primal_objective","rel_dual_gap","status","total_time_in_sec"]
            push!(l1, key)
        elseif string(key) in ["global_tightenings", "list_active_set_size", "list_discarded_set_size",
            "list_lb","list_lmo_calls_acc","list_num_nodes","list_time","list_ub","local_potential_tightenings","local_tightenings","node_level"]
            push!(l2, key)
        elseif string(key) in ["active_set_size_per_layer", "discarded_set_size_per_layer", "lmo_calls_per_layer"]
            push!(l3, key)
        elseif string(key) != "raw_solution"
            println(key, " has not been saved ")
        end
    end
    l11 = Dict(string(key) => result[key] for key in l1)
    l22 = Dict(string(key) => result[key] for key in l2)
    l33 = Dict(string(key) => result[key] for key in l3)
    l11 = DataFrame(l11)
    l11[:, :example_name] .= example_name
    l11[:, :seed] .= seed
    l11[:, :settings] .= settings

    l22 = DataFrame(l22)
    l22[:, :settings] .= settings
    l22[:, :example_name] .= example_name
    l22[:, :seed] .= seed

    l33 = DataFrame(l33)
    l33[:, :settings] .= settings
    l33[:, :example_name] .= example_name
    l33[:, :seed] .= seed

    file_name1 = "./results/" * file_name * "_summary.csv"

    
    if over_write# will always over write file if true
        append = false
    else
        if isfile(file_name1)# using this method the first line of the file will have column names
            append = true
        else
            append = false
        end
    end
    CSV.write(file_name1, l11, append= append)

    file_name2 = "./results/" * file_name * ".csv"

    CSV.write(file_name2, l22, append= append)
    file_name3 = "./results/" * file_name * "_layers.csv"
    CSV.write(file_name3, l33, append= append)
end