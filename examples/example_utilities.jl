using DataFrames, CSV



function build_branching_strategy(
    bounded_lmo, 
    mode, 
    alternative, 
    decision_function, 
    iterations_until_stable,
    μ
)   
    if mode == "hierarchy"
        
        branching_strategy = Boscia.HIERARCHY_PSEUDO_COST(
            iterations_until_stable,
            alternative,
            bounded_lmo,
            μ,
            decision_function
        )
        settings = "hierarchy_" * string(iterations_until_stable) * "_" * alternative * "_" * decision_function       
    elseif mode == "pseudocost"
        branching_strategy = Boscia.PSEUDO_COST(
            iterations_until_stable,
            alternative,
            bounded_lmo,
            μ,
            decision_function
        ) 
        settings = "pseudocost_" * string(iterations_until_stable) * "_" * alternative * "_" * decision_function   
    elseif mode == "most_infeasible"
        branching_strategy = Bonobo.MOST_INFEASIBLE()
        settings = mode
    elseif mode =="largest_gradient"
        branching_strategy = Boscia.LargestGradient()
        settings = mode
    elseif mode == "largest_most_infeasible_gradient"
        branching_strategy = Boscia.LargestMostInfeasibleGradient()
        settings = mode
    elseif mode == "strong_branching"
        blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
        branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
        MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
        settings = mode
    end
    return branching_strategy, settings
end

function save_results(
    result::Dict{Symbol, Any},
    settings::String,
    μ,
    example_name::String,
    seed,
    dimension,
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

    l11[:, :settings] .= settings
    l11[:, :example_name] .= example_name
    l11[:, :dimension] .= dimension
    l11[:, :seed] .= seed
    l11[:, :weight] .= μ

    l22 = DataFrame(l22)
    l22[:, :settings] .= settings
    l22[:, :example_name] .= example_name
    l22[:, :seed] .= seed
    l22[:, :dimension] .= dimension
    l22[:, :weight] .= μ

    l33 = DataFrame(l33)
    l33[:, :settings] .= settings
    l33[:, :example_name] .= example_name
    l33[:, :seed] .= seed
    l33[:, :dimension] .= dimension
    l33[:, :weight] .= μ

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