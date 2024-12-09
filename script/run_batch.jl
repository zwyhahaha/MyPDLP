include("../src/SimplePDLP.jl")
include("plot_result.jl")
using ArgParse

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--problem_folder"
    help = "The directory for instances."
    arg_type = String
    # required = true
    default = "./data"

    "--dataset"
    help = "The LP dataset."
    arg_type = String
    default = "MIPLIB"

    "--output_directory"
    help = "The directory for output files."
    arg_type = String
    # required = true
    default = "./output/solver_output"

    "--kkt_tolerance"
    help = "KKT tolerance of the solution"
    arg_type = Float64
    default = 1e-6

    "--iteration_limit"
    help = "Maximum iteration number."
    arg_type = Int64
    default = 20000

  end

  return ArgParse.parse_args(arg_parse)
end

function main()
    parsed_args = parse_command_line()
    problem_folder = parsed_args["problem_folder"]
    dataset = parsed_args["dataset"]
    output_directory = parsed_args["output_directory"]
    kkt_tolerance = parsed_args["kkt_tolerance"]
    iteration_limit = parsed_args["iteration_limit"]

    # easy_instances = ["neos5","mad","acc-tight2","enlight_hard"]
    # mid_instances = ["b-ball","graphdraw-domain","gsvm2rl3","gsvm2rl5"]
    # hard_instances = ["beasleyC1"]
    easy_instances = ["sc50a"]

    output_directory = joinpath(output_directory, dataset)
    problem_folder = joinpath(problem_folder, dataset)

    if !isdir(output_directory)
        mkpath(output_directory)
    end
    
    all_instances = readdir(problem_folder)
    all_instances = [replace(basename(instance), ".mps.gz" => "") for instance in all_instances if endswith(instance, ".mps.gz")]

    len = length(all_instances)

    for i in 96:len
        problem_name = all_instances[i]
        instance_path = joinpath(problem_folder, "$(problem_name).mps.gz")
        println("Solving $(i), $(problem_name)")

        try 
          lp = qps_reader_to_standard_form(instance_path)
          m, n = size(lp.constraint_matrix)

          if n > 5000 && m > 5000
              continue
          end
          
          for learning_rate in [0.0,0.001,0.01]
              solver_output = solve(lp, iteration_limit, kkt_tolerance, zeros(n), zeros(m), true, learning_rate)
              JLD2.jldsave(joinpath(output_directory, "$(problem_name)_$(string(learning_rate)).jld2"); solver_output)
          end
          
          plot(output_directory, "./output/figure/$(dataset)", problem_name)
        catch e
          println("Error in read $(i), $(problem_name)", e)
          continue
        end
    end
    
    write_csv(output_directory, "./output/table", dataset)

end


main()


