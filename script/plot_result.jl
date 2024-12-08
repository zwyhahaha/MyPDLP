import Plots
import JLD2
using ArgParse
include("../src/SimplePDLP.jl")

# @assert length(ARGS) == 3
# result_folder = ARGS[1]
# figure_directory = ARGS[2]
# problem_name = ARGS[3]

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--directory_for_solver_output"
    help = "The directory for solver output."
    arg_type = String
    required = true

    "--figure_directory"
    help = "The directory for figures."
    arg_type = String
    required = true


    "--problem_name"
    help = "The instance to plot."
    arg_type = String
    default = "neos5"
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()
  directory_for_solver_output = parsed_args["directory_for_solver_output"]
  figure_directory = parsed_args["figure_directory"]
  problem_name = parsed_args["problem_name"]
  
  kkt_plt = Plots.plot()
  
  for file in readdir(directory_for_solver_output)
    if startswith(file, problem_name) && endswith(file, ".jld2")
      solver_output = JLD2.load(joinpath(directory_for_solver_output, file))
      solver_output = solver_output["solver_output"]
      
      kkt_error = solver_output.iteration_stats[:,"kkt_error"]
      
      Plots.plot!(
        kkt_plt,
        1:5:(5*length(kkt_error)),
        kkt_error,
        linewidth=2,
        xlabel = "Iterations",
        ylabel = "KKT Residual",
        xguidefontsize=14,
        yaxis=:log,
        yguidefontsize=14,
        label=replace(basename(file), ".jld2" => ""),
        legend=:topright,
        title="$(problem_name)",
        titlefontsize=16,
        grid=true,
        framestyle=:box
      )
    end
  end
  
  Plots.savefig(kkt_plt, joinpath(figure_directory, "$(problem_name).png"), dpi=300)
end

main()
