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
    # required = true
    default = "./output/solver_output/MIPLIB"

    "--figure_directory"
    help = "The directory for figures."
    arg_type = String
    # required = true
    default = "./output/figure/MIPLIB"

  end

  return ArgParse.parse_args(arg_parse)
end

function plot(directory_for_solver_output,
              figure_directory,
              problem_name)

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
  
  if !isdir(figure_directory)
    mkpath(figure_directory)
  end

  Plots.savefig(kkt_plt, joinpath(figure_directory, "$(problem_name).png"))
end

function write_csv(directory_for_solver_output,
                   table_directory,
                   dataset)
  csv_file = joinpath(table_directory, "$(dataset).csv")

  open(csv_file, "w") do io
    println(io, "name, lr, n ,m, iterations, time, kkt_error, status")

    for file in readdir(directory_for_solver_output)

      if endswith(file, ".jld2")

        problem_name = replace(basename(file), ".jld2" => "")
        name, lr = split(problem_name, "_")
        solver_output = JLD2.load(joinpath(directory_for_solver_output, file))
        solver_output = solver_output["solver_output"]
        
        kkt_error = solver_output.iteration_stats[:,"kkt_error"]
        last_kkt = kkt_error[end]
        n = solver_output.primal_size
        m = solver_output.dual_size
        iteration = solver_output.iteration
        time = solver_output.time
        status = solver_output.status
        
        println(io, "$(name),$(lr),$(n),$(m),$(iteration),$(time),$(last_kkt),$(status)")
      end
    end

  end
end

function main()
  parsed_args = parse_command_line()
  directory_for_solver_output = parsed_args["directory_for_solver_output"]
  figure_directory = parsed_args["figure_directory"]

  for file in readdir(directory_for_solver_output)
    if endswith(file, ".jld2")
      problem_name = replace(basename(file), ".jld2" => "")
      name, lr = split(problem_name, "_")
      plot(directory_for_solver_output, figure_directory, name)
    end
  end
end

# main()
