using Debugger

# # 模拟命令行参数
# ARGS = ["--problem_folder=./data", "--output_directory=./output/solver_output", "--problem_name=neos5", "--kkt_tolerance=1e-6", "--iteration_limit=20000", "--online_scaling"]

include("../script/run_problem.jl")
@enter main()