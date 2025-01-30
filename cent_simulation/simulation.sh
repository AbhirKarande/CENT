# Pipeline Parallel
threads=$1

# python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1
# python3 run_sim.py --model Llama2-13B --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1
# python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1

# # Model Parallel
# python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1
python3 run_sim.py --model Llama2-13B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1
python3 run_sim.py --model Llama2-70B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap 1

# Long Context
python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --seqlen 2304 6400 14592 30976

