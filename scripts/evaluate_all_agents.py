import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',
                        category = UserWarning)
warnings.filterwarnings('ignore',
                        category = DeprecationWarning)
warnings.filterwarnings('ignore',
                        category = FutureWarning)
warnings.filterwarnings('ignore',
                        module = 'google.protobuf')
warnings.filterwarnings('ignore',
                        module = 'tensorboard')
warnings.filterwarnings('ignore',
                        module = 'pkg_resources')

import argparse
import atexit
import sys
import torch
import time
import traceback

from concurrent.futures import (ProcessPoolExecutor,
																as_completed)

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(FILE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.agents.evaluate_random import evaluate_random
from source.agents.evaluate_maskableppo import evaluate_maskableppo
from source.agents.evaluate_pomcp import evaluate_pomcp
from source.utilities.plot_data import plot_data
from source.utilities.helpers import (detect_optimal_resources,
                                      load_configuration,
																			delete_pycache)

def evaluate_single_agent(configuration_path: str = None,
													output_directory: str = None,
													agent_label: str = None,
													alpha: float = None,
													resources: dict = None) -> tuple[str,
																													 bool,
																													 float]:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if agent_label is None:
		raise ValueError("Agent label not specified.")
	if alpha is None:
		raise ValueError("Alpha not specified.")
	if resources is None:
		raise ValueError("Resources not specified.")

	print(f"\n[{agent_label}] Starting evaluation...")

	start_time: float = time.time()

	try:
		if agent_label.startswith('random'):
			evaluate_random(configuration_path = configuration_path,
											output_directory = output_directory,
											alpha = alpha,
											resources = resources)
		if agent_label.startswith('maskableppo'):
			evaluate_maskableppo(configuration_path = configuration_path,
													 output_directory = output_directory,
													 alpha = alpha,
													 resources = resources)
		if agent_label.startswith('pomcp'):
			evaluate_pomcp(configuration_path = configuration_path,
										 output_directory = output_directory,
										 alpha = alpha,
										 resources = resources)

		elapsed_time: float = time.time() - start_time

		print(f"[{agent_label}] Complete in {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)!")

		return (agent_label,
						True,
						elapsed_time)
	except Exception as exception:
		elapsed_time: float = time.time() - start_time

		print(f"[{agent_label}] Failed after {elapsed_time:.2f}s: {exception}")
		traceback.print_exc()

		return (agent_label,
						False,
						elapsed_time)
def evaluate_all_agents(configuration_path: str = None,
                        output_directory: str = None,
                        alphas: list[float] = None,
                        agent_types: dict = None) -> None:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if alphas is None:
		raise ValueError("Alphas not specified.")
	if agent_types is None:
		raise ValueError("Agent types not specified.")

	evaluation_arguments: list[tuple[str,
																	 str,
																	 str,
																	 float,
																	 dict]] = []
	results: dict[str,
                tuple[bool,
                      float]] = {}
	agent_counts: list[str] = []
	number_agents: int = 0

	for (agent_type,
       enabled) in agent_types.items():
		if enabled:
			count: int = len(alphas)
			number_agents += count

			agent_counts.append(f"{count} {agent_type.capitalize()} agent(s)")

			for alpha in alphas:
				agent_label = f"{agent_type} (alpha = {alpha:.2f})"

				evaluation_arguments.append((configuration_path,
																		 output_directory,
																		 agent_label,
																		 alpha,
																		 None))

	resources: dict = detect_optimal_resources(number_parallel = number_agents)
	maximum_workers: int = resources['maximum_workers']
	agents: str = " and ".join(agent_counts)
	evaluation_arguments = [evaluation_argument[:-1] + (resources,) for evaluation_argument in evaluation_arguments]

	print("=" * 80)
	print("AUTO-DETECTED SYSTEM RESOURCES")
	print("=" * 80)
	print(f"CPU Cores (Physical): {resources['cpu_physical']}")
	print(f"CPU Cores (Logical): {resources['cpu_logical']}")
	print(f"GPU Available: {resources['gpu_available']}")

	if resources['gpu_available']:
		print(f"GPU Device: {torch.cuda.get_device_name(0)}")

	print(f"\nOptimal Configuration:")
	print(f" - Device: {resources['device']}")
	print("=" * 80 + "\n")

	if resources['maximum_workers'] > 1:
		print("PARALLEL EVALUATION MODE")
		print(f"Evaluating {agents} with {resources['maximum_workers']} parallel workers")
		print(f"Each agent will use {resources['number_environments']} vectorized environments")
		print(f"Total CPU utilization: {resources['maximum_workers'] * resources['number_environments']} cores")
		print("=" * 80 + "\n")

		start_time: float = time.time()

		with ProcessPoolExecutor(max_workers = maximum_workers) as executor:
			futures = {executor.submit(evaluate_single_agent,
                              	 *arguments): arguments[2] for arguments in evaluation_arguments}

			for future in as_completed(futures):
				(agent_label,
         success,
         elapsed_time) = future.result()
				results[agent_label] = (success,
                            		elapsed_time)

		total_time: float = time.time() - start_time
	else:
		print("SEQUENTIAL EVALUATION MODE")
		print(f"Evaluating {agents} with {maximum_workers} sequential worker(s)")
		print(f"Each agent will use {resources['number_environments']} vectorized environment(s)")
		print(f"Total CPU utilization: {maximum_workers * resources['number_environments']} cores")
		print("=" * 80 + "\n")

		start_time: float = time.time()

		for arguments in evaluation_arguments:
			(agent_label,
    	 success,
       elapsed_time) = evaluate_single_agent(*arguments)
			results[agent_label] = (success,
                           		elapsed_time)

		total_time: float = time.time() - start_time

	successful: int = sum(1 for (success,
																_) in results.values() if success)
	total_evaluation_time: float = sum(elapsed_time for (_,
																												elapsed_time) in results.values())

	print("\n" + "=" * 80)
	print("EVALUATION SUMMARY")
	print("=" * 80)
	print(f"Successful: {successful}/{number_agents} ({successful / number_agents * 100:.2f}%)")
	print(f"Total wall time: {total_time:.2f}s ({total_time / 60:.2f} min)")
	print(f"Total evaluation time: {total_evaluation_time:.2f}s ({total_evaluation_time / 60:.2f} min)")

	if resources['maximum_workers'] > 1:
		print(f"Speed-up factor: {total_evaluation_time / total_time:.2f}x")

	print("\nPer-agent results:")

	for agent_label in sorted(results.keys()):
		(success,
     elapsed_time) = results[agent_label]
		status: str = "success" if success else "failure"

		print(f"  [{status}] {agent_label}: {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)")

	print("=" * 80)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Evaluate agents (MaskablePPO, POMCP, Random)")

	parser.add_argument('--configuration_path',
										type = str,
											default = os.path.join(PROJECT_ROOT,
                                             'configuration.yaml'),
											help = "Path to configuration file")
	parser.add_argument('--output_directory',
											type = str,
											default = os.path.join(PROJECT_ROOT,
                                             'results'),
											help = "Output directory")
	parser.add_argument('--alphas',
											type = float,
											nargs = '+',
											default = [0.0,
                                 0.25,
                                 0.5,
                                 0.75,
                                 1.0],
											help = "List of alpha values")
	parser.add_argument('--port',
											type = int,
											default = 6006,
											help = "Tensorboard port")

	arguments: argparse.Namespace = parser.parse_args()
	configuration: dict = load_configuration(arguments.configuration_path)
	evaluation_configuration: dict = configuration['evaluation']
	random: bool = evaluation_configuration['random']
	maskableppo: bool = evaluation_configuration['maskableppo']
	pomcp: bool = evaluation_configuration['pomcp']
	agent_types: dict = {"random": random,
											 "maskableppo": maskableppo,
											 "pomcp": pomcp}

	evaluate_all_agents(configuration_path = arguments.configuration_path,
											output_directory = arguments.output_directory,
											alphas = arguments.alphas,
           						agent_types = agent_types)

	for (agent_label,
       enabled) in agent_types.items():
		if enabled:
			plot_data(output_directory = arguments.output_directory,
								data_type = 'evaluation',
								agent_label = agent_label)

	atexit.register(lambda: delete_pycache([os.path.join(PROJECT_ROOT,
																											 'source',
																											 'agents'),
																					os.path.join(PROJECT_ROOT,
																											 'source',
																											 'environment'),
																					os.path.join(PROJECT_ROOT,
																											 'source',
																											 'updates'),
																					os.path.join(PROJECT_ROOT,
																											 'source',
																											 'utilities')]))
