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

from source.agents.train_maskableppo import train_maskableppo
from source.utilities.plot_data import plot_data
from source.utilities.helpers import (detect_optimal_resources,
                                      launch_tensorboard,
                                      delete_pycache)

def train_single_alpha(configuration_path: str = None,
											 output_directory: str = None,
											 alpha: float = None,
											 resources: dict = None) -> tuple[float,
																												bool,
																												float]:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if alpha is None:
		raise ValueError("Alpha not specified.")
	if resources is None:
		raise ValueError("Resources not specified.")

	print(f"\n[Alpha {alpha:.2f}] Starting training...")

	start_time: float = time.time()

	try:
		train_maskableppo(configuration_path = configuration_path,
											output_directory = output_directory,
											alpha = alpha,
											resources = resources)

		elapsed_time: float = time.time() - start_time
	
		print(f"[Alpha {alpha:.2f}] Complete in {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)!")

		return (alpha,
            True,
            elapsed_time)
	except Exception as exception:
		elapsed_time: float = time.time() - start_time

		print(f"[Alpha {alpha:.2f}] Failed after {elapsed_time:.2f}s: {exception}")
		traceback.print_exc()

		return (alpha,
						False,
						elapsed_time)
def train_all_alphas(configuration_path: str = None,
                     output_directory: str = None,
                     alphas: list[float] = None) -> None:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if alphas is None:
		raise ValueError("Alphas not specified.")

	number_alphas: int = len(alphas)
	resources: dict = detect_optimal_resources(number_parallel = number_alphas)
	results: dict[float,
								tuple[bool,
											float]] = {}

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
		print("PARALLEL TRAINING MODE")
		print(f"Training {number_alphas} alphas with {resources['maximum_workers']} parallel workers")
		print(f"Each alpha will use {resources['number_environments']} vectorized environments")
		print(f"Total CPU utilization: {resources['maximum_workers'] * resources['number_environments']} cores")
		print("=" * 80 + "\n")

		train_arguments: list[tuple[str,
																str,
																float,
																dict]] = [(configuration_path,
																					 output_directory,
																					 alpha,
																					 resources) for alpha in alphas]
		start_time: float = time.time()

		with ProcessPoolExecutor(max_workers = resources['maximum_workers']) as executor:
			futures: dict = {executor.submit(train_single_alpha,
                                 			 *arguments): arguments[2] for arguments in train_arguments}

			for future in as_completed(futures):
				(alpha,
         success,
         elapsed_time) = future.result()
				results[alpha] = (success,
                          elapsed_time)

		total_time: float = time.time() - start_time
		successful: int = sum(1 for (success,
                                 _) in results.values() if success)
		total_training_time: float = sum(elapsed_time for (_,
                                                  		 elapsed_time) in results.values())

		print("\n" + "=" * 80)
		print("TRAINING SUMMARY")
		print("=" * 80)
		print(f"Successful: {successful}/{number_alphas} ({successful / number_alphas * 100:.2f}%)")
		print(f"Total wall time: {total_time:.2f}s ({total_time / 60:.2f} min)")
		print(f"Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f} min)")
		print(f"Speed-up factor: {total_training_time / total_time:.2f}x")
	else:
		print("SEQUENTIAL TRAINING MODE")
		print(f"Training {number_alphas} alphas with {resources['maximum_workers']} sequential workers")
		print(f"Each alpha will use {resources['number_environments']} vectorized environments")
		print(f"Total CPU utilization: {resources['maximum_workers'] * resources['number_environments']} cores")
		print("=" * 80 + "\n")

		start_time: float = time.time()

		for alpha in alphas:
			(alpha,
       success,
       elapsed_time) = train_single_alpha(configuration_path = configuration_path,
																					output_directory = output_directory,
																					alpha = alpha,
																					resources = resources)
			results[alpha] = (success,
                        elapsed_time)

		total_time: float = time.time() - start_time
		successful: int = sum(1 for (success,
																 _) in results.values() if success)
		total_training_time: float = sum(elapsed_time for (_,
																											 elapsed_time) in results.values())

		print("\n" + "=" * 80)
		print("TRAINING SUMMARY")
		print("=" * 80)
		print(f"Successful: {successful}/{number_alphas} ({successful / number_alphas * 100:.2f}%)")
		print(f"Total wall time: {total_time:.2f}s ({total_time / 60:.2f} min)")
		print(f"Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f} min)")

	print("\nPer-alpha results:")

	for alpha in sorted(results.keys()):
		(success,
		 elapsed_time) = results[alpha]
		status: str = "success" if success else "failure"

		print(f"  [{status}] Alpha {alpha:.2f}: {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)")

	print("=" * 80)

if __name__ == "__main__":
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description = "Train MaskablePPO for all alpha values (AUTO-OPTIMIZED)")

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

	train_all_alphas(configuration_path = arguments.configuration_path,
									 output_directory = arguments.output_directory,
									 alphas = arguments.alphas)
	plot_data(output_directory = arguments.output_directory,
            data_type = 'training',
            agent_label = None)
	launch_tensorboard(output_directory = arguments.output_directory,
                     port = arguments.port)
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
