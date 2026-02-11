import psutil
import torch
import yaml
import heapq
import os
import shutil
import subprocess
import time
import webbrowser

import pandas as pd
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

def detect_cpu_cores() -> int:
	cpu_physical: int = psutil.cpu_count(logical = False)
	cpu_logical: int = psutil.cpu_count(logical = True)

	return (cpu_physical,
					cpu_logical)
def detect_optimal_resources(number_parallel: int = None) -> dict:
	if number_parallel is None:
		raise ValueError("Number of parallel processes not specified.")

	(cpu_physical,
	 cpu_logical) = detect_cpu_cores()
	gpu_available: bool = torch.cuda.is_available()
	device: str = 'cuda' if gpu_available else 'cpu'
	maximum_workers: int = min(cpu_physical // 2,
														 number_parallel)
	number_environments: int = cpu_logical // maximum_workers
	resources: dict = {'cpu_physical': cpu_physical,
										 'cpu_logical': cpu_logical,
										 'gpu_available': gpu_available,
										 'device': device,
           					 'maximum_workers': maximum_workers,
                 		 'number_environments': number_environments}

	return resources
def load_configuration(configuration_path = None) -> dict:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")

	with open(configuration_path,
						'r') as file:
		configuration: dict = yaml.safe_load(file)

	return configuration
def get_maximum_training_episode(training_data_directory: str = None) -> int:
	if training_data_directory is None:
		raise ValueError("Training data directory not specified.")

	maximum_training_episode: int = 0

	for alpha_folder in os.listdir(training_data_directory):
		alpha_folder_path: str = os.path.join(training_data_directory,
																				 	alpha_folder)

		if not os.path.isdir(alpha_folder_path):
			continue

		for file in os.listdir(alpha_folder_path):
			if file.startswith('first_episode') or file.startswith('last_episode'):
				continue

			file_path: str = os.path.join(alpha_folder_path,
																		file)

			try:
				data: pd.DataFrame = pd.read_csv(file_path)
				maximum_episode: int = data['episode'].max()

				if maximum_episode > maximum_training_episode:
					maximum_training_episode = maximum_episode
			except Exception as exception:
				print(f"Warning: Could not read {file_path}: {exception}")

	return maximum_training_episode
def get_cells(environment: np.ndarray = None,
							type: str = None) -> list:
	if environment is None:
		raise ValueError("Environment not specified.")
	if type not in ['open',
                  'closed',
									'unknown']:
		raise ValueError("Type must be 'open', 'closed', or 'unknown'.")

	cells: list = []

	for row in range(environment.shape[0]):
		for column in range(environment.shape[1]):
			if environment[row,
										 column] == (0 if type == 'open' else 1 if type == 'closed' else -1):
				cells.append((row,
											column))

	return cells
def get_shannon_entropy(belief_flattened_indices: np.ndarray = None) -> float:
	if belief_flattened_indices is None:
		raise ValueError("Belief flattened indices not specified.")

	shannon_entropy: float = -np.sum(belief_flattened_indices * np.log2(belief_flattened_indices))

	return shannon_entropy
def get_manhattan_distance(position_1: tuple = None,
													 position_2: tuple = None) -> int:
	if position_1 is None:
		raise ValueError("Position 1 not specified.")
	if position_2 is None:
		raise ValueError("Position 2 not specified.")

	manhattan_distance: int = abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1])

	return manhattan_distance
def get_shortest_distance(position_1: tuple = None,
													position_2: tuple = None,
													environment_knowledge: np.ndarray = None) -> int:
	if position_1 is None:
		raise ValueError("Position 1 not specified.")
	if position_2 is None:
		raise ValueError("Position 2 not specified.")
	if environment_knowledge is None:
		raise ValueError("Environment knowledge not specified.")

	directions: list = [(-1,
                       0),
                      (1,
                       0),
                      (0,
                       -1),
                      (0,
                       1)]
	unknown_cells_knowledge: list = get_cells(environment = environment_knowledge,
																						type = 'unknown')
	closed_cells_knowledge: list = get_cells(environment = environment_knowledge,
	                                         type = 'closed')
	queue: list = [(0,
	                position_1)]
	visited: dict = {position_1: 0}
	shortest_path: int = 2 * environment_knowledge.shape[0]

	while queue:
		(distance,
	   (row,
	    column)) = heapq.heappop(queue)

		if distance > visited[(row,
	                          column)]:
			continue
		if (row,
	      column) == position_2:
			return distance
		for (delta_row,
	       delta_column) in directions:
			new_row: int = row + delta_row
			new_column: int = column + delta_column
			new_distance: int = distance + 1

			if not (0 <= new_row < environment_knowledge.shape[0] and 0 <= new_column < environment_knowledge.shape[1]):
				continue
			if (new_row,
	        new_column) in closed_cells_knowledge:
				continue
			if (new_row,
	        new_column) in visited and new_distance >= visited[(new_row,
	                                                            new_column)]:
				continue
			if (new_row,
	        new_column) in unknown_cells_knowledge:
				estimate: int = new_distance + get_manhattan_distance((new_row,
																															 new_column),
																															position_2)
				shortest_path = min(shortest_path,
	                          estimate)

			visited[(new_row,
	             new_column)] = new_distance

			heapq.heappush(queue,
	                   (new_distance,
	                    (new_row,
	                     new_column)))

	return shortest_path
def harmonize_rollout_hyperparameters(batch_size: int = None,
																			number_steps: int = None,
																			number_environments: int = None,
																			total_timesteps: int = None) -> tuple[int,
																																						int,
																																						int]:
	if batch_size is None:
		raise ValueError("Batch size not specified.")
	if number_steps is None:
		raise ValueError("Number of steps not specified.")
	if number_environments is None:
		raise ValueError("Number of environments not specified.")
	if total_timesteps is None:
		raise ValueError("Total timesteps not specified.")

	rollout_buffer_size: int = number_steps * number_environments

	if batch_size > rollout_buffer_size:
		print(f"[warn] batch_size ({batch_size}) > rollout buffer ({rollout_buffer_size}); changing batch_size to {rollout_buffer_size}")

		batch_size = rollout_buffer_size
	if rollout_buffer_size % batch_size != 0:
		new_batch_size: int = batch_size

		while new_batch_size > 0 and rollout_buffer_size % new_batch_size != 0:
			new_batch_size -= 1

		if new_batch_size == 0:
			new_batch_size = 1

		print(f"[warn] rollout buffer ({rollout_buffer_size}) not divisible by batch_size ({batch_size}); changing batch_size to {new_batch_size}")

		batch_size = new_batch_size

	if total_timesteps % rollout_buffer_size != 0:
		new_total_timesteps: int = (total_timesteps // rollout_buffer_size) * rollout_buffer_size
		
		if new_total_timesteps == 0:
			new_total_timesteps = rollout_buffer_size
		
		print(f"[warn] total_timesteps ({total_timesteps}) not divisible by rollout buffer ({rollout_buffer_size}); changing total_timesteps to {new_total_timesteps}")

		total_timesteps = new_total_timesteps

	return (batch_size,
					number_steps,
				  total_timesteps)
def get_timestamp(data_directory: str = None) -> str | None:
	if data_directory is None:
		raise ValueError("Data directory not specified.")
	if not os.path.exists(data_directory):
		return None

	csv_files: list = [file for file in os.listdir(data_directory) if file.endswith('.csv') and file.startswith('2025') and len(file) == 19]
	csv_files: list = sorted(csv_files,
                           key = lambda file: os.path.getmtime(os.path.join(data_directory,
																																					  file)),
													 reverse = True)
	timestamp: str = csv_files[0].replace('.csv',
																				'')

	return timestamp
def get_checkpoint(checkpoint_directory: str = None) -> tuple[str,
																															str]:
	if checkpoint_directory is None:
		raise ValueError("Checkpoint directory not specified.")

	checkpoint_files: list = [file for file in os.listdir(checkpoint_directory) if file.endswith('.zip') and file.startswith('2025') and len(file) > 19]
	checkpoint_files: list = sorted(checkpoint_files,
																	key = lambda file: os.path.getmtime(os.path.join(checkpoint_directory,
                                                                                   file)),
																	reverse = True)
	latest_checkpoint_zip: str = checkpoint_files[0]
	checkpoint_model_path: str = os.path.join(checkpoint_directory,
                                           	latest_checkpoint_zip)
	base_name: str = latest_checkpoint_zip.replace('.zip',
                                                 '')
	parts: list = base_name.split('_')
	timestamp: str = parts[0]
	steps: str = parts[1]
	vector_normalize_name: str = f"{timestamp}_vecnormalize_{steps}_steps.pkl"
	checkpoint_vector_normalize_path: str = os.path.join(checkpoint_directory,
                                                       vector_normalize_name)

	return (checkpoint_model_path,
          checkpoint_vector_normalize_path)
def cleanup_checkpoints(checkpoint_directory: str = None) -> None:
	if checkpoint_directory is None:
		raise ValueError("Checkpoint directory not specified.")
	
	checkpoint_files: list = [file for file in os.listdir(checkpoint_directory) if file.endswith('.zip') and file.startswith('2025') and len(file) > 19]

	if len(checkpoint_files) <= 1:
		return

	checkpoint_files: list = sorted(checkpoint_files,
																	key = lambda file: os.path.getmtime(os.path.join(checkpoint_directory,
                                                                                	 file)))
	files_to_delete: int = len(checkpoint_files) - 1

	for index in range(files_to_delete):
		checkpoint_name: str = checkpoint_files[index].replace('.zip',
                                                           '')
		parts: list = checkpoint_name.split('_')
		timestamp: str = parts[0]
		steps: str = parts[1]
		model_file: str = os.path.join(checkpoint_directory,
                                   checkpoint_files[index])
		pkl_file: str = os.path.join(checkpoint_directory,
                                 f"{timestamp}_vecnormalize_{steps}_steps.pkl")

		try:
			os.remove(model_file)
			os.remove(pkl_file)
			print(f"Deleted checkpoint: {checkpoint_files[index]}")
			print(f"Deleted checkpoint: {os.path.basename(pkl_file)}")
		except Exception as e:
			print(f"Warning: Could not delete checkpoint {checkpoint_name}: {e}")
def launch_tensorboard(output_directory: str = None,
					   					 port: int = None) -> None:
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if port is None:
		raise ValueError("Port not specified.")

	tensorboard_directory = os.path.join(output_directory,
                                       'tensorboard')

	try:
		print("\n" + "=" * 80)
		print("LAUNCHING TENSORBOARD")
		print("=" * 80)
		print(f"Tensorboard logs: {tensorboard_directory}")
		print(f"Opening in browser at http://localhost:{port}")
		print("=" * 80)

		subprocess.Popen(['tensorboard',
											'--logdir',
                      tensorboard_directory,
											'--port',
                      str(port)],
										 stdout = subprocess.DEVNULL,
										 stderr = subprocess.DEVNULL)
		time.sleep(1)
		webbrowser.open(f'http://localhost:{port}')
	except FileNotFoundError:
		print("ERROR: tensorboard not installed")
		print("Install with: pip install tensorboard")
	except Exception as exception:
		print(f"ERROR launching tensorboard: {exception}")
def load_maskableppo(configuration_path: str = None,
  									 load_model_path: str = None,
										 load_vector_normalize_path: str = None,
										 vectorized_environment: VecNormalize = None,
           					 training: bool = None) -> tuple[MaskablePPO,
																										 VecNormalize]:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if load_model_path is None:
		raise ValueError("Model path not specified.")
	if load_vector_normalize_path is None:
		raise ValueError("VecNormalize path not specified.")
	if vectorized_environment is None:
		raise ValueError("Vectorized environment not specified.")
	if training is None:
		raise ValueError("Training mode not specified.")

	configuration: dict = load_configuration(configuration_path = configuration_path)
	load_maskableppo_configuration: dict = configuration['load_maskableppo']
	normalize_observations: bool = load_maskableppo_configuration['normalize_observations']
	normalize_rewards: bool = load_maskableppo_configuration['normalize_rewards']
	vectorized_environment = VecNormalize.load(load_path = load_vector_normalize_path,
																						 venv = vectorized_environment)
	vectorized_environment.training = training
	vectorized_environment.norm_obs = normalize_observations
	vectorized_environment.norm_reward = normalize_rewards

	model: MaskablePPO = MaskablePPO.load(path = load_model_path,
																				env = vectorized_environment)

	return (model,
          vectorized_environment)
def delete_pycache(directories):
	try:
		for directory in directories:
			pycache_path = os.path.join(directory,
                                  '__pycache__')

			if os.path.exists(pycache_path):
				shutil.rmtree(pycache_path)
				print(f"Deleted {pycache_path}")
			else:
				print(f"{pycache_path} was not found")
	except Exception as exception:
		print(f"Error deleting __pycache__ folders: {exception}")
