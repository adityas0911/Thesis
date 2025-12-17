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

import numpy as np

from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (CheckpointCallback,
                        												CallbackList)
from stable_baselines3.common.vec_env import (VecNormalize,
                                              SubprocVecEnv,
                                              DummyVecEnv)

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.environment.search_rescue_environment import SearchAndRescue
from source.utilities.helpers import (detect_optimal_resources,
																			load_configuration,
																			harmonize_rollout_hyperparameters,
																			get_timestamp,
																			get_checkpoint,
																			load_maskableppo,
                   										delete_pycache)
from source.utilities.callbacks import (TensorboardCallback,
																				EpisodeCallback,
																				StepCallback,
																				CleanupCheckpointCallback)

def action_mask_function(environment: SearchAndRescue = None) -> np.ndarray:
	if environment is None:
		raise ValueError("Environment not specified.")

	return environment.get_action_mask()
def initialize_environment(configuration_path: str = None,
                       		 alpha: float = None) -> SearchAndRescue:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if alpha is None:
		raise ValueError("Alpha value not specified.")

	environment: SearchAndRescue = SearchAndRescue(configuration_path = configuration_path,
																								 alpha = alpha)
	environment: ActionMasker = ActionMasker(env = environment,
															 						 action_mask_fn = action_mask_function)
	environment: Monitor = Monitor(env = environment,
                                 allow_early_resets = True,
                                 override_existing = True)

	return environment
def make_environment(configuration_path: str = None,
										 alpha: float = None,
                     environment_number: int = None) -> SearchAndRescue:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if alpha is None:
		raise ValueError("Alpha value not specified.")
	if environment_number is None:
		raise ValueError("Environment number not specified.")

	configuration: dict = load_configuration(configuration_path = configuration_path)
	environment_configuration: dict = configuration['environment']
	seed: int = environment_configuration['seed']
	environment: SearchAndRescue = initialize_environment(configuration_path = configuration_path,
															 				 									alpha = alpha)

	if seed is None:
		environment.reset(seed = seed)
	else:
		environment.reset(seed = seed + environment_number)

	return environment
def make_vectorized_environment(configuration_path: str = None,
																alpha: float = None,
                								number_environments: int = None) -> SubprocVecEnv | DummyVecEnv:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if alpha is None:
		raise ValueError("Alpha value not specified.")
	if number_environments is None:
		raise ValueError("Number of environments not specified.")

	environments: list = [lambda environment_number = environment_number: make_environment(configuration_path = configuration_path,
																																												 alpha = alpha,
																																												 environment_number = environment_number) for environment_number in range(number_environments)]

	if number_environments > 1:
		vectorized_environment: SubprocVecEnv = SubprocVecEnv(environments)

		print(f"Using SubprocVecEnv with {number_environments} parallel environments")
	else:
		vectorized_environment: DummyVecEnv = DummyVecEnv(environments)

		print(f"Using DummyVecEnv with {number_environments} environments")

	return vectorized_environment
def get_vector_normalized_environment(vectorized_environment: SubprocVecEnv | DummyVecEnv = None,
                                      gamma: float = None) -> VecNormalize:
	if vectorized_environment is None:
		raise ValueError("Vectorized environment not specified.")
	if gamma is None:
		raise ValueError("Gamma value not specified.")

	vectorized_environment: VecNormalize = VecNormalize(venv = vectorized_environment,
																											training = True,
																											norm_obs = True,
																											norm_reward = True,
																											clip_obs = 10,
																											clip_reward = 10,
																											gamma = gamma,
																											epsilon = 1e-8)

	return vectorized_environment
def train_maskableppo(configuration_path: str = None,
                      output_directory: str = None,
                      alpha: float = None,
                      resources: dict = None) -> None:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if alpha is None:
		raise ValueError("Alpha value not specified.")
	if resources is None:
		raise ValueError("Resources not specified.")

	configuration: dict = load_configuration(configuration_path = configuration_path)
	device: str = resources['device']
	number_environments: int = resources['number_environments']
	train_configuration: dict = configuration['train']
	policy: str = train_configuration['policy']
	start_learning_rate: float = train_configuration['start_learning_rate']
	end_learning_rate: float = train_configuration['end_learning_rate']
	number_steps: int = train_configuration['number_steps']
	batch_size: int = train_configuration['batch_size']
	number_epochs: int = train_configuration['number_epochs']
	gamma: float = train_configuration['gamma']
	generalized_advantage_estimation_lambda: float = train_configuration['generalized_advantage_estimation_lambda']
	clip_range: float = train_configuration['clip_range']
	normalize_advantage: bool = train_configuration['normalize_advantage']
	entropy_coefficient: float = train_configuration['entropy_coefficient']
	value_function_coefficient: float = train_configuration['value_function_coefficient']
	maximum_gradient_norm: float = train_configuration['maximum_gradient_norm']
	verbose: int = train_configuration['verbose']
	stats_window_size: int = train_configuration['stats_window_size']
	_init_setup_model: bool = train_configuration['_init_setup_model']
	total_timesteps: int = train_configuration['total_timesteps']
	checkpoint_frequency: int = train_configuration['checkpoint_frequency']
	save_frequency: int = checkpoint_frequency / number_environments
	run_name: str = f"alpha_{alpha:.2f}"
	models_directory: str = os.path.join(output_directory,
																			 'models',
																			 run_name)
	vector_normalize_directory: str = os.path.join(output_directory,
																							 	 'vector_normalize')
	tensorboard_directory: str = os.path.join(output_directory,
																						'tensorboard',
																						run_name)
	checkpoint_directory: str = os.path.join(output_directory,
																					 'checkpoints',
																					 run_name)
	training_data_directory: str = os.path.join(output_directory,
																							'training_data',
																							run_name)
	model_path: str = os.path.join(models_directory,
                                 'final_model.zip')
	vector_normalize_path: str = os.path.join(vector_normalize_directory,
																						f'{run_name}.pkl')
	resuming_from_final: bool = os.path.exists(model_path) and os.path.exists(vector_normalize_path)
	resuming_from_checkpoint: bool = os.path.exists(checkpoint_directory)

	if resuming_from_final:
		resuming: bool = True
		load_model_path: str = model_path
		load_vector_normalize_path: str = vector_normalize_path
	elif resuming_from_checkpoint:
		resuming: bool = True
		(load_model_path,
   	 load_vector_normalize_path) = get_checkpoint(checkpoint_directory = checkpoint_directory)
	else:
		resuming: bool = False
	if resuming:
		timestamp: str = get_timestamp(data_directory = training_data_directory)

		if timestamp is None:
			raise ValueError(f"No training data CSV files found in {training_data_directory} to determine timestamp for resuming.")

		first_episode_logged: bool = True
	else:
		timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
		first_episode_logged: bool = False

	episode_path: str = os.path.join(training_data_directory,
                                   f'{timestamp}.csv')
	tensorboard_callback: TensorboardCallback = TensorboardCallback(episode_path = episode_path,
																																	first_episode_logged = first_episode_logged,
																																	verbose = 0)
	episode_callback: EpisodeCallback = EpisodeCallback(episode_path = episode_path,
                                                      episodes = None,
																											first_episode_logged = first_episode_logged,
																											verbose = 0)
	step_callback: StepCallback = StepCallback(data_directory = training_data_directory,
																						 first_episode_logged = first_episode_logged,
																						 verbose = 0)
	checkpoint_callback: CheckpointCallback = CheckpointCallback(save_freq = save_frequency,
																															 save_path = checkpoint_directory,
																															 name_prefix = timestamp,
																															 save_vecnormalize = True)
	cleanup_callback: CleanupCheckpointCallback = CleanupCheckpointCallback(checkpoint_directory = checkpoint_directory,
																																					verbose = 0)
	callbacks: list = [tensorboard_callback,
										 episode_callback,
										 step_callback,
										 checkpoint_callback,
										 cleanup_callback]
	callback: CallbackList = CallbackList(callbacks = callbacks)

	os.makedirs(name = models_directory,
              exist_ok = True)
	os.makedirs(name = vector_normalize_directory,
              exist_ok = True)
	os.makedirs(name = tensorboard_directory,
              exist_ok = True)
	os.makedirs(name = checkpoint_directory,
							exist_ok = True)
	os.makedirs(name = training_data_directory,
							exist_ok = True)

	(batch_size,
	 number_steps,
	 total_timesteps) = harmonize_rollout_hyperparameters(batch_size = batch_size,
																												number_steps = number_steps,
																												number_environments = number_environments,
																												total_timesteps = total_timesteps)

	print(f"Creating environment with alpha = {alpha}")

	vectorized_environment: SubprocVecEnv | DummyVecEnv = make_vectorized_environment(configuration_path = configuration_path,
																																										alpha = alpha,
																																										number_environments = number_environments)
	vectorized_environment: VecNormalize = get_vector_normalized_environment(vectorized_environment = vectorized_environment,
                                                                           gamma = gamma)

	if resuming:
		print(f"Found existing model at {load_model_path}")
		print(f"Found existing VecNormalize at {load_vector_normalize_path}")

		(model,
     vectorized_environment) = load_maskableppo(load_model_path = load_model_path,
																								load_vector_normalize_path = load_vector_normalize_path,
																								vectorized_environment = vectorized_environment,
																								training_mode = True)

		print("Loading MaskablePPO agent...")
		print(f"Resuming training for additional {total_timesteps} timesteps...")
		model.learn(total_timesteps = total_timesteps,
								callback = callback,
								log_interval = 1,
								tb_log_name = timestamp,
								reset_num_timesteps = False,
								use_masking = True,
                progress_bar = True)
		model.save(path = model_path)
		vectorized_environment.save(save_path = vector_normalize_path)
		print(f"\nResumed training complete!")
		print(f"Updated model saved to: {model_path}")
		print(f"Updated VecNormalize saved to: {vector_normalize_path}")
	else:
		print("Creating MaskablePPO agent...")

		def learning_rate_schedule(progress_remaining: float) -> float:
			learning_rate: float = end_learning_rate + (start_learning_rate - end_learning_rate) * progress_remaining

			return learning_rate

		model: MaskablePPO = MaskablePPO(policy = policy,
																		 env = vectorized_environment,
																		 learning_rate = learning_rate_schedule,
																		 n_steps = number_steps,
																		 batch_size = batch_size,
																		 n_epochs = number_epochs,
																		 gamma = gamma,
																		 gae_lambda = generalized_advantage_estimation_lambda,
																		 clip_range = clip_range,
																		 normalize_advantage = normalize_advantage,
																		 ent_coef = entropy_coefficient,
																		 vf_coef = value_function_coefficient,
																		 max_grad_norm = maximum_gradient_norm,
																		 verbose = verbose,
																		 stats_window_size = stats_window_size,
																		 tensorboard_log = tensorboard_directory,
																		 device = device,
																		 _init_setup_model = _init_setup_model)

		print(f"Training for {total_timesteps} timesteps...")
		model.learn(total_timesteps = total_timesteps,
								callback = callback,
								log_interval = 1,
								tb_log_name = timestamp,
								reset_num_timesteps = True,
								use_masking = True,
								progress_bar = True)
		model.save(path = model_path)
		vectorized_environment.save(save_path = vector_normalize_path)
		print(f"\nTraining complete!")
		print(f"Model saved to: {model_path}")
		print(f"VecNormalize saved to: {vector_normalize_path}")

if __name__ == "__main__":
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description = "Train MaskablePPO with specified configuration")

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
	parser.add_argument('--alpha',
                      type = float,
                      default = 0.5,
											help = "Alpha value")
	parser.add_argument('--resources',
											type = dict,
											default = detect_optimal_resources(number_parallel = 1),
											help = "Resources dictionary")

	arguments: argparse.Namespace = parser.parse_args()
	
	train_maskableppo(configuration_path = arguments.configuration_path,
                    output_directory = arguments.output_directory,
										alpha = arguments.alpha,
										resources = arguments.resources)
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
