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
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import (SubprocVecEnv,
																							DummyVecEnv)

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.agents.train_maskableppo import make_vectorized_environment
from source.agents.pomcp_agent import POMCPWrapper
from source.utilities.helpers import (detect_optimal_resources,
                                      load_configuration,
                                      get_maximum_training_episode,
																			get_timestamp,
																			delete_pycache)
from source.utilities.callbacks import (EpisodeCallback,
																				StepCallback)

def evaluate_pomcp(configuration_path: str = None,
                   output_directory: str = None,
                   alpha: float = None,
                   resources: dict = None) -> None:
	if configuration_path is None:
		raise ValueError("Configuration path not specified.")
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if alpha is None:
		raise ValueError("Alpha not specified.")
	if resources is None:
		raise ValueError("Resources not specified.")

	configuration: dict = load_configuration(configuration_path)
	number_environments: int = resources['number_environments']
	evaluation_configuration: dict = configuration['evaluation']
	training_episode_percentage: float = evaluation_configuration['training_episode_percentage']
	run_name: str = f'alpha_{alpha:.2f}'
	training_data_directory: str = os.path.join(output_directory,
															 								'training_data')
	evaluation_data_directory: str = os.path.join(output_directory,
																							  'evaluation_data',
																							  'pomcp',
																							 	run_name)
	evaluating: bool = os.path.exists(evaluation_data_directory)

	if evaluating:
		timestamp: str = get_timestamp(data_directory = evaluation_data_directory)

		if timestamp is None:
			timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
			first_episode_logged: bool = False
		else:
			first_episode_logged: bool = True
	else:
		timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
		first_episode_logged: bool = False
	if training_episode_percentage:
		maximum_training_episode: int = get_maximum_training_episode(training_data_directory = training_data_directory)
		episodes: int = int(maximum_training_episode * training_episode_percentage)
	else:
		raise ValueError("Evaluation episodes not specified in configuration.")

	episode_path: str = os.path.join(evaluation_data_directory,
																	 f'episodes_{timestamp}.csv')
	episode_callback: EpisodeCallback = EpisodeCallback(episode_path = episode_path,
																											episodes = episodes,
																											first_episode_logged = first_episode_logged,
																											verbose = 0)
	step_callback: StepCallback = StepCallback(data_directory = evaluation_data_directory,
                                             first_episode_logged = first_episode_logged,
																						 verbose = 0)
	callbacks: list = [episode_callback,
                     step_callback]
	callback: CallbackList = CallbackList(callbacks = callbacks)

	os.makedirs(evaluation_data_directory,
              exist_ok = True)

	print(f"Creating environment with alpha = {alpha}")

	vectorized_environment: SubprocVecEnv | DummyVecEnv = make_vectorized_environment(configuration_path = configuration_path,
																																										alpha = alpha,
																																										number_environments = number_environments)

	print("Loading POMCP agent...")

	model: POMCPWrapper = POMCPWrapper(vectorized_environment = vectorized_environment,
																		 configuration_path = configuration_path,
																		 alpha = alpha)

	print(f"Evaluating for {episodes} episodes...")
	callback.init_callback(model = model)

	observations: dict = vectorized_environment.reset()

	while episode_callback.episode_count < episodes:
		action_masks: np.ndarray = vectorized_environment.env_method('get_action_mask')
		actions: np.ndarray = model.predict(observations = observations,
                                        action_masks = action_masks)
		(observations,
		 rewards,
		 _,
		 informations) = vectorized_environment.step(actions = actions)

		for (environment_number,
				 information) in enumerate(informations):
			if information['success'] or information['failure']:
				model.reset_environment(environment_number = environment_number)
		for callback_instance in callback.callbacks:
			callback_instance.locals = {'infos': informations,
																	'actions': actions,
																	'rewards': rewards}

		continue_running: bool = callback.on_step()

		if not continue_running or episode_callback.episode_count >= episodes:
			break

	callback.on_training_end()
	print(f"\nEvaluation complete!")
	print(f"Results saved to: {episode_path}")

if __name__ == "__main__":
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description = "Evaluate POMCP agent for all alpha values (AUTO-OPTIMIZED)")

	parser.add_argument('--configuration_path',
											type = str,
											default = os.path.join(PROJECT_ROOT,
																						'configuration.yaml'),
											help = "Path to the configuration file")
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

	evaluate_pomcp(configuration_path = arguments.configuration_path,
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
