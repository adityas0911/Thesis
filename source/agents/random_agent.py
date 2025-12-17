import sys
import os

import numpy as np

from stable_baselines3.common.vec_env import (SubprocVecEnv,
                                              DummyVecEnv)

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.utilities.helpers import load_configuration

class RandomWrapper:
	def __init__(self,
							 vectorized_environment: SubprocVecEnv | DummyVecEnv = None,
							 configuration_path: str = None):
		if vectorized_environment is None:
			raise ValueError("vectorized_environment not specified.")
		if configuration_path is None:
			raise ValueError("configuration_path not specified.")

		self.vectorized_environment: SubprocVecEnv | DummyVecEnv = vectorized_environment
		configuration: dict = load_configuration(configuration_path = configuration_path)
		random_configuration: dict = configuration['random']
		seed: int = random_configuration['seed']
		self.number_environments: int = vectorized_environment.num_envs
		self.actions: np.ndarray = None
		self.num_timesteps: int = 0
		self.generators = []

		if seed is None:
			for _ in range(self.number_environments):
				self.generators.append(np.random.default_rng(seed = seed))
		else:
			for environment_number in range(self.number_environments):
				self.generators.append(np.random.default_rng(seed = seed + environment_number))
	def predict(self,
              action_masks: np.ndarray = None):
		if action_masks is None:
			raise ValueError("action_masks not specified.")

		actions: list = []

		for environment_number in range(self.number_environments):
			action_mask: np.ndarray = action_masks[environment_number]
			valid_actions: list = [index for (index,
                                     		valid) in enumerate(action_mask) if valid]
			seed = self.generators[environment_number]
			best_action: int = seed.choice(valid_actions)

			actions.append(best_action)

		self.actions = np.array(actions)
		self.num_timesteps += 1

		return self.actions
