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
import pygame

import numpy as np
import gymnasium as gym
import multiprocessing as mp

from gymnasium import spaces

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.environment.environment_generator import generate_environment
from source.updates.sensor import get_beep
from source.updates.reward import get_reward
from source.utilities.helpers import (load_configuration,
																			get_cells,
																			get_shannon_entropy,
																			get_shortest_distance,
                   										delete_pycache)
from source.updates.belief import (normalize_belief,
																	 update_sense,
																	 update_robot_move,
																	 update_victim_move,
																	 update_found_victim)

class SearchAndRescue(gym.Env):
	metadata: dict = {'render_modes': ['human']}
	ACTION_SENSE: int = 0
	ACTION_UP: int = 1
	ACTION_DOWN: int = 2
	ACTION_LEFT: int = 3
	ACTION_RIGHT: int = 4
	COLOR_OPEN = (255,
								255,
								255)
	COLOR_WALL = (0,
								0,
								0)
	COLOR_UNKNOWN = (40,
									 40,
									 40)
	COLOR_GRID = (40,
								40,
								40)
	COLOR_ROBOT = (0,
								 0,
								 255)
	COLOR_VICTIM = (255,
									0,
									0)
	actions: list = [(0,
										0),
									 (-1,
										0),
									 (1,
										0),
									 (0,
										-1),
									 (0,
										1)]

	def __init__(self,
               configuration_path: str = None,
							 alpha: float = None) -> None:
		super().__init__()

		if configuration_path is None:
			raise ValueError("Configuration path not specified.")
		if alpha is None:
			raise ValueError("Alpha value not specified.")

		configuration: dict = load_configuration(configuration_path = configuration_path)
		self.alpha: float = alpha
		environment_configuration: dict = configuration['environment']
		visualization_configuration: dict = configuration['visualization']
		self.reward_configuration: dict = configuration['reward']
		self.seeded: bool = environment_configuration['seed'] is not None
		self.sensor_alpha: float = environment_configuration['sensor_alpha']
		self.environment_size: int = environment_configuration['environment_size']
		self.vision_range: int = environment_configuration['vision_range']
		self.maximum_steps: int = environment_configuration['maximum_steps']
		self.frames_per_second: int = visualization_configuration['frames_per_second']
		self.render_mode: str = visualization_configuration['render_mode']
		self.show_heatmap: bool = visualization_configuration['show_heatmap']
		self.show_victim: bool = visualization_configuration['show_victim']
		self.show_path: bool = visualization_configuration['show_path']
		self.window_size: int = visualization_configuration['window_size']

		if mp.current_process().name != 'MainProcess':
			self._render: bool = False
		else:
			self._render: bool = visualization_configuration['render']

		self.cell_size: int = self.window_size // self.environment_size
		self.environment: np.ndarray = -1 * np.ones((self.environment_size,
																								 self.environment_size),
																								dtype = np.float32)
		self.robot_position: tuple = (-1,
                                	-1)
		self.victim_position: tuple = (-1,
                                   -1)
		self.open_cells: list = []
		self.open_cells_count: int = -1
		self.open_cells_knowledge: list = []
		self.closed_cells_knowledge: list = []
		vision_size: int = 2 * self.vision_range + 1
		self.local_vision: np.ndarray = -1 * np.ones((vision_size,
																									vision_size),
																								 dtype = np.int8)
		self.environment_knowledge: np.ndarray = -1 * np.ones((self.environment_size,
																													 self.environment_size),
																													dtype = np.int8)
		self.belief: np.ndarray = -1 * np.ones((self.environment_size,
																						self.environment_size),
																					 dtype = np.float32)
		self.action_space: spaces.Discrete = spaces.Discrete(5)
		self.observation_space = spaces.Dict({'global_map': spaces.Box(low = 0.0,
																																	 high = 1.0,
																																	 shape = (self.environment_size,
																																						self.environment_size,
																																						5),
																																	 dtype = np.float32),
																					'normalized_belief_shannon_entropy': spaces.Box(low = 0.0,
																																													high = 1.0,
																																													shape = (1,),
																																													dtype = np.float32),
																					'normalized_distance_to_maximum_belief': spaces.Box(low = 0.0,
																																															high = 1.0,
																																															shape = (1,),
																																															dtype = np.float32)})
		self.steps: int = -1
		self.total_moves: int = -1
		self.total_senses: int = -1
		self.initial_distance_to_victim: int = -1
		self.distance_to_maximum_belief_before: int = -1
		self.distance_to_maximum_belief: int = -1
		self.distance_to_maximum_belief_reduction: int = -1
		self.belief_flattened: np.ndarray = np.array([])
		self.positive_belief_flattened_indices: np.ndarray = np.array([])
		self.initial_belief_shannon_entropy: float = -1.0
		self.belief_shannon_entropy_before: float = -1.0
		self.belief_shannon_entropy: float = -1.0
		self.belief_shannon_entropy_reduction: float = -1.0
		self.terminated: bool = False
		self.truncated: bool = False
		self.information: dict = {}
		self.global_row: int = -1
		self.global_column: int = -1
		self.local_vision_row: int = -1
		self.local_vision_column: int = -1
		self.new_row: int = -1
		self.new_column: int = -1
		self.path: list = []
		self.window: object = None
		self.clock: object = None
	def reset(self,
						seed: int = None,
            **kwargs) -> tuple:
		super().reset(seed = seed)

		if self.sensor_alpha is None:
			print("Sensor alpha value not specified. Using default: random sensor alpha")

			self.sensor_alpha = float(self.np_random.random())

		self.environment = generate_environment(environment_size = self.environment_size,
																						seed = self.np_random)
		self.open_cells: list = get_cells(environment = self.environment,
                                 			type = 'open')
		self.open_cells_count = len(self.open_cells)
		indices: np.ndarray = self.np_random.choice(len(self.open_cells),
																								size = 2,
																								replace = False)
		self.robot_position = tuple(self.open_cells[indices[0]])
		self.victim_position = tuple(self.open_cells[indices[1]])
		self.local_vision.fill(-1)
		self.environment_knowledge.fill(-1)
		self.belief.fill(-1)
		self.update_local_vision()
		self.update_environment_knowledge()
		self.initialize_belief()
		self.maximum_distance_to_maximum_belief: float = float(2 * (self.environment_size - 1))
		no_robot_open_cells_count: int = self.open_cells_count - 1
		uniform_probability: float = 1.0 / no_robot_open_cells_count
		belief_flattened_indices: np.ndarray = np.full(no_robot_open_cells_count,
																									 uniform_probability,
																									 dtype = np.float32)
		self.maximum_belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = belief_flattened_indices)
		self.observation = self.get_observation()
		self.steps = 0
		self.total_moves = 0
		self.total_senses = 0
		self.initial_distance_to_victim = get_shortest_distance(position_1 = self.robot_position,
																														position_2 = self.victim_position,
																														environment_knowledge = self.environment_knowledge)
		self.distance_to_maximum_belief_before = self.initial_distance_to_victim
		self.distance_to_maximum_belief = self.distance_to_maximum_belief_before
		self.distance_to_maximum_belief_reduction = self.distance_to_maximum_belief_before - self.distance_to_maximum_belief
		self.belief_flattened = self.belief.flatten()
		self.positive_belief_flattened_indices = self.belief_flattened[self.belief_flattened > 0]
		self.initial_belief_shannon_entropy = get_shannon_entropy(belief_flattened_indices = self.positive_belief_flattened_indices)
		self.belief_shannon_entropy_before = self.initial_belief_shannon_entropy
		self.belief_shannon_entropy = self.belief_shannon_entropy_before
		self.belief_shannon_entropy_reduction = self.belief_shannon_entropy_before - self.belief_shannon_entropy
		self.terminated = False
		self.truncated = False
		self.information = self.get_information()
		self.path = [self.robot_position]

		return (self.observation,
            self.information)
	def step(self,
           action: int = None) -> tuple:
		if action is None:
			raise ValueError("Action not specified.")

		self.steps += 1
		maximum_belief_index = self.belief.argmax()
		maximum_belief_position = np.unravel_index(maximum_belief_index,
																							 self.belief.shape)

		if self.steps == 1:
			self.distance_to_maximum_belief_before = self.initial_distance_to_victim
		else:
			self.distance_to_maximum_belief_before = get_shortest_distance(position_1 = self.robot_position,
																																		 position_2 = maximum_belief_position,
																																		 environment_knowledge = self.environment_knowledge)
		self.belief_flattened = self.belief.flatten()
		self.positive_belief_flattened_indices = self.belief_flattened[self.belief_flattened > 0]
		self.belief_shannon_entropy_before: float = get_shannon_entropy(belief_flattened_indices = self.positive_belief_flattened_indices)

		if action == self.ACTION_SENSE:
			self.total_senses += 1
			beep: int = get_beep(sensor_alpha = self.sensor_alpha,
													 robot_position = self.robot_position,
													 victim_position = self.victim_position,
													 seed = self.np_random if self.seeded else None)
			self.belief = update_sense(sensor_alpha = self.sensor_alpha,
																 robot_position = self.robot_position,
																 belief = self.belief,
																 beep = beep)
		else:
			self.total_moves += 1
			self.robot_position = self.move_robot_position(action = action)
			self.path.append(self.robot_position)
			self.update_local_vision()
			self.update_environment_knowledge()
			self.belief = update_robot_move(robot_position = self.robot_position,
																			environment_knowledge = self.environment_knowledge,
																			belief = self.belief)

			if self.robot_position == self.victim_position:
				self.belief = update_found_victim(robot_position = self.robot_position,
																					belief = self.belief)
				self.terminated = True
		if not self.terminated:
			self.victim_position = self.move_victim_position()
			self.belief = update_victim_move(robot_position = self.robot_position,
																			 environment_knowledge = self.environment_knowledge,
																			 belief = self.belief)

			if self.robot_position == self.victim_position:
				self.belief = update_found_victim(robot_position = self.robot_position,
																					belief = self.belief)
				self.terminated = True
		if not self.terminated and self.steps >= self.maximum_steps:
			self.truncated = True

		normalized_step: float = 1.0 / float(self.maximum_steps)
		self.distance_to_maximum_belief = get_shortest_distance(position_1 = self.robot_position,
																														position_2 = maximum_belief_position,
																														environment_knowledge = self.environment_knowledge)
		self.distance_to_maximum_belief_reduction = self.distance_to_maximum_belief_before - self.distance_to_maximum_belief
		normalized_distance_to_maximum_belief_reduction: float = float(self.distance_to_maximum_belief_reduction) / self.maximum_distance_to_maximum_belief
		self.belief_flattened = self.belief.flatten()
		self.positive_belief_flattened_indices = self.belief_flattened[self.belief_flattened > 0]
		self.belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = self.positive_belief_flattened_indices)
		self.belief_shannon_entropy_reduction: float = self.belief_shannon_entropy_before - self.belief_shannon_entropy
		normalized_belief_shannon_entropy_reduction: float = self.belief_shannon_entropy_reduction / self.maximum_belief_shannon_entropy
		self.observation = self.get_observation()
		reward: float = get_reward(configuration = self.reward_configuration,
															 alpha = self.alpha,
															 normalized_step = normalized_step,
															 normalized_distance_to_maximum_belief_reduction = normalized_distance_to_maximum_belief_reduction,
															 normalized_belief_shannon_entropy_reduction = normalized_belief_shannon_entropy_reduction,
															 terminated = self.terminated,
															 truncated = self.truncated)
		self.information = self.get_information()

		if self._render:
			self.render()

		return (self.observation,
            reward,
            self.terminated,
            self.truncated,
            self.information)
	def update_local_vision(self) -> None:
		for row in range(-self.vision_range,
                     self.vision_range + 1):
			for column in range(-self.vision_range,
                          self.vision_range + 1):
				self.global_row = row + self.robot_position[0]
				self.global_column = column + self.robot_position[1]
				self.local_vision_row = row + self.vision_range
				self.local_vision_column = column + self.vision_range

				if 0 <= self.global_row < self.environment_size and 0 <= self.global_column < self.environment_size:
					self.local_vision[self.local_vision_row,
                            self.local_vision_column] = self.environment[self.global_row,
                                                  											 self.global_column]
				else:
					self.local_vision[self.local_vision_row,
                            self.local_vision_column] = 1.0
	def update_environment_knowledge(self) -> None:
		for row in range(-self.vision_range,
										 self.vision_range + 1):
			for column in range(-self.vision_range,
											    self.vision_range + 1):
				self.global_row = row + self.robot_position[0]
				self.global_column = column + self.robot_position[1]
				self.local_vision_row = row + self.vision_range
				self.local_vision_column = column + self.vision_range

				if 0 <= self.global_row < self.environment_size and 0 <= self.global_column < self.environment_size:
					self.environment_knowledge[self.global_row,
                                     self.global_column] = self.local_vision[self.local_vision_row,
                                                                        		 self.local_vision_column]
	def initialize_belief(self) -> None:
		self.closed_cells_knowledge: list = get_cells(environment = self.environment_knowledge,
																						 			type = 'closed')

		for row in range(self.environment_size):
			for column in range(self.environment_size):
				if (row,
						column) in self.closed_cells_knowledge:
					self.belief[row,
											column] = 0.0
				else:
					self.belief[row,
											column] = 1.0

		self.belief[self.robot_position[0],
								self.robot_position[1]] = 0.0

		self.belief = normalize_belief(belief = self.belief)
	def get_robot_map(self) -> np.ndarray:
		robot_map: np.ndarray = np.zeros((self.environment_size, 
																			self.environment_size),
																		 dtype = np.float32)
		robot_map[self.robot_position[0],
          		self.robot_position[1]] = 1.0

		return robot_map
	def get_observation(self) -> dict:
		open_map: np.ndarray = (self.environment_knowledge == 0).astype(np.float32)
		closed_map: np.ndarray = (self.environment_knowledge == 1).astype(np.float32)
		unknown_map: np.ndarray = (self.environment_knowledge == -1).astype(np.float32)
		belief_map = np.clip(a = self.belief,
                       	 a_min = 1.0e-6,
                         a_max = 1.0)
		belief_map = belief_map / belief_map.sum()
		robot_map: np.ndarray = self.get_robot_map()
		global_map: np.ndarray = np.stack([open_map,
																			 closed_map,
																			 unknown_map,
																			 belief_map,
																			 robot_map],
																			axis = -1)
		normalized_distance_to_maximum_belief: float = self.distance_to_maximum_belief / self.maximum_distance_to_maximum_belief
		normalized_distance_to_maximum_belief = np.clip(a = normalized_distance_to_maximum_belief,
																										a_min = 0.0,
																										a_max = 1.0)
		normalized_belief_shannon_entropy: float = self.belief_shannon_entropy / self.maximum_belief_shannon_entropy
		normalized_belief_shannon_entropy = np.clip(a = normalized_belief_shannon_entropy,
																								a_min = 0.0,
																								a_max = 1.0)
		self.observation = {'global_map': global_map,
												'normalized_belief_shannon_entropy': np.array([normalized_belief_shannon_entropy],
																																			dtype = np.float32),
												'normalized_distance_to_maximum_belief': np.array([normalized_distance_to_maximum_belief],
																															 						dtype = np.float32)}

		return self.observation
	def get_information(self) -> dict:
		if self.terminated:
			success: int = 1
			failure: int = 0
		elif self.truncated:
			success: int = 0
			failure: int = 1
		else:
			success: int = 0
			failure: int = 0

		self.information = {'total_moves': self.total_moves,
												'total_senses': self.total_senses,
												'initial_distance_to_victim': self.initial_distance_to_victim,
												'distance_to_maximum_belief': self.observation['distance_to_maximum_belief'].item(),
												'distance_to_maximum_belief_reduction': self.distance_to_maximum_belief_reduction,
												'initial_belief_shannon_entropy': self.initial_belief_shannon_entropy,
												'belief_shannon_entropy': self.observation['belief_shannon_entropy'].item(),
												'belief_shannon_entropy_reduction': self.belief_shannon_entropy_reduction,
												'success': success,
												'failure': failure}

		return self.information
	def move_robot_position(self,
                          action: int = None) -> tuple[int,
                                                			 int]:
		if action is None:
			raise ValueError("Action not specified.")
		if action == self.ACTION_UP:
			robot_position: tuple[int,
													  int] = (self.robot_position[0] - 1,
																	  self.robot_position[1])
		elif action == self.ACTION_DOWN:
			robot_position: tuple[int,
													  int] = (self.robot_position[0] + 1,
																	  self.robot_position[1])
		elif action == self.ACTION_LEFT:
			robot_position: tuple[int,
													  int] = (self.robot_position[0],
																	  self.robot_position[1] - 1)
		elif action == self.ACTION_RIGHT:
			robot_position: tuple[int,
													  int] = (self.robot_position[0],
																	  self.robot_position[1] + 1)
		else:
			raise ValueError("Invalid action for moving robot position.")

		return robot_position
	def move_victim_position(self) -> tuple[int,
																					int]:
		valid_actions: list = []

		for (row,
			   column) in self.actions:
			self.new_row = row + self.victim_position[0]
			self.new_column = column + self.victim_position[1]
	
			if (self.new_row,
					self.new_column) in self.open_cells:
				valid_actions.append((self.new_row,
														 	self.new_column))

		self.victim_position = tuple(self.np_random.choice(valid_actions))

		return self.victim_position
	def render(self) -> None:
		if self.render_mode == 'human':
			if self.window is None:
				pygame.init()

				self.window = pygame.display.set_mode((self.window_size,
                                               self.window_size))

				pygame.display.set_caption('Search and Rescue')

				self.clock = pygame.time.Clock()

			for event in pygame.event.get():
				if event.type == pygame.QUIT or self.truncated or self.terminated:
					pygame.quit()

					self.window = None
					self.clock = None

					return None

			self.open_cells_knowledge = get_cells(environment = self.environment_knowledge,
                                	 					type = 'open')
			self.closed_cells_knowledge = get_cells(environment = self.environment_knowledge,
																							type = 'closed')
			unknown_cells: list = get_cells(environment = self.environment_knowledge,
																			type = 'unknown')

			for row in range(self.environment_size):
				for column in range(self.environment_size):
					cell_row: int = row * self.cell_size
					cell_column: int = column * self.cell_size

					if (row,
              column) in self.open_cells_knowledge:
						color = self.COLOR_OPEN
					elif (row,
								column) in self.closed_cells_knowledge:
						color = self.COLOR_WALL
					elif (row,
                column) in unknown_cells:
						color = self.COLOR_UNKNOWN
					else:
						raise ValueError("Cell state is invalid.")

					pygame.draw.rect(self.window,
                           color,
                           (cell_column,
                            cell_row,
                            self.cell_size,
                            self.cell_size))
					pygame.draw.rect(self.window,
                           self.COLOR_GRID,
                           (cell_column,
                            cell_row,
                            self.cell_size,
                            self.cell_size),
                           1)

			robot_row: int = self.robot_position[0] * self.cell_size + self.cell_size // 2
			robot_column: int = self.robot_position[1] * self.cell_size + self.cell_size // 2
	
			pygame.draw.circle(self.window,
                         self.COLOR_ROBOT,
                         (robot_column,
                          robot_row),
                         self.cell_size // 3)

			if self.show_heatmap:
				maximum_belief: float = float(np.max(self.belief))

				for row in range(self.environment_size):
					for column in range(self.environment_size):
						normalized_belief: float = self.belief[row,
																									 column] / maximum_belief
						opacity: int = int(50 + 150 * normalized_belief)
						heatmap_surface: pygame.Surface = pygame.Surface((self.cell_size,
																															self.cell_size),
																															pygame.SRCALPHA)

						heatmap_surface.fill((255,
																	80,
																	80,
																	opacity))

						cell_row: int = row * self.cell_size
						cell_column: int = column * self.cell_size

						self.window.blit(heatmap_surface,
														 (cell_column,
															cell_row))
			if self.show_path:
				for (path_row,
	    			 path_column) in self.path:
					path_cell_row: int = path_row * self.cell_size + self.cell_size // 2
					path_cell_column: int = path_column * self.cell_size + self.cell_size // 2
					path_surface: pygame.Surface = pygame.Surface((self.cell_size // 3,
                                                         self.cell_size // 3),
                                                        pygame.SRCALPHA)

					path_surface.fill((200,
                             200,
                             200,
                             30))

					cell_row: int = path_cell_row - self.cell_size // 6
					cell_column: int = path_cell_column - self.cell_size // 6

					self.window.blit(path_surface,
                           (cell_column,
                            cell_row))
			if self.show_victim:
				victim_row: int = self.victim_position[0] * self.cell_size + self.cell_size // 2
				victim_column: int = self.victim_position[1] * self.cell_size + self.cell_size // 2

				pygame.draw.circle(self.window,
                           self.COLOR_VICTIM,
                           (victim_column,
                            victim_row),
                           self.cell_size // 3)

			pygame.display.flip()
			self.clock.tick(self.frames_per_second)
		else:
			raise NotImplementedError
	def get_action_mask(self) -> np.ndarray:
		action_mask: np.ndarray = np.zeros(5,
                                    	 dtype = np.bool_)
		self.open_cells_knowledge: list = get_cells(environment = self.environment_knowledge,
																								type = 'open')

		for (index,
         (row,
					column)) in enumerate(self.actions):
			self.new_row: int = row + self.robot_position[0]
			self.new_column: int = column + self.robot_position[1]

			if (self.new_row,
					self.new_column) in self.open_cells_knowledge:
				action_mask[index] = 1

		return action_mask

if __name__ == "__main__":
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description = "Test SearchAndRescue Environment")

	parser.add_argument('--configuration_path',
											type = str,
											default = os.path.join(PROJECT_ROOT,
																						 'configuration.yaml'),
											help = "Path to configuration file")
	parser.add_argument('--alpha',
											type = float,
											default = 0.5,
											help = "Alpha value")

	arguments: argparse.Namespace = parser.parse_args()

	SearchAndRescue(configuration_path = arguments.configuration_path,
									alpha = arguments.alpha)
	atexit.register(lambda: delete_pycache([os.path.join(PROJECT_ROOT,
																											 'source',
																											 'environment'),
																					os.path.join(PROJECT_ROOT,
																											 'source',
																											 'updates'),
																					os.path.join(PROJECT_ROOT,
																											 'source',
																											 'utilities')]))
