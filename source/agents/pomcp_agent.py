import sys
import os
import math

import numpy as np

from stable_baselines3.common.vec_env import (SubprocVecEnv,
                                              DummyVecEnv)

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.updates.reward import get_reward
from source.updates.sensor import get_beep
from source.updates.belief import (normalize_belief,
                                   update_sense,
                                   update_robot_move,
                                   update_found_victim,
																	 update_victim_move)
from source.utilities.helpers import (load_configuration,
																		  get_cells,
																			get_shannon_entropy,
                   										get_shortest_distance)

ACTION_SENSE: int = 0
ACTION_UP: int = 1
ACTION_DOWN: int = 2
ACTION_LEFT: int = 3
ACTION_RIGHT: int = 4
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

class TreeNode:
	def __init__(self) -> None:
		self.visit_count: int = 0
		self.total_value: float = 0.0
		self.children: dict[int,
                      	TreeNode] = {}
	def get_best_action(self,
											action_mask: list = None,
											exploration_constant: float = None,
           						seed: np.random.Generator = None) -> int:
		if action_mask is None:
			raise ValueError("Action mask not specified.")
		if exploration_constant is None:
			raise ValueError("Exploration constant not specified.")

		seed.shuffle(action_mask)

		parent_visit_count: int = max(self.visit_count,
                                	1)
		best_value: float = -float('inf')
		best_action: int = action_mask[0]

		for action in action_mask:
			child: TreeNode = self.children.get(action)

			if child is None or child.visit_count == 0:
				value: float = float('inf')
			else:
				exploitation_value: float = child.get_exploitation_value()
				exploration_value: float = child.get_exploration_value(exploration_constant = exploration_constant,
																															 parent_visit_count = parent_visit_count)
				value: float = exploitation_value + exploration_value

			if value > best_value:
				best_value = value
				best_action = action

		return best_action
	def get_exploitation_value(self) -> float:
		exploitation_value: float = self.total_value / self.visit_count

		return exploitation_value
	def get_exploration_value(self,
												 		exploration_constant: float = None,
												 		parent_visit_count: int = None) -> float:
		if exploration_constant is None:
			raise ValueError("Exploration constant not specified.")
		if parent_visit_count is None:
			raise ValueError("Parent visit count not specified.")

		exploration_value: float = exploration_constant * math.sqrt(math.log(parent_visit_count) / self.visit_count)

		return exploration_value
class InternalSimulator:
	def __init__(self,
							 environment_knowledge: np.ndarray = None) -> None:
		if environment_knowledge is None:
			raise ValueError("Environment knowledge not specified.")

		self.open_cells_knowledge: list = get_cells(environment = environment_knowledge,
                                                type = 'open')
		self.unknown_cells_knowledge: list = get_cells(environment = environment_knowledge,
                                                   type = 'unknown')
		self.valid_cells_knowledge: set = set(self.open_cells_knowledge).union(set(self.unknown_cells_knowledge))
	def move_robot_position(self,
													robot_position: tuple[int,
                                                int] = None,
													action: int = None) -> tuple[int,
                                                       int]:
		if robot_position is None:
			raise ValueError("Robot position not specified.")
		if action is None:
			raise ValueError("Action not specified.")
		if action == ACTION_UP:
			robot_position = (robot_position[0] - 1,
												robot_position[1])
		elif action == ACTION_DOWN:
			robot_position = (robot_position[0] + 1,
            						robot_position[1])
		elif action == ACTION_LEFT:
			robot_position = (robot_position[0],
           							robot_position[1] - 1)
		elif action == ACTION_RIGHT:
			robot_position = (robot_position[0],
           	 						robot_position[1] + 1)
		else:
			raise ValueError("Invalid action for moving robot position.")

		return robot_position
	def move_victim_position(self,
													 victim_position: tuple[int,
                                     							int] = None,
													 seed: np.random.Generator = None) -> tuple[int,
                                                         							int]:
		if victim_position is None:
			raise ValueError("Victim position not specified.")

		candidates: list = []

		for (row,
         column) in actions:
			new_row: int = row + victim_position[0]
			new_column: int = column + victim_position[1]

			if (new_row,
       		new_column) in self.valid_cells_knowledge:
				candidates.append((new_row,
                       		 new_column))

		victim_position: tuple[int,
                           int] = tuple(seed.choice(candidates))

		return victim_position
class POMCP:
	def __init__(self,
							 configuration_path: str = None,
							 alpha: float = None) -> None:
		if configuration_path is None:
			raise ValueError("Configuration path not specified.")
		if alpha is None:
			raise ValueError("Alpha value not specified.")

		self.alpha: float = alpha
		configuration: dict = load_configuration(configuration_path = configuration_path)
		self.reward_configuration: dict = configuration['reward']
		environment_configuration: dict = configuration['environment']
		train_configuration: dict = configuration['train']
		pomcp_configuration: dict = configuration['pomcp']
		self.sensor_alpha: float = environment_configuration['sensor_alpha']
		self.environment_size: int = environment_configuration['environment_size']
		self.number_simulations: int = pomcp_configuration['number_simulations']
		self.maximum_depth: int = pomcp_configuration['maximum_depth']
		self.exploration_constant: float = pomcp_configuration['exploration_constant']
		self.number_particles: int = pomcp_configuration['number_particles']
		self.gamma: float = train_configuration['gamma']
		self.seed: np.random.Generator = None
		self.simulator: InternalSimulator = None
		self.root: TreeNode = None
		self.particles: list = []
		self.environment_knowledge: np.ndarray = None
		self.belief: np.ndarray = None
	def reset(self,
						observation: dict = None,
						seed: int = None) -> None:
		if observation is None:
			raise ValueError("Observation not specified.")

		self.seed = np.random.default_rng(seed = seed)
		self.environment_knowledge = observation['environment_knowledge'].squeeze()
		self.belief = observation['belief'].squeeze()
		robot_position: tuple[int,
                          int] = observation['robot_position']

		if isinstance(robot_position,
                  np.ndarray):
			robot_position = tuple(robot_position.tolist())
		elif isinstance(robot_position,
                    list):
			robot_position = tuple(robot_position)
		elif isinstance(robot_position,
                    tuple):
			if len(robot_position) == 1 and isinstance(robot_position[0],
                                                 (list,
                                                  np.ndarray)):
				robot_position = tuple(robot_position[0])
			elif all(isinstance(coordinate,
                         (int,
                          np.integer)) for coordinate in robot_position):
				pass
			else:
				robot_position = tuple([int(coordinate) if isinstance(coordinate,
                                                              np.integer) else coordinate for coordinate in robot_position])

		self.root = TreeNode()
		self.simulator = InternalSimulator(environment_knowledge = self.environment_knowledge)
		self.particles = self.get_particles(robot_position = robot_position,
                                        belief = self.belief)
	def get_particles(self,
										robot_position: tuple[int,
                                					int] = None,
										belief: np.ndarray = None) -> list:
		if robot_position is None:
			raise ValueError("Robot position not specified.")
		if belief is None:
			raise ValueError("Belief grid not specified.")

		particles: list = []
		belief = normalize_belief(belief = belief)
		belief_flattened: np.ndarray = belief.flatten()
		valid_victim_indices: np.ndarray = np.where(belief_flattened > 0)[0]
		valid_victim_probabilities: np.ndarray = belief_flattened[valid_victim_indices]

		for _ in range(self.number_particles):
			victim_position_index: int = self.seed.choice(a = valid_victim_indices,
                                                    p = valid_victim_probabilities)
			victim_position: tuple[int,
                          	 int] = tuple(np.unravel_index(victim_position_index,
                                                           belief.shape))

			particles.append((robot_position,
                     		victim_position))

		return particles
	def get_best_action(self,
                      action_mask: np.ndarray = None) -> int:
		if action_mask is None:
			raise ValueError("Action mask not specified.")

		action_mask: list = [index for (index,
                                    valid) in enumerate(action_mask) if valid]

		self.seed.shuffle(action_mask)

		maximum_visit_count: int = -1
		best_action: int = action_mask[0]

		for _ in range(self.number_simulations):
			belief: np.ndarray = self.belief.copy()
			particle: tuple = self.particles[self.seed.choice(len(self.particles))]
			_ = self.simulate(particle = particle,
												node = self.root,
												steps = 0,
												belief = belief)
		for action in action_mask:
			child: TreeNode = self.root.children.get(action)

			if child is None:
				continue
			if child.visit_count > maximum_visit_count:
				maximum_visit_count = child.visit_count
				best_action = action

		return best_action
	def simulate(self,
							 particle: tuple = None,
							 node: TreeNode = None,
							 steps: int = None,
							 belief: np.ndarray = None) -> float:
		if particle is None:
			raise ValueError("Particle not specified.")
		if node is None:
			raise ValueError("Node not specified.")
		if steps is None:
			raise ValueError("Steps not specified.")
		if belief is None:
			raise ValueError("Belief not specified.")

		steps += 1
		(robot_position,
   	 victim_position) = particle
		action_mask: list = []
		terminated: bool = False
		truncated: bool = False

		if isinstance(robot_position,
                  np.ndarray):
			robot_position = tuple(robot_position.tolist())
		elif isinstance(robot_position,
                    list):
			robot_position = tuple(robot_position)
		elif isinstance(robot_position,
                    tuple):
			if len(robot_position) == 1 and isinstance(robot_position[0],
                                                 (list,
                                                  np.ndarray)):
				robot_position = tuple(robot_position[0])
			elif all(isinstance(coordinate,
                          (int,
                           np.integer)) for coordinate in robot_position):
				pass
			else:
				robot_position = tuple([int(coordinate) if isinstance(coordinate,
                                                              np.integer) else coordinate for coordinate in robot_position])

		victim_position = tuple(victim_position)
		maximum_belief_index: int = belief.argmax()
		maximum_belief_position: np.ndarray = np.unravel_index(maximum_belief_index,
                                                        	 belief.shape)
		maximum_belief_position = tuple(maximum_belief_position)
		distance_to_maximum_belief_before: float = get_shortest_distance(position_1 = robot_position,
																																		 position_2 = maximum_belief_position,
																																		 environment_knowledge = self.environment_knowledge)
		belief_flattened: np.ndarray = belief.flatten()
		positive_belief_flattened_indices: np.ndarray = belief_flattened[belief_flattened > 0]
		belief_shannon_entropy_before: float = get_shannon_entropy(belief_flattened_indices = positive_belief_flattened_indices)
		open_cells_knowledge = get_cells(environment = self.environment_knowledge,
                                   	 type = 'open')
		unknown_cells_knowledge = get_cells(environment = self.environment_knowledge,
																			  type = 'unknown')
		valid_cells_knowledge: set = set(open_cells_knowledge).union(set(unknown_cells_knowledge))

		for (index,
       	 (row,
          column)) in enumerate(actions):
			new_row = row + robot_position[0]
			new_column = column + robot_position[1]

			if (new_row,
       		new_column) in valid_cells_knowledge:
				action_mask.append(index)

		best_action = node.get_best_action(action_mask = action_mask,
                                     	 exploration_constant = self.exploration_constant,
																		 	 seed = self.seed)

		if best_action == ACTION_SENSE:
			beep: int = get_beep(sensor_alpha = self.sensor_alpha,
													 robot_position = robot_position,
													 victim_position = victim_position,
													 seed = self.seed)
			belief = update_sense(sensor_alpha = self.sensor_alpha,
														robot_position = robot_position,
														belief = belief,
														beep = beep)
		else:
			robot_position = self.simulator.move_robot_position(robot_position = robot_position,
															   													action = best_action)
			belief = update_robot_move(robot_position = robot_position,
																 environment_knowledge = self.environment_knowledge,
																 belief = belief)
			if robot_position == victim_position:
				belief = update_found_victim(robot_position = robot_position,
																		 belief = belief)
				terminated = True
		if not terminated:
			victim_position = self.simulator.move_victim_position(victim_position = victim_position,
																 														seed = self.seed)
			belief = update_victim_move(robot_position = robot_position,
																	environment_knowledge = self.environment_knowledge,
																	belief = belief)

			if robot_position == victim_position:
				belief = update_found_victim(robot_position = robot_position,
																		 belief = belief)
				terminated = True
		if not terminated and steps >= self.maximum_depth:
			truncated = True

		particle = (robot_position,
	 							victim_position)
		normalized_step: float = 1.0 / float(self.maximum_depth)
		distance_to_maximum_belief: float = get_shortest_distance(position_1 = robot_position,
																															position_2 = maximum_belief_position,
																															environment_knowledge = self.environment_knowledge)
		distance_to_maximum_belief_reduction: float = distance_to_maximum_belief_before - distance_to_maximum_belief
		maximum_distance_to_maximum_belief: float = 2.0 * (self.environment_size - 1)
		normalized_distance_to_maximum_belief_reduction: float = distance_to_maximum_belief_reduction / maximum_distance_to_maximum_belief
		belief_flattened = belief.flatten()
		positive_belief_flattened_indices = belief_flattened[belief_flattened > 0]
		belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = positive_belief_flattened_indices)
		belief_shannon_entropy_reduction: float = belief_shannon_entropy_before - belief_shannon_entropy
		open_cells: list = get_cells(environment = self.environment_knowledge,
                               	 type = 'open')
		open_cells_count: int = len(open_cells)
		unknown_cells: list = get_cells(environment = self.environment_knowledge,
																		type = 'unknown')
		unknown_cells_count: int = len(unknown_cells)
		open_unknown_cells_count: int = open_cells_count + unknown_cells_count
		no_robot_open_cells_count: int = open_unknown_cells_count - 1
		uniform_probability: float = 1.0 / no_robot_open_cells_count
		belief_flattened_indices: np.ndarray = np.full(no_robot_open_cells_count,
                                                 	 uniform_probability,
                                                   dtype = np.float32)
		maximum_belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = belief_flattened_indices)
		normalized_belief_shannon_entropy_reduction: float = belief_shannon_entropy_reduction / maximum_belief_shannon_entropy
		reward = get_reward(configuration = self.reward_configuration,
												alpha = self.alpha,
												normalized_step = normalized_step,
												normalized_distance_to_maximum_belief_reduction = normalized_distance_to_maximum_belief_reduction,
												normalized_belief_shannon_entropy_reduction = normalized_belief_shannon_entropy_reduction,
												terminated = terminated)
		child = node.children.get(best_action)

		if child is None:
			child = TreeNode()

		node.children[best_action] = child

		if terminated or truncated:
			node.visit_count += 1
			node.total_value += reward

			return reward

		future_reward = self.simulate(particle = particle,
																	node = child,
																	steps = steps,
																	belief = belief)
		total_reward = reward + self.gamma * future_reward
		node.visit_count += 1
		node.total_value += total_reward

		return total_reward
	def update(self,
						 observation: dict = None) -> None:
		if observation is None:
			raise ValueError("Observation not specified.")

		self.environment_knowledge = observation['environment_knowledge'].squeeze()
		self.belief = observation['belief'].squeeze()
		robot_position = observation['robot_position']

		if isinstance(robot_position,
                  np.ndarray):
			robot_position = tuple(robot_position.tolist())
		elif isinstance(robot_position,
                    list):
			robot_position = tuple(robot_position)
		elif isinstance(robot_position,
                    tuple):
			if len(robot_position) == 1 and isinstance(robot_position[0],
                                                 (list,
                                                  np.ndarray)):
				robot_position = tuple(robot_position[0])
			elif all(isinstance(coordinate,
                          (int,
                           np.integer)) for coordinate in robot_position):
				pass
			else:
				robot_position = tuple([int(coordinate) if isinstance(coordinate,
                                                              np.integer) else coordinate for coordinate in robot_position])

		self.root = TreeNode()
		self.simulator = InternalSimulator(environment_knowledge = self.environment_knowledge)
		self.particles = self.get_particles(robot_position = robot_position,
                                      	belief = self.belief)
class POMCPWrapper:
	def __init__(self,
							 vectorized_environment: SubprocVecEnv | DummyVecEnv = None,
							 configuration_path: str = None,
							 alpha: float = None) -> None:
		if vectorized_environment is None:
			raise ValueError("Vectorized environment not specified.")
		if configuration_path is None:
			raise ValueError("Configuration path not specified.")
		if alpha is None:
			raise ValueError("Alpha value not specified.")

		configuration: dict = load_configuration(configuration_path = configuration_path)
		pomcp_configuration: dict = configuration['pomcp']
		self.seed: int = pomcp_configuration['seed']
		self.number_environments: int = vectorized_environment.num_envs
		self.agents: list = [POMCP(configuration_path = configuration_path,
															 alpha = alpha) for _ in range(self.number_environments)]
		self.initialized: list = [False] * self.number_environments
		self.num_timesteps: int = 0
	def predict(self,
							observations: dict = None,
							action_masks: np.ndarray = None) -> np.ndarray:
		if observations is None:
			raise ValueError("Observations not specified.")
		if action_masks is None:
			raise ValueError("Action masks not specified.")
		self.num_timesteps += 1
		actions: list = []

		for environment_number in range(self.number_environments):
			observation = self.get_observation(observations = observations,
																				 environment_number = environment_number)
			action_mask = action_masks[environment_number]

			if self.initialized[environment_number]:
				self.agents[environment_number].update(observation = observation)
			else:
				if self.seed is None:
					self.agents[environment_number].reset(observation = observation,
                                           			seed = self.seed)
				else:
					self.agents[environment_number].reset(observation = observation,
                                           			seed = self.seed + environment_number)

				self.initialized[environment_number] = True

			best_action: int = self.agents[environment_number].get_best_action(action_mask = action_mask)

			actions.append(best_action)

		actions = np.array(actions)

		return actions
	def get_observation(self,
                      observations: dict = None,
                      environment_number: int = None) -> dict:
		if observations is None:
			raise ValueError("Observations not specified.")
		if environment_number is None:
			raise ValueError("Environment number not specified.")

		observation: dict = {key: value[environment_number] if isinstance(value,
																																			(np.ndarray,
																																			list)) and len(value) == self.number_environments else value for (key,
                                                                                                     																		value) in observations.items()}

		return observation
	def reset_environment(self,
												environment_number: int = None) -> None:
		if environment_number is None:
			raise ValueError("Environment number not specified.")

		self.initialized[environment_number] = False
