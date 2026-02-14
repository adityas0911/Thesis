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
																			get_cells)

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

def format_position(position: object = None) -> tuple[int,
																										  int]:
	if position is None:
		raise ValueError("Position not specified.")

	if isinstance(position,
								np.ndarray):
		position = position.tolist()
	if isinstance(position,
								list):
		if len(position) == 1 and isinstance(position[0],
																				 (list,
																					np.ndarray)):
			position = position[0]

		position = tuple(position)
	elif isinstance(position,
								  tuple):
		if len(position) == 1 and isinstance(position[0],
																				 (list,
																					np.ndarray)):
			position = tuple(position[0])

	return (int(position[0]),
					int(position[1]))
def get_state_key(robot_position: tuple[int,
																				int] = None,
									environment_knowledge: np.ndarray = None) -> tuple:
	if robot_position is None:
		raise ValueError("Robot position not specified.")
	if environment_knowledge is None:
		raise ValueError("Environment knowledge not specified.")

	state_key: tuple = (robot_position,
											environment_knowledge.tobytes())

	return state_key
def get_observation_key(observation: dict = None) -> tuple:
	if observation is None:
		raise ValueError("Observation not specified.")

	robot_position: tuple[int,
												int] = format_position(observation['robot_position'])
	environment_knowledge: np.ndarray = observation['environment_knowledge'].squeeze().astype(np.int8,
																																														copy = False)
	observation_key: tuple = get_state_key(robot_position = robot_position,
																				 environment_knowledge = environment_knowledge)

	return observation_key

class ActionNode:
	def __init__(self) -> None:
		self.visit_count: int = 0
		self.total_value: float = 0.0
		self.children: dict[tuple,
                      	TreeNode] = {}
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
class TreeNode:
	def __init__(self) -> None:
		self.visit_count: int = 0
		self.total_value: float = 0.0
		self.children: dict[int,
                      	ActionNode] = {}
	def get_best_action(self,
											action_mask: list = None,
											exploration_constant: float = None,
           						seed: np.random.Generator = None) -> int:
		if action_mask is None:
			raise ValueError("Action mask not specified.")
		if exploration_constant is None:
			raise ValueError("Exploration constant not specified.")
		if seed is None:
			raise ValueError("Seed not specified.")

		seed.shuffle(action_mask)

		parent_visit_count: int = max(self.visit_count,
                                	1)
		best_value: float = -float('inf')
		best_action: int = action_mask[0]

		for action in action_mask:
			child: ActionNode = self.children.get(action)

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
class InternalSimulator:
	def __init__(self,
							 environment_size: int = None,
							 vision_range: int = None) -> None:
		if environment_size is None:
			raise ValueError("Environment size not specified.")
		if vision_range is None:
			raise ValueError("Vision range not specified.")

		self.environment_size: int = environment_size
		self.vision_range: int = vision_range
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
	def update_environment_knowledge(self,
																	 environment_knowledge: np.ndarray = None,
																	 robot_position: tuple[int,
																												 int] = None,
																	 environment: np.ndarray = None) -> None:
		if environment_knowledge is None:
			raise ValueError("Environment knowledge not specified.")
		if robot_position is None:
			raise ValueError("Robot position not specified.")
		if environment is None:
			raise ValueError("Environment not specified.")

		for row in range(-self.vision_range,
										 self.vision_range + 1):
			for column in range(-self.vision_range,
											    self.vision_range + 1):
				global_row: int = row + robot_position[0]
				global_column: int = column + robot_position[1]

				if 0 <= global_row < self.environment_size and 0 <= global_column < self.environment_size:
					environment_knowledge[global_row,
																global_column] = environment[global_row,
																														 global_column]
	def move_victim_position(self,
													 victim_position: tuple[int,
                                     							int] = None,
													 environment: np.ndarray = None,
													 seed: np.random.Generator = None) -> tuple[int,
                                                         							int]:
		if victim_position is None:
			raise ValueError("Victim position not specified.")
		if environment is None:
			raise ValueError("Environment not specified.")
		if seed is None:
			raise ValueError("Seed not specified.")

		valid_actions: list = []
		open_cells: np.ndarray = get_cells(environment = environment,
																		 	 type = 'open')

		for (row,
         column) in actions:
			new_row: int = row + victim_position[0]
			new_column: int = column + victim_position[1]

			if (new_row,
					new_column) in open_cells:
				valid_actions.append((new_row,
                       		 		new_column))

		victim_position: tuple[int,
                           int] = tuple(seed.choice(valid_actions))

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
		self.vision_range: int = environment_configuration['vision_range']
		self.number_simulations: int = pomcp_configuration['number_simulations']
		self.maximum_depth: int = pomcp_configuration['maximum_depth']
		self.exploration_constant: float = pomcp_configuration['exploration_constant']
		self.number_particles: int = pomcp_configuration['number_particles']
		self.gamma: float = train_configuration['gamma']
		self.seed: np.random.Generator = None
		self.simulator: InternalSimulator = InternalSimulator(environment_size = self.environment_size,
																													vision_range = self.vision_range)
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
		self.environment_knowledge = observation['environment_knowledge'].squeeze().astype(np.int8)
		self.belief = observation['belief'].squeeze().astype(np.float32)
		robot_position: tuple[int,
                          int] = format_position(observation['robot_position'])
		self.root = TreeNode()
		self.particles = self.get_particles(robot_position = robot_position,
                                        belief = self.belief,
																				environment_knowledge = self.environment_knowledge)
	def get_particles(self,
										robot_position: tuple[int,
                                					int] = None,
										belief: np.ndarray = None,
										environment_knowledge: np.ndarray = None) -> list:
		if robot_position is None:
			raise ValueError("Robot position not specified.")
		if belief is None:
			raise ValueError("Belief grid not specified.")
		if environment_knowledge is None:
			raise ValueError("Environment knowledge not specified.")

		particles: list = []
		belief = normalize_belief(belief = belief)
		belief_flattened: np.ndarray = belief.flatten()
		positive_belief_flattened_indices: np.ndarray = np.where(belief_flattened > 0)
		valid_victim_indices: np.ndarray = positive_belief_flattened_indices[0]
		valid_victim_probabilities: np.ndarray = belief_flattened[valid_victim_indices]

		for _ in range(self.number_particles):
			victim_position_index: int = int(self.seed.choice(a = valid_victim_indices,
                                                    		p = valid_victim_probabilities))
			victim_position: tuple[int,
                          	 int] = tuple(np.unravel_index(victim_position_index,
                                                           belief.shape))
			environment: np.ndarray = self.get_environment(environment_knowledge = environment_knowledge)

			particles.append((robot_position,
                     		victim_position,
												environment))

		return particles
	def get_environment(self,
											environment_knowledge: np.ndarray = None) -> np.ndarray:
		if environment_knowledge is None:
			raise ValueError("Environment knowledge not specified.")

		environment: np.ndarray = environment_knowledge.copy()
		open_cells: np.ndarray = get_cells(environment = environment,
																			 type = 'open')
		closed_cells: np.ndarray = get_cells(environment = environment,
																				 type = 'closed')
		unknown_cells: np.ndarray = get_cells(environment = environment,
																					type = 'unknown')
		closed_cells_count: int = len(closed_cells)
		open_cells_count: int = len(open_cells)
		known_cells_count: int = closed_cells_count + open_cells_count
		closed_cell_probability: float = float(closed_cells_count) / float(known_cells_count)

		for unknown_cell in unknown_cells:
			cell_probability: float = self.seed.random()

			if cell_probability < closed_cell_probability:
				environment[unknown_cell] = 1
			else:
				environment[unknown_cell] = 0

		return environment
	def get_best_action(self,
                      action_mask: np.ndarray = None) -> int:
		if action_mask is None:
			raise ValueError("Action mask not specified.")
		if self.root is None:
			self.root = TreeNode()

		action_mask: list = [index for (index,
                                    valid) in enumerate(action_mask) if valid]

		self.seed.shuffle(action_mask)

		maximum_visit_count: int = -1
		best_action: int = action_mask[0]

		for _ in range(self.number_simulations):
			belief: np.ndarray = self.belief.copy()
			environment_knowledge: np.ndarray = self.environment_knowledge.copy()
			number_particles: int = len(self.particles)
			particle: tuple = self.particles[self.seed.choice(a = number_particles)]
			_: float = self.simulate(particle = particle,
															 tree_node = self.root,
															 steps = 0,
															 belief = belief,
															 environment_knowledge = environment_knowledge)
		for action in action_mask:
			action_node: ActionNode = self.root.children.get(action)

			if action_node is None:
				continue
			if action_node.visit_count > maximum_visit_count:
				maximum_visit_count = action_node.visit_count
				best_action = action

		return best_action
	def simulate(self,
							 particle: tuple = None,
							 tree_node: TreeNode = None,
							 steps: int = None,
							 belief: np.ndarray = None,
							 environment_knowledge: np.ndarray = None) -> float:
		if particle is None:
			raise ValueError("Particle not specified.")
		if tree_node is None:
			raise ValueError("Tree node not specified.")
		if steps is None:
			raise ValueError("Steps not specified.")
		if belief is None:
			raise ValueError("Belief not specified.")
		if environment_knowledge is None:
			raise ValueError("Environment knowledge not specified.")

		steps += 1
		(robot_position,
   	 victim_position,
		 environment) = particle
		action_mask: list = []
		terminated: bool = False
		truncated: bool = False
		maximum_belief_index: int = int(belief.argmax())
		maximum_belief_position: tuple[int,
																	 int] = tuple(np.unravel_index(maximum_belief_index,
																																 belief.shape))
		distance_to_maximum_belief_before: float = get_shortest_distance(position_1 = robot_position,
																																		 position_2 = maximum_belief_position,
																																		 environment_knowledge = environment_knowledge)
		belief_flattened: np.ndarray = belief.flatten()
		positive_belief_flattened_indices: np.ndarray = belief_flattened[belief_flattened > 0]
		belief_shannon_entropy_before: float = get_shannon_entropy(belief_flattened_indices = positive_belief_flattened_indices)
		open_cells_knowledge: np.ndarray = get_cells(environment = environment_knowledge,
												 												 type = 'open')

		for (index,
       	 (row,
          column)) in enumerate(actions):
			new_row: int = row + robot_position[0]
			new_column: int = column + robot_position[1]

			if (new_row,
					new_column) in open_cells_knowledge:
				action_mask.append(index)

		best_action: int = tree_node.get_best_action(action_mask = action_mask,
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
			self.simulator.update_environment_knowledge(environment_knowledge = environment_knowledge,
																									robot_position = robot_position,
																									environment = environment)
			belief = update_robot_move(robot_position = robot_position,
																 environment_knowledge = environment_knowledge,
																 belief = belief)

			if robot_position == victim_position:
				belief = update_found_victim(robot_position = robot_position,
																		 belief = belief)
				terminated = True
		if not terminated:
			victim_position = self.simulator.move_victim_position(victim_position = victim_position,
																 														environment = environment,
																 														seed = self.seed)
			belief = update_victim_move(robot_position = robot_position,
																	environment_knowledge = environment_knowledge,
																	belief = belief)

			if robot_position == victim_position:
				belief = update_found_victim(robot_position = robot_position,
																		 belief = belief)
				terminated = True
		if not terminated and steps >= self.maximum_depth:
			truncated = True

		particle = (robot_position,
	 							victim_position,
								environment)
		normalized_step: float = 1.0 / float(self.maximum_depth)
		distance_to_maximum_belief: float = get_shortest_distance(position_1 = robot_position,
																															position_2 = maximum_belief_position,
																															environment_knowledge = environment_knowledge)
		distance_to_maximum_belief_reduction: int = distance_to_maximum_belief_before - distance_to_maximum_belief
		maximum_distance_to_maximum_belief: int = 2 * (self.environment_size - 1)
		normalized_distance_to_maximum_belief_reduction: float = float(distance_to_maximum_belief_reduction) / float(maximum_distance_to_maximum_belief)
		belief_flattened = belief.flatten()
		positive_belief_flattened_indices = belief_flattened[belief_flattened > 0]
		belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = positive_belief_flattened_indices)
		belief_shannon_entropy_reduction: float = belief_shannon_entropy_before - belief_shannon_entropy
		open_cells: np.ndarray = get_cells(environment = environment,
													 						 type = 'open')
		open_cells_count: int = len(open_cells)
		no_robot_open_cells_count: int = open_cells_count - 1
		uniform_probability: float = 1.0 / float(no_robot_open_cells_count)
		belief_flattened_indices: np.ndarray = np.full(no_robot_open_cells_count,
                                                 	 uniform_probability,
                                                   dtype = np.float32)
		maximum_belief_shannon_entropy: float = get_shannon_entropy(belief_flattened_indices = belief_flattened_indices)
		normalized_belief_shannon_entropy_reduction: float = belief_shannon_entropy_reduction / maximum_belief_shannon_entropy
		reward: float = get_reward(configuration = self.reward_configuration,
															 alpha = self.alpha,
															 normalized_step = normalized_step,
															 normalized_distance_to_maximum_belief_reduction = normalized_distance_to_maximum_belief_reduction,
															 normalized_belief_shannon_entropy_reduction = normalized_belief_shannon_entropy_reduction,
															 terminated = terminated)
		action_node: ActionNode = tree_node.children.get(best_action)

		if action_node is None:
			action_node: ActionNode = ActionNode()
			tree_node.children[best_action] = action_node

		state_key: tuple = get_state_key(robot_position = robot_position,
																		 environment_knowledge = environment_knowledge)
		child: TreeNode = action_node.children.get(state_key)

		if child is None:
			child: TreeNode = TreeNode()
			action_node.children[state_key] = child
		if terminated or truncated:
			tree_node.visit_count += 1
			action_node.visit_count += 1
			tree_node.total_value += reward
			action_node.total_value += reward

			return reward

		future_reward: float = self.simulate(particle = particle,
																				 tree_node = child,
																				 steps = steps,
																				 belief = belief,
																				 environment_knowledge = environment_knowledge.copy())
		total_reward: float = reward + self.gamma * future_reward
		tree_node.visit_count += 1
		action_node.visit_count += 1
		tree_node.total_value += total_reward
		action_node.total_value += total_reward

		return total_reward
	def reroot(self,
						 action: int = None,
						 observation: dict = None) -> None:
		if action is None:
			raise ValueError("Action not specified.")
		if observation is None:
			raise ValueError("Observation not specified.")
		if self.root is None:
			self.root = TreeNode()

			return

		observation_key: tuple = get_observation_key(observation = observation)
		action_node: ActionNode = self.root.children.get(action)

		if action_node is None:
			self.root = TreeNode()

			return

		next_tree_node: TreeNode = action_node.children.get(observation_key)

		if next_tree_node is None:
			self.root = TreeNode()

			return

		self.root = next_tree_node
	def update(self,
						 observation: dict = None) -> None:
		if observation is None:
			raise ValueError("Observation not specified.")

		self.environment_knowledge = observation['environment_knowledge'].squeeze().astype(np.int8)
		self.belief = observation['belief'].squeeze().astype(np.float32)
		robot_position: tuple[int,
                        	int] = format_position(observation['robot_position'])

		if self.root is None:
			self.root = TreeNode()

		self.particles = self.get_particles(robot_position = robot_position,
                                      	belief = self.belief,
																				environment_knowledge = self.environment_knowledge)
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
		self.agents: list[POMCP] = [POMCP(configuration_path = configuration_path,
															 				alpha = alpha) for _ in range(self.number_environments)]
		self.initialized: list = [False] * self.number_environments
		self.num_timesteps: int = 0
		self.actions: list = None
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
			observation: dict = self.get_observation(observations = observations,
																				 			 environment_number = environment_number)
			action_mask: np.ndarray = action_masks[environment_number]

			if not self.initialized[environment_number]:
				if self.seed is None:
					self.agents[environment_number].reset(observation = observation,
                                           			seed = self.seed)
				else:
					self.agents[environment_number].reset(observation = observation,
                                           			seed = self.seed + environment_number)

				self.initialized[environment_number] = True

			best_action: int = self.agents[environment_number].get_best_action(action_mask = action_mask)
			actions.append(best_action)

		self.actions = np.array(actions,
														dtype = np.int64)

		return self.actions
	def get_observation(self,
                      observations: dict = None,
                      environment_number: int = None) -> dict:
		if observations is None:
			raise ValueError("Observations not specified.")
		if environment_number is None:
			raise ValueError("Environment number not specified.")

		observation: dict = {key: value[environment_number] for (key,
                                                             value) in observations.items()}

		return observation
	def step(self,
           dones: np.ndarray = None,
					 observations: dict = None,
           informations: list = None) -> None:
		if dones is None:
			raise ValueError("Dones not specified.")
		if observations is None:
			raise ValueError("Observations not specified.")
		if informations is None:
			raise ValueError("Informations not specified.")

		for environment_number in range(self.number_environments):
			done: bool = dones[environment_number]
			information: dict = informations[environment_number]
			action: int = self.actions[environment_number]
			reset: bool = information['success'] or information['failure'] or done

			if reset:
				self.initialized[environment_number] = False

				continue

			observation: dict = self.get_observation(observations = observations,
																							 environment_number = environment_number)
			self.agents[environment_number].reroot(action = action,
																					 	 observation = observation)
			self.agents[environment_number].update(observation = observation)
