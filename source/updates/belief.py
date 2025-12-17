import sys
import os

import numpy as np

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.updates.sensor import get_beep_probability
from source.utilities.helpers import get_cells

def normalize_belief(belief: np.ndarray = None) -> np.ndarray:
	if belief is None:
		raise ValueError("Belief distribution not specified.")

	total_probability: float = np.sum(belief)

	if total_probability > 0:
		belief /= total_probability
	else:
		raise ValueError("Cannot normalize belief distribution with zero total probability")

	return belief
def update_sense(sensor_alpha: float = None,
                 robot_position: tuple = None,
								 belief: np.ndarray = None,
                 beep: bool = None) -> np.ndarray:
	if sensor_alpha is None:
		raise ValueError("Sensor alpha not specified.")
	if robot_position is None:
		raise ValueError("Robot position not specified.")
	if belief is None:
		raise ValueError("Belief distribution not specified.")
	if beep is None:
		raise ValueError("Beep value not specified.")

	for row in range(belief.shape[0]):
		for column in range(belief.shape[1]):
			if belief[row,
								column] > 0:
				beep_probability: float = get_beep_probability(sensor_alpha = sensor_alpha,
                                                       position_1 = robot_position,
																											 position_2 = (row,
																												  					 column))

				if beep:
					belief[row,
								 column] = belief[row,
																	column] * beep_probability
				else:
					belief[row,
								 column] = belief[row,
																	column] * (1.0 - beep_probability)
			else:
				belief[row,
							 column] = 0.0

	belief[robot_position[0],
				 robot_position[1]] = 0.0
	belief = normalize_belief(belief = belief)

	return belief
def update_robot_move(robot_position: tuple = None,
											environment_knowledge: np.ndarray = None,
											belief: np.ndarray = None) -> np.ndarray:
	if robot_position is None:
		raise ValueError("Robot position not specified.")
	if environment_knowledge is None:
		raise ValueError("Environment knowledge not specified.")
	if belief is None:
		raise ValueError("Belief distribution not specified.")

	belief[environment_knowledge == 1] = 0.0
	belief[robot_position[0],
				 robot_position[1]] = 0.0
	belief = normalize_belief(belief = belief)

	return belief
def update_victim_move(robot_position: tuple = None,
  										 environment_knowledge: np.ndarray = None,
											 belief: np.ndarray = None) -> np.ndarray:
	if robot_position is None:
		raise ValueError("Robot position not specified.")
	if environment_knowledge is None:
		raise ValueError("Environment knowledge not specified.")
	if belief is None:
		raise ValueError("Belief distribution not specified.")

	closed_cells_knowledge: list = get_cells(environment = environment_knowledge,
																					 type = 'closed')
	move_actions: list = [(-1,
                         0),
                        (1,
                         0),
                        (0,
                         -1),
                        (0,
                         1)]
	updated_belief: np.ndarray = np.zeros_like(belief)

	for row in range(environment_knowledge.shape[0]):
		for column in range(environment_knowledge.shape[1]):
			neighbors: list = []

			for (delta_row,
        	 delta_column) in move_actions:
				new_row: int = row + delta_row
				new_column: int = column + delta_column

				if 0 <= new_row < environment_knowledge.shape[0] and 0 <= new_column < environment_knowledge.shape[1]:
					if (new_row,
         		  new_column) not in closed_cells_knowledge:
						neighbors.append((new_row,
                        			new_column))
					else:
						continue

			destinations: list = [(row,
                       			 column)] + neighbors
			probability: float = 1 / len(destinations)

			for (destination_row,
        	 destination_column) in destinations:
				updated_belief[destination_row,
               		 		 destination_column] += belief[row,
																						 				 column] * probability

	updated_belief[robot_position[0],
         		 		 robot_position[1]] = 0.0
	belief = normalize_belief(belief = updated_belief)

	return belief
def update_found_victim(robot_position: tuple = None,
												belief: np.ndarray = None) -> np.ndarray:
	if robot_position is None:
		raise ValueError("Robot position not specified.")
	if belief is None:
		raise ValueError("Belief distribution not specified.")

	belief.fill(0.0)

	belief[robot_position[0],
				 robot_position[1]] = 1.0

	return belief
