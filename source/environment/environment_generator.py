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

import numpy as np
import random as rd

def count_get_neighbors(environment: np.ndarray = None,
												row: int = None,
												column: int = None,
												type: str = None) -> tuple:
	if environment is None:
		raise ValueError("Environment not specified.")
	if row is None:
		raise ValueError("Row must be specified.")
	if column is None:
		raise ValueError("Column must be specified.")
	if type not in ['open',
									'closed']:
		raise ValueError("Type must be 'open' or 'closed'.")

	count: int = 0
	neighbors: list = []

	for (delta_row,
			 delta_column) in [(-1,
													0),
												 (1,
													0),
												 (0,
													-1),
												 (0,
													1)]:
		new_row: int = row + delta_row
		new_column: int = column + delta_column

		if 0 <= new_row < environment.shape[0] and 0 <= new_column < environment.shape[1] and environment[new_row,
																																								 				 							new_column] == (0 if type == 'open' else 1 if type == 'closed' else -1):
			count += 1
			neighbors.append((new_row,
                        new_column))

	return (count,
          neighbors)
def find_dead_ends(environment: np.ndarray = None) -> tuple:
	if environment is None:
		raise ValueError("Environment not specified.")

	dead_ends: list = []
	closed_neighbors: dict = {}

	for row in range(environment.shape[0]):
		for column in range(environment.shape[1]):
			if environment[row,
              			 column] == 0 and count_get_neighbors(environment = environment,
																													row = row,
																													column = column,
																													type = 'open')[0] == 1:
				dead_ends.append((row,
                          column))
				closed_neighbors[(row,
                          column)] = count_get_neighbors(environment = environment,
																												 row = row,
																												 column = column,
																												 type = 'closed')[1]

	return (dead_ends,
          closed_neighbors)
def get_one_neighbors(environment: np.ndarray = None,
											type: str = None) -> list:
	if environment is None:
		raise ValueError("Environment not specified.")
	if type not in ['open',
									'closed']:
		raise ValueError("Type must be 'open' or 'closed'.")

	one_neighbors: list = []

	for row in range(environment.shape[0]):
		for column in range(environment.shape[1]):
			if environment[row,
                     column] == 1 and count_get_neighbors(environment = environment,
																													row = row,
																													column = column,
																													type = type)[0] == 1:
				one_neighbors.append((row,
														  column))

	return one_neighbors
def open_half_dead_ends(environment: np.ndarray = None,
												_shuffle: callable = None,
												_choice: callable = None) -> np.ndarray:
	if environment is None:
		raise ValueError("Environment not specified.")
	if _shuffle is None:
		raise ValueError("Shuffle function not specified.")
	if _choice is None:
		raise ValueError("Choice function not specified.")

	(dead_ends,
   closed_neighbors) = find_dead_ends(environment = environment)

	_shuffle(dead_ends)

	half_dead_ends: list = dead_ends[0:len(dead_ends) // 2]

	for (row,
       column) in half_dead_ends:
		neighbors_to_choose: list = closed_neighbors[(row,
                                            			column)]

		if neighbors_to_choose:
			(neighbor_row,
       neighbor_column) = _choice(neighbors_to_choose)
			environment[neighbor_row,
           				neighbor_column] = 0

	return environment
def generate_environment(environment_size: int = None,
												 seed: int | np.random.Generator = None) -> np.ndarray:
	if environment_size is None:
		raise ValueError("Environment size not specified.")
	if seed is None:
		_randint: callable = rd.randint
		_choice: callable = rd.choice
		_shuffle: callable = rd.shuffle
	else:
		_randint: callable = lambda x, y: seed.integers(x,
                                                    y + 1)
		_choice: callable = lambda list: list[seed.integers(len(list))]
		_shuffle: callable = lambda list: seed.shuffle(list)

	environment: np.ndarray = np.ones((environment_size,
																	   environment_size),
																    dtype = np.int8)
	initial_row: int = _randint(0,
                              environment_size - 1)
	initial_column: int = _randint(0,
                                 environment_size - 1)
	environment[initial_row,
       				initial_column] = 0

	while True:
		one_open_neighbors: list = get_one_neighbors(environment = environment,
																					 		   type = 'open')

		if not one_open_neighbors:
			break

		(row,
     column) = _choice(one_open_neighbors)
		environment[row,
                column] = 0

	environment = open_half_dead_ends(environment = environment,
																		_shuffle = _shuffle,
																		_choice = _choice)

	return environment

if __name__ == "__main__":
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description = "Generate and save a random environment")

	parser.add_argument('--environment_size',
											type = int,
											default = 40,
											help = "Environment size")
	parser.add_argument('--seed',
											type = int,
											default = None,
											help = "Seed")

	arguments: argparse.Namespace = parser.parse_args()

	environment: np.ndarray = generate_environment(environment_size = arguments.environment_size,
											 													 seed = arguments.seed)
