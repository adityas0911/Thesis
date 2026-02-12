import sys
import os

import numpy as np

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.utilities.helpers import get_manhattan_distance

def get_beep_probability(alpha: float = None,
                         position_1: tuple = None,
                         position_2: tuple = None) -> float:
  if alpha is None:
    raise ValueError("Alpha not specified.")
  if position_1 is None:
    raise ValueError("Position 1 not specified.")
  if position_2 is None:
    raise ValueError("Position 2 not specified.")

  manhattan_distance: int = get_manhattan_distance(position_1 = position_1,
                                                   position_2 = position_2)
  beep_probability: float = np.exp(-alpha * (manhattan_distance - 1))

  return beep_probability
def get_beep(alpha: float = None,
             robot_position: tuple = None,
             victim_position: tuple = None,
             seed: np.random.Generator = None) -> int:
  if alpha is None:
    raise ValueError("Alpha not specified.")
  if robot_position is None:
    raise ValueError("Robot position not specified.")
  if victim_position is None:
    raise ValueError("Victim position not specified.")

  beep_probability: float = get_beep_probability(alpha = alpha,
                                                 position_1 = robot_position,
                                                 position_2 = victim_position)
  if seed is None:
    beep: int = int(np.random.rand() < beep_probability)
  else:
    beep: int = int(seed.random() < beep_probability)

  return beep
