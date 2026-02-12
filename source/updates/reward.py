def get_reward(configuration: dict = None,
               alpha: float = None,
               normalized_step: float = None,
               terminated: bool = None,
               truncated: bool = None) -> float:
  if configuration is None:
    raise ValueError("Configuration not specified.")
  if alpha is None:
    raise ValueError("Alpha not specified.")
  if normalized_step is None:
    raise ValueError("Normalized step not specified.")
  if terminated is None:
    raise ValueError("Terminated flag not specified.")

  step_weight: float = configuration['step_weight']
  terminated_weight: float = configuration['terminated_weight']
  truncated_weight: float = configuration['truncated_weight']
  step_term: float = step_weight * normalized_step
  reward: float = -step_term

  if terminated:
    reward += terminated_weight
  if truncated:
    reward -= truncated_weight

  return reward
