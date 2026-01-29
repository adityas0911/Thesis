def get_reward(configuration: dict = None,
               alpha: float = None,
               normalized_step: float = None,
               normalized_distance_to_maximum_belief_reduction: float = None,
               normalized_belief_shannon_entropy_reduction: float = None,
               terminated: bool = None,
               truncated: bool = None) -> float:
  if configuration is None:
    raise ValueError("Configuration not specified.")
  if alpha is None:
    raise ValueError("Alpha not specified.")
  if normalized_step is None:
    raise ValueError("Normalized step not specified.")
  if normalized_distance_to_maximum_belief_reduction is None:
    raise ValueError("Normalized distance to maximum belief reduction not specified.")
  if normalized_belief_shannon_entropy_reduction is None:
    raise ValueError("Normalized belief Shannon entropy reduction not specified.")
  if terminated is None:
    raise ValueError("Terminated flag not specified.")

  step_weight: float = configuration['step_weight']
  distance_reduction_weight: float = configuration['distance_reduction_weight']
  entropy_reduction_weight: float = configuration['entropy_reduction_weight']
  terminated_weight: float = configuration['terminated_weight']
  truncated_weight: float = configuration['truncated_weight']
  step_term: float = step_weight * normalized_step
  distance_term: float = distance_reduction_weight * normalized_distance_to_maximum_belief_reduction
  belief_term: float = entropy_reduction_weight * normalized_belief_shannon_entropy_reduction
  sparse_reward: float = -step_term

  if terminated:
    sparse_reward += terminated_weight
  if truncated:
    sparse_reward -= truncated_weight

  shaped_reward: float = alpha * distance_term + (1 - alpha) * belief_term
  reward: float = sparse_reward + shaped_reward

  return reward
