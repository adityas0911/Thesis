from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeaturesExtractor(BaseFeaturesExtractor):
  def __init__(self,
							 observation_space: spaces.Dict,
							 use_coordinate_channels: bool,
							 add_robot_position_channel: bool,
							 convolution_channels: Tuple[int,
																					 int],
							 global_embedding_dimension: int,
							 scalar_embedding_dimension: int,):
    if observation_space is None:
      raise ValueError("Observation space not specified.")
    if use_coordinate_channels is None:
      raise ValueError("Use coordinate channels not specified.")
    if add_robot_position_channel is None:
      raise ValueError("Add robot position channel not specified.")
    if convolution_channels is None:
      raise ValueError("Convolution channels not specified.")
    if global_embedding_dimension is None:
      raise ValueError("Global embedding dimension not specified.")
    if scalar_embedding_dimension is None:
      raise ValueError("Scalar embedding dimension not specified.")

    super().__init__(observation_space = observation_space,
                     features_dim = 1)

    if not isinstance(observation_space,
                      spaces.Dict):
      raise TypeError("FeaturesExtractor expects observation_space to be gymnasium.spaces.Dict")

    belief_shape: Tuple[int,
                        int,
                        int] = observation_space.spaces["belief"].shape
    environment_knowledge_shape: Tuple[int,
                                       int,
                                       int] = observation_space.spaces["environment_knowledge"].shape

    (belief_height,
     belief_width,
     belief_channels) = belief_shape
    (environment_knowledge_height,
     environment_knowledge_width,
     environment_knowledge_channels) = environment_knowledge_shape

    if (belief_height,
        belief_width) != (environment_knowledge_height,
                          environment_knowledge_width):
      raise ValueError("belief and environment_knowledge must have same (height, width)")
    if belief_channels != 1 or environment_knowledge_channels != 1:
      raise ValueError("belief and environment_knowledge must be single-channel grids (height, width, 1)")

    self.height: int = belief_height
    self.width: int = belief_width
    self.use_coordinate_channels: bool = use_coordinate_channels
    self.add_robot_position_channel: bool = add_robot_position_channel
    input_channels: int = 2

    if self.add_robot_position_channel:
      input_channels += 1
    if self.use_coordinate_channels:
      input_channels += 2

    self.spatial_encoder: nn.Sequential = nn.Sequential(nn.Conv2d(input_channels,
                                                                  convolution_channels[0],
                                                                  kernel_size = 3,
                                                                  padding = 1),
																												nn.ReLU(),
																												nn.Conv2d(convolution_channels[0],
                                      														convolution_channels[1],
                                                    							kernel_size = 3,
																																	padding = 1),
																												nn.ReLU(),
																												nn.Flatten())

    with torch.no_grad():
      dummy: torch.Tensor = torch.zeros(1,
                                        input_channels,
                                        self.height,
                                        self.width)
      flattened_spatial_dimension: int = int(self.spatial_encoder(dummy).shape[1])

    self.global_projection: nn.Sequential = nn.Sequential(nn.Linear(flattened_spatial_dimension,
                                                                    global_embedding_dimension),
      																										nn.ReLU())
    self.scalar_encoder: nn.Sequential = nn.Sequential(nn.Linear(6,
                                                                 64),
																											 nn.ReLU(),
																											 nn.Linear(64,
                                      													 scalar_embedding_dimension),
																											 nn.ReLU())
    self.features_dim: int = global_embedding_dimension + scalar_embedding_dimension
    self.coordinate_cache: Dict[Tuple[torch.device,
                                      torch.dtype],
                                torch.Tensor] = {}
  def get_coordinate_channels(self,
                              device: torch.device,
                              dtype: torch.dtype) -> torch.Tensor:
    cache_key: Tuple[torch.device,
                     torch.dtype] = (device,
                                     dtype)

    if cache_key not in self.coordinate_cache:
      row_coordinates: torch.Tensor = torch.linspace(start = -1.0,
																										 end = 1.0,
																										 steps = self.height,
																										 device = device,
																										 dtype = dtype)
      column_coordinates: torch.Tensor = torch.linspace(start = -1.0,
																												end = 1.0,
																												steps = self.width,
																												device = device,
																												dtype = dtype)
      (row_grid,
       column_grid) = torch.meshgrid(row_coordinates,
                                     column_coordinates,
                                     indexing = "ij")
      coordinate_channels: torch.Tensor = torch.stack([row_grid,
                                                       column_grid],
                                                      dim = 0).unsqueeze(0)
      self.coordinate_cache[cache_key] = coordinate_channels

    coordinate_channels: torch.Tensor = self.coordinate_cache[cache_key]

    return coordinate_channels
  def forward(self, observations: Dict[str,
                                       torch.Tensor]) -> torch.Tensor:
    belief_grid: torch.Tensor = observations["belief"].float().permute(0,
                                                                       3,
                                                                       1,
                                                                       2)
    environment_knowledge_grid: torch.Tensor = observations["environment_knowledge"].float().permute(0,
                                                                                                     3,
                                                                                                     1,
                                                                                                     2)
    spatial_tensor: torch.Tensor = torch.cat([belief_grid,
                                              environment_knowledge_grid],
                                             dim = 1)

    if self.add_robot_position_channel:
      robot_position_int: torch.Tensor = observations["robot_position"].long()
      batch_size: int = int(robot_position_int.shape[0])
      robot_channel: torch.Tensor = torch.zeros((batch_size,
                                                 1,
                                                 self.height,
                                                 self.width),
																								device = robot_position_int.device,
																								dtype = torch.float32)
      robot_rows: torch.Tensor = torch.clamp(robot_position_int[:,
                                                                0],
                                             0,
                                             self.height - 1)
      robot_columns: torch.Tensor = torch.clamp(robot_position_int[:,
                                                                   1],
                                                0,
                                                self.width - 1)
      robot_channel[torch.arange(batch_size,
                                 device = robot_position_int.device),
                    						 0,
                           			 robot_rows,
                               	 robot_columns] = 1.0
      spatial_tensor = torch.cat([spatial_tensor,
                                  robot_channel],
                                 dim = 1)

    if self.use_coordinate_channels:
      coordinate_channels: torch.Tensor = self.get_coordinate_channels(spatial_tensor.device,
                                                                       spatial_tensor.dtype)
      coordinate_channels: torch.Tensor = coordinate_channels.repeat(spatial_tensor.shape[0],
                                                                     1,
                                                                     1,
                                                                     1)
      spatial_tensor: torch.Tensor = torch.cat([spatial_tensor,
                                                coordinate_channels],
                                               dim = 1)

    global_features: torch.Tensor = self.global_projection(self.spatial_encoder(spatial_tensor))
    robot_position: torch.Tensor = observations["robot_position"].float()
    distance_to_maximum_belief: torch.Tensor = observations["distance_to_maximum_belief"].float()
    distance_to_maximum_belief_reduction: torch.Tensor = observations["distance_to_maximum_belief_reduction"].float()
    belief_shannon_entropy: torch.Tensor = observations["belief_shannon_entropy"].float()
    belief_shannon_entropy_reduction: torch.Tensor = observations["belief_shannon_entropy_reduction"].float()
    scalar_tensor: torch.Tensor = torch.cat([robot_position,
																						 distance_to_maximum_belief,
																						 distance_to_maximum_belief_reduction,
																						 belief_shannon_entropy,
																						 belief_shannon_entropy_reduction],
																						dim = 1)
    scalar_features: torch.Tensor = self.scalar_encoder(scalar_tensor)

    return torch.cat([global_features,
                      scalar_features],
                     dim = 1)
