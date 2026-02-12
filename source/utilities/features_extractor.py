import torch

import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FeaturesExtractor(BaseFeaturesExtractor):
  def __init__(self,
               observation_space: spaces.Dict = None,
               cnn_channels: tuple = None,
               cnn_kernel_sizes: tuple = None,
               cnn_strides: tuple = None,
               cnn_embedding_dimensions: int = None,
               scalar_hidden_sizes: tuple = None,
               scalar_embedding_dimensions: int = None):
    if observation_space is None:
      raise ValueError("Observation space not specified.")
    if cnn_channels is None:
      raise ValueError("CNN channels not specified.")
    if cnn_kernel_sizes is None:
      raise ValueError("CNN kernel sizes not specified.")
    if cnn_strides is None:
      raise ValueError("CNN strides not specified.")
    if cnn_embedding_dimensions is None:
      raise ValueError("CNN embedding dimensions not specified.")
    if scalar_hidden_sizes is None:
      raise ValueError("Scalar hidden sizes not specified.")
    if scalar_embedding_dimensions is None:
      raise ValueError("Scalar embedding dimensions not specified.")

    super().__init__(observation_space,
                     features_dim = 1)

    global_map_space: spaces.Box = observation_space.spaces['global_map']
    alpha_space: spaces.Box = observation_space.spaces['alpha'].shape[0]
    number_channels: int = global_map_space.shape[-1]
    (height,
     width) = global_map_space.shape[0:2]
    layers: list[nn.Module] = []
    previous: int = number_channels

    for (channel,
         kernel_size,
         stride) in zip(cnn_channels,
                        cnn_kernel_sizes,
                        cnn_strides):
      padding: int = kernel_size // 2

      layers.append(nn.Conv2d(previous,
                              channel,
                              kernel_size,
                              stride,
                              padding))
      layers.append(nn.ReLU())

      previous = channel

    self.cnn = nn.Sequential(*layers)

    with torch.no_grad():
      dummy: torch.Tensor = torch.zeros(1,
                                        number_channels,
                                        height,
                                        width)
      cnn_output_dimensions: int = self.cnn(dummy).view(1,
                                                        -1).shape[1]

    self.cnn_head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(cnn_output_dimensions,
                                            cnn_embedding_dimensions),
                                  nn.ReLU())
    self.scalar_multilayer_perceptron = nn.Sequential(nn.Linear(alpha_space,
                                                                scalar_hidden_sizes[0]),
                                                      nn.ReLU(),
                                                      nn.Linear(scalar_hidden_sizes[0],
                                                                scalar_embedding_dimensions),
                                                      nn.ReLU())
    self._features_dim = cnn_embedding_dimensions + scalar_embedding_dimensions
  def forward(self,
              observations: dict = None) -> torch.Tensor:
    if observations is None:
      raise ValueError("Observations not specified.")

    global_map: torch.Tensor = observations['global_map'].permute(0,
                                                                  3,
                                                                  1,
                                                                  2)
    spatial_latent: torch.Tensor = self.cnn_head(self.cnn(global_map))
    scalar = observations['alpha']
    scalar_latent = self.scalar_multilayer_perceptron(scalar)
    output: torch.Tensor = torch.cat([spatial_latent,
                                      scalar_latent],
                                     dim = 1)

    return output
