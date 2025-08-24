"""
cnn_encoder_co2storage.py

Deeper CNN feature extractor for PPO agent in GCS storage optimization.

- Input shape: (48, 60, 60) state tensor from saturation + normalized pressure
- Architecture: 3 conv blocks (64 → 128 → 256), stride-2 downsampling, adaptive pool to 4×4
- Output: 512-dim latent vector passed to actor/critic MLPs


Usage:
    policy_kwargs = dict(
        features_extractor_class=LargerCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
"""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LargerCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        """
        CNN feature extractor for large 3D reservoir observations.

        Parameters
        ----------
        observation_space : gym.Space
            The observation space, expected shape (48, 60, 60)
        features_dim : int, optional
            Dimensionality of the output latent vector, default 512
        """
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Output block
            nn.AdaptiveAvgPool2d((4, 4)),  # → (256, 4, 4)
            nn.Flatten()  # → 256*4*4 = 4096
        )

        # Infer flattened size dynamically
        with torch.no_grad():
            sample_input = torch.zeros((1, n_input_channels, observation_space.shape[1], observation_space.shape[2]))
            n_flatten = self.cnn(sample_input).shape[1]

        # Project to latent vector z (default: 512)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))
