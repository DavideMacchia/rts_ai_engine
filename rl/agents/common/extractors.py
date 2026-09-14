"""
Feature Extractors for RL Policy Networks
- MLPExtractor: Simple MLP with residual connections (recommended for flat state vectors)
- TransformerExtractor: Transformer-based (use only for sequential/entity-based inputs)
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Any, Optional
from config.config_loader import load_transformer_config


class MLPExtractor(BaseFeaturesExtractor):
    """
    MLP feature extractor with residual connections and LayerNorm.
    Better suited than Transformer for flat, fixed-size state vectors
    (no sequential/temporal structure in the input).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        hidden_dims: tuple = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(nn.LayerNorm(features_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

        print(f"MLPExtractor: {input_dim}D -> {hidden_dims} -> {features_dim}D")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class TransformerExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor.
    Best suited for sequential or entity-based inputs.
    For flat state vectors, prefer MLPExtractor.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        config_path: Optional[str] = None
    ):
        super().__init__(observation_space, features_dim)

        config = load_transformer_config(config_path)
        arch_config = config['transformer_standard']
        init_config = config['initialization']

        input_dim = observation_space.shape[0]

        embedding_dim = arch_config['embedding_dim']
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        pos_encoding_scale = arch_config['positional_encoding_scale']
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1, embedding_dim) * pos_encoding_scale
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=arch_config['d_model'],
            nhead=arch_config['nhead'],
            dim_feedforward=arch_config['dim_feedforward'],
            dropout=arch_config['dropout'],
            activation=arch_config['activation'],
            batch_first=arch_config['batch_first'],
            norm_first=arch_config['norm_first']
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=arch_config['num_layers']
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self._initialize_weights(init_config)

    def _initialize_weights(self, config: Dict[str, Any]):
        method = config['method']
        gain = config['gain']
        bias_constant = config['bias_constant']

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'orthogonal':
                    nn.init.orthogonal_(module.weight, gain=gain)
                elif method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_constant)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.embedding(observations)
        x = x.unsqueeze(1)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)


class MapExtractor(BaseFeaturesExtractor):
    """Feature extractor for the SPATIAL district agent (D28).

    The observation is two different kinds of thing and they are treated as such:

    - `patch`: the district's plot, a stack of feature maps over a (2R+1)² grid. This is
      what a convolution is for — the same "is there forest here" filter applies at every
      tile, so what is learned about one square transfers to all the others. Flattening the
      grid into the scalar vector would throw away the adjacency that the placement rules
      are actually written in terms of (a well is built NEXT TO water).
    - `scalars`: the district summary (stock, population, army, policies) — flat, no
      structure, so an MLP, exactly as in `MLPExtractor`.

    The two are encoded separately and concatenated. The tile head reads the result, so it
    can see both the ground and the state of the economy that is about to be spent on it.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256,
                 channels: int = 64):
        super().__init__(observation_space, features_dim)

        c, h, w = observation_space['patch'].shape
        n_scalars = observation_space['scalars'].shape[0]

        # Stride 1, padding 1: the plot is small (13x13), so nothing is downsampled — every
        # tile keeps its own cell in the output, because the tile head has to choose among
        # them individually. A 1x1 conv then squeezes the channels back down before the
        # flatten: at 64 channels the flattened grid is 10,816 features against the summary's
        # 128, and the district's economy — which is what decides WHAT to build — was simply
        # drowned out. Measured: action-match 50% (against 82% for the flat agent).
        self.cnn = nn.Sequential(
            nn.Conv2d(c, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 16, kernel_size=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.ground = nn.Sequential(
            nn.Linear(16 * h * w, 256), nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_scalars, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 128, features_dim), nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        ground = self.ground(self.cnn(observations['patch']))
        summary = self.mlp(observations['scalars'])
        return self.head(torch.cat([ground, summary], dim=1))
