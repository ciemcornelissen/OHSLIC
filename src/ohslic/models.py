"""Model definitions for the OHSLIC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    shared_hidden_sizes: Sequence[int]
    classification_hidden_sizes: Sequence[int]
    regression_hidden_sizes: Sequence[int]
    dropout_shared: float = 0.0
    dropout_classification: float = 0.0
    dropout_regression: float = 0.0
    num_classes: int = 2
    use_cnn: bool = False
    activation_output_reg: bool = False


class MultiTaskMLP(nn.Module):
    """Multi-task network with shared trunk and classification/regression heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_cnn = config.use_cnn

        if config.use_cnn:
            in_channels = 1
            layers = []
            current_length = config.input_size
            for hidden_size in config.shared_hidden_sizes:
                layers.extend(
                    [
                        nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(p=config.dropout_shared),
                        nn.MaxPool1d(kernel_size=2, stride=2),
                    ]
                )
                in_channels = hidden_size
                current_length //= 2
            self.shared_layers = nn.Sequential(*layers)
            self.flatten_size = max(current_length, 1) * config.shared_hidden_sizes[-1]
        else:
            in_features = config.input_size
            layers = []
            for hidden_size in config.shared_hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(in_features, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(p=config.dropout_shared),
                    ]
                )
                in_features = hidden_size
            self.shared_layers = nn.Sequential(*layers)
            self.flatten_size = config.shared_hidden_sizes[-1]

        self.classification_head = _build_head(
            self.flatten_size,
            config.classification_hidden_sizes,
            config.dropout_classification,
            config.num_classes,
        )
        self.regression_head1 = _build_head(
            self.flatten_size,
            config.regression_hidden_sizes,
            config.dropout_regression,
            1,
        )
        self.regression_head2 = _build_head(
            self.flatten_size,
            config.regression_hidden_sizes,
            config.dropout_regression,
            1,
        )
        self.regression_head3 = _build_head(
            self.flatten_size,
            config.regression_hidden_sizes,
            config.dropout_regression,
            1,
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        if self.use_cnn:
            x = x.view(x.size(0), 1, -1)
        shared_output = self.shared_layers(x)
        if self.use_cnn:
            shared_output = torch.flatten(shared_output, start_dim=1)

        classification_output = self.classification_head(shared_output)
        regression_output1 = self.regression_head1(shared_output)
        regression_output2 = self.regression_head2(shared_output)
        regression_output3 = self.regression_head3(shared_output)

        if self.config.activation_output_reg:
            regression_output1 = torch.sigmoid(regression_output1)
            regression_output2 = torch.sigmoid(regression_output2)
            regression_output3 = torch.sigmoid(regression_output3)

        return (
            classification_output,
            regression_output1,
            regression_output2,
            regression_output3,
        )


def _build_head(
    in_features: int,
    hidden_sizes: Sequence[int],
    dropout: float,
    out_features: int,
) -> nn.Sequential:
    layers = []
    current_features = in_features
    for hidden_size in hidden_sizes:
        layers.extend(
            [
                nn.Dropout(p=dropout),
                nn.Linear(current_features, hidden_size),
                nn.ReLU(),
            ]
        )
        current_features = hidden_size
    layers.append(nn.Linear(current_features, out_features))
    return nn.Sequential(*layers)


def get_default_config(input_size: int) -> ModelConfig:
    """Configuration tuned for the lightweight inference checkpoint shipped with the repo."""

    return ModelConfig(
        input_size=input_size,
        shared_hidden_sizes=(64, 128, 64),
        classification_hidden_sizes=(32, 16),
        regression_hidden_sizes=(32, 16),
        dropout_shared=0.0,
        dropout_classification=0.0,
        dropout_regression=0.0,
        use_cnn=True,
    )


def load_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load a checkpoint that stores its state dict inside the 'model_state_dict' key."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
