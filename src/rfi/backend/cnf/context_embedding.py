from torch.nn import Module
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from torch import Tensor
from rfi.backend.cnf.transforms import ContextualCompositeTransform


class ContextEmbedding(Module):
    """
    Conditioning global_context embedding neural network. Used together with conditional normalizing flow.
    Implemented as MLP with weights normalisation.
    """

    def __init__(self,
                 transform: ContextualCompositeTransform,
                 input_units: int,
                 hidden_units: tuple = None,
                 hidden_nonlinearity=torch.tanh):
        """
        Args:
            transform: ContextualCompositeTransform of conditional normalizing flow
            input_units: Context dimensionality
            hidden_units: Tuple of hidden units dimensions
            hidden_nonlinearity: Hidden non-linear transformation
        """
        super().__init__()
        self.layers = nn.ModuleList()

        if hidden_units is None or len(hidden_units) == 0:
            self.layers.append(weight_norm(nn.Linear(input_units, transform.n_params)))
        else:
            self.layers.append(weight_norm(nn.Linear(input_units, hidden_units[0])))
            for i in range(0, len(hidden_units) - 1):
                self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.layers.append(weight_norm(nn.Linear(hidden_units[-1], transform.n_params)))
        self.hidden_nonlinearity = hidden_nonlinearity

    def reset_parameters(self) -> None:
        """
        Reset parameters of network
        """
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method
        """
        for layer_ind in range(len(self.layers) - 1):
            x = self.layers[layer_ind](x)
            x = self.hidden_nonlinearity(x)
        # Last layer
        x = self.layers[-1](x)
        return x
