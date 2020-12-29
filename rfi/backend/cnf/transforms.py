"""
Conditional re-implementation of basic flow transformations from nflows package (https://github.com/bayesiains/nflows)
"""

from torch import nn
import torch
import torch.nn.init as init
from typing import Optional, Tuple, List
from torch import Tensor
from nflows.transforms import Transform, CompositeTransform, PointwiseAffineTransform


class ContextualInvertableRadialTransform(Transform):
    """
    Conditional Radial flow: z = f(y/x) = y + β(x) h(α(x), r)(y − γ(x))
    [Rezende and Mohamed 2015]

    α and β are reparametrized, so that transformation is invertable
    [Trippe and Turner 2018, https://arxiv.org/abs/1802.04908]

    Attributes:
        n_params (int): Number of parameters of transformation
    """

    n_params = 3

    def __init__(self, unconditional=False):
        """
        Args:
            unconditional: If True, parameters are initialized as torch.nn.Parameter. Else - None
        """
        super().__init__()

        if unconditional:
            self.gamma = nn.Parameter(torch.Tensor((1,)))
            self.alpha_hat = nn.Parameter(torch.Tensor((1,)))
            self.beta_hat = nn.Parameter(torch.Tensor((1,)))

            init.normal_(self.gamma, 0.0, 1.0)
            init.normal_(self.alpha_hat, 0.0, 1.0)
            init.normal_(self.beta_hat, 0.0, 1.0)
        else:
            self.gamma = None
            self.alpha_hat = None
            self.beta_hat = None

    def _params_from_context(self, context: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        if context is not None:
            assert context.shape[-1] == self.n_params
            if context.dim() == 2:
                alpha_hat = context[:, 0:1]
                beta_hat = context[:, 1:2]
                gamma = context[:, 2:3]
            else:
                alpha_hat = context[:, :, 0:1]
                beta_hat = context[:, :, 1:2]
                gamma = context[:, :, 2:3]
            return alpha_hat, beta_hat, gamma
        else:
            return self.alpha_hat, self.beta_hat, self.gamma

    @staticmethod
    def _alpha_beta_hat_to_alpha_beta(alpha_hat, beta_hat):
        alpha = torch.log(torch.exp(alpha_hat) + 1)
        beta = torch.exp(beta_hat) - 1
        return alpha, beta

    def forward(self, inputs, context=None):
        """
        Given context_vars (y) and global_context(x), returns transformed input (z = f(y/x)) and the abs log-determinant log|dz/dy|.
        """
        self.alpha_hat, self.beta_hat, self.gamma = self._params_from_context(context)
        alpha, beta = self._alpha_beta_hat_to_alpha_beta(self.alpha_hat, self.beta_hat)

        r = torch.norm(inputs - self.gamma, dim=1).unsqueeze(1)
        h = 1 / (alpha + r)
        z = inputs + alpha * beta * h * (inputs - self.gamma)
        log_det = torch.log(1 + beta * alpha**2 * h**2)
        return z, log_det.squeeze()

    def inverse(self, z, context=None):
        """
        Given context_vars (z) and global_context(x), returns inverse transformed input (y = f^-1(z/x)).
        """
        self.alpha_hat, self.beta_hat, self.gamma = self._params_from_context(context)
        alpha, beta = self._alpha_beta_hat_to_alpha_beta(self.alpha_hat, self.beta_hat)

        det1 = (alpha * beta + alpha - self.gamma - z) ** 2 - 4 * (-alpha * beta * self.gamma - alpha * z + self.gamma * z)
        inputs1 = 0.5 * (- alpha * beta - alpha + self.gamma + z + torch.sqrt(det1))
        inputs1[torch.isnan(inputs1)] = 0.0
        # inputs12 = 0.5 * (- alpha * beta - alpha + gamma + z - torch.sqrt(det1))

        det2 = (alpha * beta + alpha + self.gamma + z) ** 2 - 4 * (alpha * beta * self.gamma + alpha * z + self.gamma * z)
        # inputs21 = 0.5 * (alpha * beta + alpha + gamma + z + torch.sqrt(det2))
        inputs2 = 0.5 * (alpha * beta + alpha + self.gamma + z - torch.sqrt(det2))
        inputs2[torch.isnan(inputs2)] = 0.0

        mask = (z >= self.gamma).float()
        inputs_merged = inputs1 * mask + inputs2 * (1.0 - mask)

        log_det = 0.0  # Too complex and not needed for sampling
        return inputs_merged, log_det


class ContextualAffineTransform(PointwiseAffineTransform):
    """
    Affine flow: z = f(y/x) = y * scale(x) + shift(x)
    scale is exponentiated, so no worries about devision on zero

    Attributes:
        n_params (int): Number of parameters of transformation
    """

    n_params = 2

    def _params_from_context(self, context: Tensor = None) -> Tuple[Tensor, Tensor]:
        if context is not None:
            assert context.shape[-1] == self.n_params
            if context.dim() == 2:
                scale = context[:, 0:1]
                shift = context[:, 1:2]
            else:
                scale = context[:, :, 0:1]
                shift = context[:, :, 1:2]
            return scale, shift
        else:
            return torch.log(self._scale), self._shift

    def forward(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Given context_vars (y) and global_context(x), returns transformed input (z = f(y/x))
        and the abs log-determinant log|dz/dy|.
        """
        log_scale, self._shift = self._params_from_context(context)
        self._scale = torch.exp(log_scale)
        # RuntimeError here means shift/scale not broadcastable to input.
        outputs = inputs * self._scale + self._shift
        logabsdet = self._log_scale.squeeze()
        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Given context_vars (z) and global_context(x), returns inverse transformed input (y = f^-1(z/x))
        and the abs log-determinant log|dy/dz|.
        """
        log_scale, self._shift = self._params_from_context(context)
        self._scale = torch.exp(log_scale)
        outputs = (inputs - self._shift) / self._scale
        logabsdet = - self._log_scale.squeeze()
        return outputs, logabsdet


class ContextualCompositeTransform(CompositeTransform):
    """
    Composition of different transformations, embedded global_context is tiled for each transformation.

    Attributes:
        n_params (int): Number of parameters of transformation
    """

    def __init__(self, transforms: List[Transform]):
        """
        Args:
            transforms: List of contextual transformations
        """
        super().__init__(transforms)

        # Calculating number of parameters
        self.n_params = sum([transform.n_params for transform in transforms])

    @staticmethod
    def _cascade(inputs, funcs, context, forward=True):
        outputs = inputs
        total_logabsdet = torch.zeros(context.shape[0] if context.dim() == 2 else context.shape[:2]).to(inputs.device)

        assert context.shape[-1] == sum([func.n_params for func in funcs])

        if forward:
            ind_acc = 0
            for func in funcs:
                # Splitting global_context along transformations
                context_tile = context[:, ind_acc:ind_acc + func.n_params] if context.dim() == 2 else \
                    context[:, :, ind_acc:ind_acc + func.n_params]
                ind_acc += func.n_params
                outputs, logabsdet = func(outputs, context_tile)
                total_logabsdet += logabsdet
        else:
            ind_acc = context.shape[-1]
            for func in funcs[::-1]:
                # Splitting global_context along transformations
                context_tile = context[:, ind_acc - func.n_params:ind_acc] if context.dim() == 2 else \
                    context[:, ind_acc - func.n_params:ind_acc]
                ind_acc -= func.n_params
                outputs, logabsdet = func.inverse(outputs, context_tile)
                total_logabsdet += logabsdet

        return outputs, total_logabsdet

    def inverse(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context, forward=False)
