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
    Conditional Radial flow: z = f(y/context) = y + α(context) β(context) (y - γ(context)) / (α(context) + ||y − γ(context)||)
    [Rezende and Mohamed 2015]

    α and β are reparametrized, so that transformation is invertable
    [Trippe and Turner 2018, https://arxiv.org/abs/1802.04908]

    Attributes:
        n_params (int): Number of parameters of transformation
    """

    def __init__(self, inputs_size: int, conditional=True):
        super().__init__()
        self.inputs_size = inputs_size

        if conditional:
            self.alpha_hat, self.beta_hat, self.gamma = None, None, None  # Will be initialized from context
        else:
            gamma = torch.nn.Parameter(torch.empty((inputs_size,)))
            alpha_hat = torch.nn.Parameter(torch.empty((1,)))
            beta_hat = torch.nn.Parameter(torch.empty((1,)))

            init.normal_(gamma, 0.0, 1.0)
            init.normal_(alpha_hat, 0.0, 1.0)
            init.normal_(beta_hat, 0.0, 1.0)

            self.register_parameter("gamma", gamma)
            self.register_parameter("alpha_hat", alpha_hat)
            self.register_parameter("beta_hat", beta_hat)

    @property
    def n_params(self):
        return 2 + self.inputs_size

    def _params_from_context(self, context: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        if context is not None:
            assert context.shape[-1] == self.n_params
            if context.dim() == 2:
                alpha_hat = context[:, 0:1]
                beta_hat = context[:, 1:2]
                gamma = context[:, 2:2 + self.inputs_size]
            else:
                alpha_hat = context[:, :, 0:1]
                beta_hat = context[:, :, 1:2]
                gamma = context[:, :, 2:2 + self.inputs_size]
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
        Given context_vars (y) and global_context(context), returns transformed input (z = f(y/context))
        and the abs log-determinant log|dz/dy|.
        """
        self.alpha_hat, self.beta_hat, self.gamma = self._params_from_context(context)
        alpha, beta = self._alpha_beta_hat_to_alpha_beta(self.alpha_hat, self.beta_hat)

        r_norm = torch.linalg.norm(inputs - self.gamma, dim=1).unsqueeze(1)
        h = 1 / (alpha + r_norm)
        z = inputs + alpha * beta * h * (inputs - self.gamma)
        # log_det = torch.log(1 + beta * alpha**2 * h**2)
        log_det = (self.inputs_size - 1) * torch.log(1 + alpha * beta * h) + torch.log(1 + beta * alpha**2 * h**2)
        return z, log_det.squeeze()

    def inverse(self, z, context=None):
        """
        Given context_vars (z) and global_context(context), returns inverse transformed input (y = f^-1(z/context)).
        """
        self.alpha_hat, self.beta_hat, self.gamma = self._params_from_context(context)
        alpha, beta = self._alpha_beta_hat_to_alpha_beta(self.alpha_hat, self.beta_hat)

        # det1 = (alpha * beta + alpha - self.gamma - z) ** 2 - 4 * (-alpha * beta * self.gamma - alpha * z + self.gamma * z)
        # inputs1 = 0.5 * (- alpha * beta - alpha + self.gamma + z + torch.sqrt(det1))
        # inputs1[torch.isnan(inputs1)] = 0.0
        # # inputs12 = 0.5 * (- alpha * beta - alpha + gamma + z - torch.sqrt(det1))
        #
        # det2 = (alpha * beta + alpha + self.gamma + z) ** 2 - 4 * (alpha * beta * self.gamma + alpha * z + self.gamma * z)
        # # inputs21 = 0.5 * (alpha * beta + alpha + gamma + z + torch.sqrt(det2))
        # inputs2 = 0.5 * (alpha * beta + alpha + self.gamma + z - torch.sqrt(det2))
        # inputs2[torch.isnan(inputs2)] = 0.0
        #
        # mask = (z >= self.gamma).float()
        # inputs_merged = inputs1 * mask + inputs2 * (1.0 - mask)

        z_min_gamma = z - self.gamma
        z_min_gamma_norm = torch.linalg.norm(z_min_gamma, dim=1).unsqueeze(1)
        det = (alpha + alpha * beta - z_min_gamma_norm) ** 2 + 4 * alpha * z_min_gamma_norm
        r_norm = 0.5 * (-(alpha + alpha * beta - z_min_gamma_norm) + torch.sqrt(det))
        inputs = (alpha + r_norm) / (alpha + r_norm + alpha * beta) * z_min_gamma + self.gamma
        log_det = 0.0  # Too complex and not needed for sampling
        return inputs, log_det


class ContextualPointwiseAffineTransform(Transform):
    """
    Affine flow: z = f(y/context) = y * scale(context) + shift(context)
    scale is exponentiated, so no worries about devision on zero
    """

    def __init__(self, inputs_size: int, conditional=True):
        super().__init__()
        self.inputs_size = inputs_size

        if conditional:
            self.shift, self.scale = None, None
        else:
            shift = torch.nn.Parameter(torch.zeros((1, inputs_size)))
            scale = torch.nn.Parameter(torch.ones((1, inputs_size)))

            self.register_parameter("shift", shift)
            self.register_parameter("scale", scale)

    @property
    def n_params(self):
        return self.inputs_size + self.inputs_size

    def _params_from_context(self, context: Tensor = None) -> Tuple[Tensor, Tensor]:
        if context is not None:
            assert context.shape[-1] == self.n_params
            if context.dim() == 2:
                log_scale = context[:, 0:self.inputs_size]
                shift = context[:, self.inputs_size:(self.inputs_size + self.inputs_size)]
            else:
                log_scale = context[:, :, 0:self.inputs_size]
                shift = context[:, :, self.inputs_size:(self.inputs_size + self.inputs_size)]
            return torch.exp(log_scale), shift
        else:
            return self.scale, self.shift

    @property
    def _log_scale(self) -> Tensor:
        return torch.log(self.scale)

    def forward(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Given context_vars (y) and global_context(context), returns transformed input (z = f(y/context))
        and the abs log-determinant log|dz/dy|.
        """
        self.scale, self.shift = self._params_from_context(context)
        # RuntimeError here means shift/scale not broadcastable to input.
        outputs = inputs * self.scale + self.shift
        logabsdet = self._log_scale.sum(1) + torch.zeros((len(inputs), ))
        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Given context_vars (z) and global_context(context), returns inverse transformed input (y = f^-1(z/context))
        and the abs log-determinant log|dy/dz|.
        """
        self.scale, self.shift = self._params_from_context(context)
        outputs = (inputs - self.shift) / self.scale
        logabsdet = - self._log_scale.sum(1) + torch.zeros((len(inputs), ))
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
