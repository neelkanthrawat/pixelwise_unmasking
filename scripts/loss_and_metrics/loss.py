import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
 
from math import sqrt, prod

from collections import namedtuple

from typing import Union, Callable



SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate", "regularizations"])
LossOutput = namedtuple("LossOutput", ["loss", "x", "x_rec", "nll", "z", "surrogate"])
Transform = Callable[[torch.Tensor], torch.Tensor]


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data.
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return:
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples > total_dim:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")
    v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples) #neel: can we extract like this? if yes, then it is correct–
    return q * sqrt(total_dim)


def nll_surrogate(x: torch.Tensor, encode: Transform, decode: Transform,
                  hutchinson_samples: int = 1, cond=None) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :param manifold: Manifold on which the latent space lies. If provided, the volume change is computed on the manifold.
    :return: Per-sample loss. Shape: (batch_size,)
    """
    data_dim = x.shape[-1]
    if cond is not None:
        cond_dim = cond.shape[-1]
    x.requires_grad_()

    if cond is not None:
        z = encode(x, cond)
    else:
        z = encode(x)
    z_red = z[..., :data_dim]

    metrics = {}

    surrogate = 0
    vs = sample_v(z_red, hutchinson_samples)

    for k in range(hutchinson_samples):
        v = vs[..., k]
        if cond is not None:
            v_red = v[...,:data_dim + cond_dim]
        else:
            v_red = v[..., :data_dim]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z_red, v_red)
            if cond is None:
                dual_x1 = decode(dual_z)
            if cond is not None:
                dual_x1 = decode(dual_z, cond)

            x1, v1 = unpack_dual(dual_x1)
            v1_red = v1[...,:data_dim]

        # $ v^T f'(x) $ via backward-mode AD
        v2, = grad(z_red, x, v, create_graph=True)
        v2_red = v2[...,:data_dim]

        # $ v^T f'(x) stop_grad(g'(z)) v $
        v2_red = v2_red.reshape((v2_red.shape[0], -1))
        v1_red = v1_red.reshape((v1_red.shape[0], -1))
        surrogate += sum_except_batch(v2_red * v1_red.detach()) / hutchinson_samples

    # Per-sample negative log-likelihood
    nll = sum_except_batch((z ** 2)) / 2 - surrogate

    return SurrogateOutput(z, x1, nll, surrogate, metrics)


def fff_loss(x: torch.Tensor,
             encode: Transform, decode: Transform,
             beta: Union[float, torch.Tensor],
             hutchinson_samples: int = 1,
             cond = None
             ) -> torch.Tensor:
    """
    Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :param fff_loss_dims: The first n dimensions of x_tilde that are to be trained with the FFF-loss.
    :param l: Weight of the regression loss for extra outputs (e.g. change-of-variables term).
    :return: Per-sample loss. Shape: (batch_size,)
    """
    data_dim = x.shape[-1]

    if cond is not None:
        surrogate = nll_surrogate(x, encode, decode, hutchinson_samples, cond=cond)
    else:
        surrogate = nll_surrogate(x, encode, decode, hutchinson_samples)

    x1_red = surrogate.x1[...,:data_dim]
    x = x.flatten(start_dim=1)
    mse = torch.sum((x - x1_red) ** 2, dim=tuple(range(1, len(x.shape))))
    loss = beta * mse + surrogate.nll
    return LossOutput(loss, x, x1_red, surrogate.nll, surrogate.z, surrogate.surrogate)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum over all dimensions except the first.
    :param x: Input tensor.
    :return: Sum over all dimensions except the first.
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)

def mmd_inverse_multi_quadratic(x, y, bandwidths=None):
    batch_size = x.size()[0]
    # compute the kernel matrices for each combination of x, y
    # (cleverly using broadcasting to do this efficiently)
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    # compute the sum of kernels at different bandwidths
    K, L, P = 0, 0, 0
    if bandwidths is None:
        bandwidths = [0.4, 0.8, 1.6]
    for sigma in bandwidths:
        s = 1.0 / sigma**2
        K += 1.0 / (1.0 + s * (rx.t() + rx - 2.0*xx))
        L += 1.0 / (1.0 + s * (ry.t() + ry - 2.0*yy))
        P += 1.0 / (1.0 + s * (rx.t() + ry - 2.0*xy))

    beta = 1./(batch_size*(batch_size-1)*len(bandwidths))
    gamma = 2./(batch_size**2 * len(bandwidths))
    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)

