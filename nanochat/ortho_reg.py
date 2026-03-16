"""
Orthogonal regularization for weight matrices.

Implements Gram-matrix based orthogonal regularization with two modes:
1. Coupled: adds λ * Σ_ℓ ||Gram(W_ℓ) - s·I||_F^2 / d as an auxiliary loss.
   Gradients flow through the optimizer's momentum/variance estimates.
2. Decoupled (AdamO-style): applies orthogonal gradient step separately from
   the optimizer, keeping isometry gradients out of Adam/Muon moments.

The Frobenius penalty on the Gram deviation is:
  R_iso(W) = 0.5 * ||Gram(W) - s·I||_F^2 / gram_dim

where Gram(W) is computed on the smaller dimension of W for efficiency,
s is the activation scale (e.g., 2.0 for ReLU to compensate half-zeroed signal),
and gram_dim normalizes the penalty to be independent of layer width.

Two optional scaling corrections (each independently configurable):
- activation_scale: scale s for the identity target W^T W ≈ sI.
    1.0 = standard orthogonality.
    2.0 = ReLU-scaled (compensates E[relu(x)^2] = σ^2/2, same for relu^2).
- rect_scale: for tall matrices (out_dim > in_dim), multiply s by (out_dim/in_dim).
    True (default): adjusts for signal norm propagation when orthogonalizing
    the shorter dimension of a rectangular matrix.
    False: always use s directly, regardless of aspect ratio.

Reference: "Preserving Plasticity in Continual Learning via Dynamical Isometry"
  - Section 4.2: Gram-matrix regularization
  - Section 4.4: AdamO (decoupled orthogonal optimization)

Ported from the JAX implementation in continual-isometry/isometry_lab/losses.py
and adamo.py, adapted for PyTorch and the nanochat architecture.
"""

import torch
from torch import Tensor


def _gram_deviation(weight: Tensor, activation_scale: float = 1.0,
                    rect_scale: bool = True):
    """
    Compute Gram matrix deviation from scaled identity.

    For PyTorch nn.Linear, weight has shape (out_dim, in_dim).
    We compute the Gram on the smaller dimension for efficiency and because
    only min(out_dim, in_dim) singular values can be independently controlled.

    Convention note: PyTorch stores W as (out_dim, in_dim) and computes y = x @ W^T.
    The Jacobian dy/dx = W, so for isometry we want W's singular values near sqrt(s).
    This is equivalent to Gram(W) ≈ s·I on the smaller dimension.

    For the JAX code (where weight is (in_dim, out_dim)), the same penalty is
    expressed as W^T @ W ≈ s·I. The math is identical, just transposed.

    Args:
        weight: 2D weight tensor of shape (out_dim, in_dim)
        activation_scale: scale factor s for the identity target.
            1.0 = standard orthogonality (W^T W ≈ I).
            2.0 = ReLU-scaled (compensates E[relu(x)^2] = σ^2/2).
        rect_scale: if True and out_dim > in_dim, multiply s by (out_dim/in_dim)
            to account for signal norm propagation when orthogonalizing the
            shorter dimension of a tall rectangular matrix. Set to False to
            always use activation_scale directly.

    Returns:
        deviation: Gram - scale * I, shape (gram_dim, gram_dim)
        gram_dim: dimension of the Gram matrix
    """
    out_dim, in_dim = weight.shape
    if out_dim <= in_dim:
        # "Wide or square": rows of W should be orthogonal
        gram = weight @ weight.T  # (out_dim, out_dim)
        scale = activation_scale
        gram_dim = out_dim
    else:
        # "Tall": columns of W should be orthogonal
        gram = weight.T @ weight  # (in_dim, in_dim)
        if rect_scale:
            # The (out_dim / in_dim) factor accounts for signal norm propagation
            # when orthogonalizing the shorter dimension of a rectangular matrix.
            scale = activation_scale * (out_dim / in_dim)
        else:
            scale = activation_scale
        gram_dim = in_dim
    identity = torch.eye(gram_dim, dtype=weight.dtype, device=weight.device)
    return gram - scale * identity, gram_dim


def gram_frobenius_penalty(weight: Tensor, activation_scale: float = 1.0,
                            rect_scale: bool = True) -> Tensor:
    """
    Compute 0.5 * ||Gram(W) - s·I||_F^2 / gram_dim for a single weight matrix.
    Normalized by gram_dim so the penalty magnitude is independent of layer width.
    """
    deviation, gram_dim = _gram_deviation(weight, activation_scale, rect_scale)
    return 0.5 * (deviation ** 2).sum() / gram_dim


@torch.no_grad()
def gram_frobenius_grad(weight: Tensor, activation_scale: float = 1.0,
                         rect_scale: bool = True) -> Tensor:
    """
    Analytical gradient of the normalized Frobenius penalty w.r.t. weight.

    For penalty R = 0.5 * ||Gram(W) - s·I||_F^2 / d:

    If out_dim <= in_dim (Gram = W @ W^T):
        ∂R/∂W = 2 * (W W^T - sI) @ W / gram_dim

    If out_dim > in_dim (Gram = W^T @ W):
        ∂R/∂W = 2 * W @ (W^T W - sI) / gram_dim

    Note: for zero-initialized weights (e.g., c_proj at init), the gradient
    is zero since W=0 makes the matmul vanish, so this naturally doesn't
    disturb zero-init training dynamics.
    """
    out_dim, in_dim = weight.shape
    deviation, gram_dim = _gram_deviation(weight, activation_scale, rect_scale)
    if out_dim <= in_dim:
        return (2.0 / gram_dim) * (deviation @ weight)
    else:
        return (2.0 / gram_dim) * (weight @ deviation)


def get_ortho_reg_params(model) -> list[Tensor]:
    """
    Collect all 2D weight parameters from transformer blocks for orthogonal
    regularization. Includes all attention/MLP weights plus small matrices
    (e.g., VE gates).

    Excludes embeddings (wte), unembedding (lm_head), and 1D scalars
    (resid_lambdas, x0_lambdas) since these are not matrix operators.
    """
    params = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'transformer.h.' in name:
            params.append(param)
    return params


def compute_ortho_reg_loss(params: list[Tensor], lambda_reg: float,
                           activation_scale: float = 1.0,
                           rect_scale: bool = True) -> Tensor:
    """
    Compute total orthogonal regularization loss (coupled / auxiliary loss mode).

    Returns λ * Σ_ℓ R_iso(W_ℓ).

    Use this in the training loop by adding the result to the task loss
    and calling .backward(). Note: the paper (Section 4.4) warns that
    mixing isometry gradients with task gradients in adaptive optimizer
    moments can be undesirable; consider using apply_decoupled_ortho_reg
    (AdamO-style) instead.
    """
    device = params[0].device
    total = torch.zeros(1, device=device)
    for p in params:
        total = total + gram_frobenius_penalty(p, activation_scale, rect_scale)
    return lambda_reg * total


@torch.no_grad()
def apply_decoupled_ortho_reg(params: list[Tensor], lr: float, lambda_reg: float,
                               activation_scale: float = 1.0,
                               rect_scale: bool = True) -> None:
    """
    Apply decoupled orthogonal regularization step (AdamO-style).

    Updates: W -= η_iso * λ * ∇R_iso(W)

    This is applied after the optimizer step, keeping the orthogonal
    gradients out of the optimizer's momentum/variance estimates.
    Analogous to how AdamW decouples weight decay from Adam's moments.

    Args:
        params: list of weight tensors to regularize
        lr: isometry learning rate η_iso (typically = base lr * schedule * lr_scale)
        lambda_reg: regularization strength λ
        activation_scale: scale factor for the identity target
        rect_scale: whether to apply (out_dim/in_dim) correction for tall matrices
    """
    for p in params:
        grad = gram_frobenius_grad(p, activation_scale, rect_scale)
        p.sub_(lr * lambda_reg * grad)
