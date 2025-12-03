import torch

def dot(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("m...,n...->mn", l1.conj(), l2)

def matmul(l: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.einsum("m...,mn->n...", l, m)

def square_norm(l: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...,...->", l.conj(), l)

def orthogonalize(l: torch.Tensor, shuffle: bool=True) -> torch.Tensor:
    """
        Orthogonalize vectors.
    """
    if shuffle:
        idx = torch.randperm(l.shape[0], device=l.device)
        for i in range(l.ndim-1):
            idx = idx.unsqueeze(-1)
        l = torch.gather(l, 0, idx.expand_as(l))
    for i in range(l.shape[0]):
        l[i] = l[i] - matmul(l[:i], dot(l[:i], l[i].unsqueeze(0))).squeeze(0)
        norm = square_norm(l[i]).real.sqrt()
        l[i] = l[i] / norm
    if shuffle:
        # restore original order
        rev = torch.argsort(idx.squeeze())
        for i in range(l.ndim-1):
            rev = rev.unsqueeze(-1)
        l = torch.gather(l, 0, rev.expand_as(l))
    return l

def projection(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    k = dot(x, z)
    k = (k + k.H) / 2
    # Project gradient onto tangent space
    m = z - matmul(x, k)
    return m

def cayley_update(
    current_point: torch.Tensor,
    steepest_grad: torch.Tensor,
    lr: float,
    upper_norm: float,
    n_iter: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
        Cayley transform SGD updated.
        See https://arxiv.org/abs/2002.01113 for original Cayley transform SGD.

        Args:
            current_point: Current point in the Lie algebra
            steepest_grad: Steepest gradient at current point
            lr: Learning rate
            upper_norm: Upper bound on the norm of the tangent vector
            n_iter: Number of iterations for the Cayley transform
            eps: Epsilon for numerical stability

        Returns:
            Updated point in the Lie algebra
    """
    if isinstance(lr, torch.Tensor):
        lr = lr.item()

    x = current_point
    z = steepest_grad

    # Project gradient onto tangent space
    m = projection(x, z)
    # Compute norm of the projection matrix
    norm = square_norm(m).real.abs().sqrt().item()
    # Select adaptive learning rate
    alpha = min(lr, 2 * upper_norm / (norm + eps))

    # Initialize y
    y = x - m.mul(alpha)
    # Iterative estimation of the Cayley transform
    for _ in range(n_iter):
        p = dot(x, y)
        y = x - matmul(m, p).add(m).mul(alpha/2)

    y = orthogonalize(y)

    return y, m
