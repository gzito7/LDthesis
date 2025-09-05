
# -----------------------------------------------------------------------------
# Large Deviations rate functions for single-input Gaussian NNs (M=1):
# - Covariance rate IK via recursion
# - Output rate IF(y) = min_k [ IK(k) + ||y||^2/(2k) ]
#
# Supports Linear activation (sigma(x)=sqrt(a)*x) and ReLU.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Callable, Literal, Tuple, Optional

# Optional plotting (only used in main())
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -----------------------------------------------------------------------------
# Φ(λ | K) for common activations (Section 4.3)
# -----------------------------------------------------------------------------

def phi_linear(lmbda: float, K: float, a: float) -> float:
    """
    Φ(λ|K) = -0.5 * log(1 - 2 * λ * a * K), valid for λ < 1/(2 a K) if K>0; Φ=0 if K=0.
    """
    if K == 0.0:
        return 0.0
    cap = 1.0 / (2.0 * a * K)
    if lmbda >= cap:
        return np.inf
    term = 1.0 - 2.0 * lmbda * a * K
    if term <= 0.0:
        return np.inf
    return -0.5 * np.log(term)


def phi_relu(lmbda: float, K: float) -> float:
    """
    Φ(λ|K) = log( 0.5 * ( 1 + 1/sqrt(1 - 2 λ K) ) ), valid for λ < 1/(2K) if K>0; Φ=0 if K=0.
    """
    if K == 0.0:
        return 0.0
    cap = 1.0 / (2.0 * K)
    if lmbda >= cap:
        return np.inf
    term = 1.0 - 2.0 * lmbda * K
    if term <= 0.0:
        return np.inf
    return np.log(0.5 * (1.0 + 1.0 / np.sqrt(term)))


def get_phi(activation: Literal["linear", "relu"], a: float = 0.5) -> Callable[[float, float], float]:
    if activation == "linear":
        return lambda lmbda, K: phi_linear(lmbda, K, a)
    elif activation == "relu":
        return phi_relu
    else:
        raise ValueError("activation must be 'linear' or 'relu'")


# -----------------------------------------------------------------------------
# Fenchel–Legendre transform Φ*(k | K)
# - Linear: closed form (Section 5.1.1, eq. (5.5))
# - ReLU: 1-D numerical sup over λ ∈ [0, 1/(2K))
# -----------------------------------------------------------------------------

def phi_star_linear(k: float, K: float, a: float) -> float:
    """
    Closed-form Φ*(k|K) for Linear:
        Φ*(k|K) = 0.5 * [ k/(aK) - log(k/(aK)) - 1 ], for k>0, K>0.
        Boundary handling: Φ*(0|0)=0; Φ*(k<=0|K>0)=+inf.
    """
    if K == 0.0:
        return 0.0 if k == 0.0 else np.inf
    if k <= 0.0:
        return np.inf
    ratio = k / (a * K)
    return 0.5 * (ratio - np.log(ratio) - 1.0)


def phi_star_numerical(
    k: float,
    K: float,
    phi: Callable[[float, float], float],
    lmbda_lo: float,
    lmbda_hi: float,
    n_grid: int = 2000,
) -> float:
    """
    Numerical FL transform: Φ*(k|K) = sup_λ [ λ k - Φ(λ|K) ] over λ ∈ [lmbda_lo, lmbda_hi).
    (Simple grid-based robust solver. Increase n_grid for more accuracy.)
    """
    if K == 0.0:
        return 0.0 if k == 0.0 else np.inf
    if k < 0.0:
        return np.inf
    lambdas = np.linspace(lmbda_lo, lmbda_hi, n_grid, endpoint=False)
    vals = lambdas * k - np.array([phi(lmbda, K) for lmbda in lambdas])
    return float(np.max(vals))


def get_phi_star(
    activation: Literal["linear", "relu"],
    a: float = 0.5,
    relu_n_grid: int = 2000,
) -> Callable[[float, float], float]:
    if activation == "linear":
        return lambda k, K: phi_star_linear(k, K, a)
    elif activation == "relu":
        phi = get_phi("relu")
        def phi_star(k: float, K: float) -> float:
            if K == 0.0:
                return 0.0 if k == 0.0 else np.inf
            cap = 1.0 / (2.0 * K)
            return phi_star_numerical(k, K, phi, 0.0, cap, n_grid=relu_n_grid)
        return phi_star
    else:
        raise ValueError("activation must be 'linear' or 'relu'")


# -----------------------------------------------------------------------------
# IK recursion (single-input case, Section 5.1):
#   IK_2(k) = Φ*(k | K1)
#   IK_{l+1}(k_next) = min_{k>=0} [ Φ*(k_next | k) + IK_l(k) ]
# Represent IK_l on a uniform k-grid; use brute-force search (robust) over that grid.
# -----------------------------------------------------------------------------

def compute_IK(
    L: int,
    K1: float,
    activation: Literal["linear", "relu"] = "linear",
    a: float = 0.5,
    k_grid_max: float = 6.0,
    k_grid_size: int = 1000,
    relu_n_grid: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (k_values, IK_{L+1}(k)) on a uniform k-grid in [eps, k_grid_max].
    Args:
      L           : number of hidden layers (we output IK at layer L+1)
      K1          : variance at layer 1
      activation  : 'linear' or 'relu'
      a           : scale for Linear activation σ(x)=sqrt(a)*x
      k_grid_max  : max k on the grid
      k_grid_size : number of grid points
      relu_n_grid : inner λ-grid for ReLU φ* numerical sup
    """
    eps = 1e-8
    grid = np.linspace(eps, k_grid_max, k_grid_size)
    phi_star = get_phi_star(activation, a=a, relu_n_grid=relu_n_grid)

    # Base layer: IK_2(k) = Φ*(k | K1)
    IK_prev = np.array([phi_star(k, K1) for k in grid])

    # Build up to IK_{L+1}
    for _ in range(2, L + 1):
        IK_next = np.empty_like(grid)
        # For each target k_next, minimize over all k in the grid
        for i, k_next in enumerate(grid):
            vals = np.array([phi_star(k_next, k) + IK_prev[j] for j, k in enumerate(grid)])
            IK_next[i] = float(np.min(vals))
        IK_prev = IK_next

    return grid, IK_prev


# -----------------------------------------------------------------------------
# IF (output rate) for M=1:
#   IF(y) = min_{k>=0} [ IK_{L+1}(k) + ||y||^2 / (2k) ]
# -----------------------------------------------------------------------------

def compute_IF_from_IK(
    y_norm: float,
    k_grid: np.ndarray,
    IK_vals: np.ndarray,
) -> float:
    denom = 2.0 * k_grid
    total = IK_vals + (y_norm ** 2) / denom
    return float(np.min(total))


def make_IF_regularizer(
    L: int,
    K1: float,
    activation: Literal["linear", "relu"] = "linear",
    a: float = 0.5,
    k_grid_max: float = 6.0,
    k_grid_size: int = 1000,
    relu_n_grid: int = 2000,
) -> Callable[[np.ndarray], float]:
    """
    Build a callable IF_reg(y_vec) that returns I_F(||y||) for the given configuration.
    Use it like: total_loss = data_loss(y) + gamma * IF_reg(y).
    """
    k_grid, IK_vals = compute_IK(
        L=L, K1=K1, activation=activation, a=a,
        k_grid_max=k_grid_max, k_grid_size=k_grid_size, relu_n_grid=relu_n_grid
    )

    def IF_y(y_vec: np.ndarray) -> float:
        y_norm = float(np.linalg.norm(y_vec))
        return compute_IF_from_IK(y_norm, k_grid, IK_vals)

    return IF_y


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def main():
    # Configuration
    L = 2            # hidden depth (compute IK at layer L+1)
    K1 = 1.0         # variance at first layer
    a = 0.5          # linear activation scaling
    k_max = 4.0
    k_size = 300
    relu_n = 800     # ReLU λ-grid

    # Compute IK for Linear and ReLU
    k_lin, IK_lin = compute_IK(L, K1, activation="linear", a=a, k_grid_max=k_max, k_grid_size=k_size)
    k_relu, IK_relu = compute_IK(L, K1, activation="relu", a=a, k_grid_max=k_max, k_grid_size=k_size, relu_n_grid=relu_n)

    # Compute IF vs ||y||
    y_norms = np.linspace(0.0, 3.0, 31)
    IF_lin = [compute_IF_from_IK(y, k_lin, IK_lin) for y in y_norms]
    IF_relu = [compute_IF_from_IK(y, k_relu, IK_relu) for y in y_norms]

    # Print a small table
    print("||y||    IF_linear    IF_ReLU")
    for y, il, ir in zip(y_norms[::5], IF_lin[::5], IF_relu[::5]):
        print(f"{y:5.2f}   {il:9.6f}   {ir:9.6f}")

    # Plots
    if _HAS_PLT:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(k_lin, IK_lin, label="IK (Linear)")
        plt.plot(k_relu, IK_relu, label="IK (ReLU)")
        plt.xlabel("k (variance at output layer)")
        plt.ylabel("IK_{L+1}(k)")
        plt.title("Covariance Rate Function IK")
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(y_norms, IF_lin, label="IF (Linear)")
        plt.plot(y_norms, IF_relu, label="IF (ReLU)")
        plt.xlabel("||y||")
        plt.ylabel("IF(y)")
        plt.title("Output Rate Function IF vs ||y||")
        plt.legend()
        plt.tight_layout()

        plt.show()
    else:
        print("[matplotlib not available] Skipping plots.")

if __name__ == "__main__":
    main()
