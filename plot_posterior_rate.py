# plot_posterior_rate_both.py
import numpy as np
import matplotlib.pyplot as plt

def phi_star_linear(k: float, K: float, a: float) -> float:
    if K == 0.0:
        return 0.0 if k == 0.0 else np.inf
    if k <= 0.0:
        return np.inf
    ratio = k / (a * K)
    return 0.5 * (ratio - np.log(ratio) - 1.0)

def compute_IK_linear(L:int, K1:float, a:float=0.5, k_grid_max:float=4.0, k_grid_size:int=200):
    eps = 1e-8
    grid = np.linspace(eps, k_grid_max, k_grid_size)
    IK_prev = np.array([phi_star_linear(k, K1, a) for k in grid])
    for _ in range(2, L+1):
        IK_next = np.empty_like(grid)
        for i, k_next in enumerate(grid):
            vals = np.array([phi_star_linear(k_next, k, a) + IK_prev[j] for j,k in enumerate(grid)])
            IK_next[i] = np.min(vals)
        IK_prev = IK_next
    return grid, IK_prev

def phi_relu(lmbda: float, K: float) -> float:
    if K == 0.0:
        return 0.0
    cap = 1.0/(2.0*K)
    if lmbda >= cap:
        return np.inf
    term = 1.0 - 2.0*lmbda*K
    if term <= 0.0:
        return np.inf
    return np.log(0.5*(1.0 + 1.0/np.sqrt(term)))

def phi_star_relu(k: float, K: float, n_grid:int=220) -> float:
    if K == 0.0:
        return 0.0 if k == 0.0 else np.inf
    if k < 0.0:
        return np.inf
    cap = 1.0/(2.0*K)
    lambdas = np.linspace(0.0, cap, n_grid, endpoint=False)
    term = 1.0 - 2.0*lambdas*K
    phi_vals = np.log(0.5*(1.0 + 1.0/np.sqrt(term)))
    vals = lambdas*k - phi_vals
    return float(np.max(vals))

def compute_IK_relu(L:int, K1:float, k_grid_max:float=4.0, k_grid_size:int=140, n_grid:int=220):
    eps = 1e-8
    grid = np.linspace(eps, k_grid_max, k_grid_size)
    IK_prev = np.array([phi_star_relu(k, K1, n_grid=n_grid) for k in grid])
    for _ in range(2, L+1):
        IK_next = np.empty_like(grid)
        for i, k_next in enumerate(grid):
            phi_stars = np.array([phi_star_relu(k_next, k, n_grid=n_grid) for k in grid])
            vals = phi_stars + IK_prev
            IK_next[i] = np.min(vals)
        IK_prev = IK_next
    return grid, IK_prev

def compute_IF_from_IK(y_norm: float, k_grid: np.ndarray, IK_vals: np.ndarray) -> float:
    total = IK_vals + (y_norm**2)/(2.0*k_grid)
    return float(np.min(total))

def main():
    L = 2
    K1 = 1.0
    a = 0.5
    t = np.array([1.0, 0.0])
    def ell(y_vec: np.ndarray) -> float:
        return 0.5 * np.sum((y_vec - t)**2)

    k_lin, IK_lin = compute_IK_linear(L=L, K1=K1, a=a, k_grid_max=4.0, k_grid_size=220)
    k_relu, IK_relu = compute_IK_relu(L=L, K1=K1, k_grid_max=3.5, k_grid_size=130, n_grid=200)

    ys = np.linspace(-2, 3, 220)

    ell_vals = []
    IF_lin_vals, R_lin_vals = [], []
    IF_relu_vals, R_relu_vals = [], []

    for y0 in ys:
        y = np.array([y0, 0.0])
        e = ell(y)
        IF_lin = compute_IF_from_IK(np.linalg.norm(y), k_lin, IK_lin)
        IF_relu = compute_IF_from_IK(np.linalg.norm(y), k_relu, IK_relu)

        ell_vals.append(e)
        IF_lin_vals.append(IF_lin)
        R_lin_vals.append(e + IF_lin)
        IF_relu_vals.append(IF_relu)
        R_relu_vals.append(e + IF_relu)

    plt.figure(figsize=(9,6))
    plt.plot(ys, ell_vals, label="Empirical loss ell(y)", linewidth=2)
    plt.plot(ys, IF_lin_vals, label="IF(y) Linear", linestyle="--")
    plt.plot(ys, R_lin_vals, label="Posterior rate ell+IF (Linear)", linestyle="--")
    plt.plot(ys, IF_relu_vals, label="IF(y) ReLU", linestyle="-.")
    plt.plot(ys, R_relu_vals, label="Posterior rate ell+IF (ReLU)", linestyle="-.")
    plt.xlabel("y[0] (with y[1]=0)")
    plt.ylabel("Value")
    plt.title("Posterior Rate Function: Linear vs ReLU (L=2, single input)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
