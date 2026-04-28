"""
Exercise 1(c) — Monte Carlo: FD vs WG vs Optimal IV
Econometrics II (Panel Data) · April 2026

Estimators implemented from the solution's derived formulas:
  (3)  β̂_FD  — 2SLS with Δz_it as instrument in the FD equation
  (4)  β̂_WG  — 2SLS with z̃_it = z_it − z̄_i in the WG equation
  (14) β̂_IV  — Unfeasible optimal IV using K=D, known Σ(z_i), and linear first stage
"""

import time
import numpy as np
from tqdm import tqdm
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ══════════════════════════════════════════════════════════════════════════════
#  Matrix helpers
# ══════════════════════════════════════════════════════════════════════════════

@njit
def make_D(T: int) -> np.ndarray:
    """(T-1)×T first-difference operator  D  such that Dv = Δv."""
    D = np.zeros((T - 1, T))
    for k in range(T - 1):
        D[k, k]     = -1.0
        D[k, k + 1] =  1.0
    return D

@njit
def make_Sigma(rho: float, sigma2: float, T: int) -> np.ndarray:
    """
    T×T conditional covariance matrix  Σ  for the AR(1) error process

        v_{i1} ~ (0, σ²),   v_{it} = ρ v_{i,t-1} + ε_{it},   Var(ε_{it}|z_i) = σ².

    Derived in eq. (9)–(10) of the solution:

      Var(v_{it}) = σ² Σ_{j=0}^{t-1} ρ^{2j}
                  = σ²(1 − ρ^{2t})/(1 − ρ²)     for |ρ| ≠ 1
                  = σ² · t                         for  ρ  = ±1

      Cov(v_{it}, v_{is}) = ρ^{|t−s|} Var(v_{min(t,s)})
    """
    Sigma = np.zeros((T, T))
    for t in range(1, T + 1):
        for s in range(1, T + 1):
            m = min(t, s)
            if abs(abs(rho) - 1.0) < 1e-10:          # |ρ| = 1  (includes random walk)
                var_m = sigma2 * m
            else:
                var_m = sigma2 * (1.0 - rho ** (2 * m)) / (1.0 - rho**2)
            Sigma[t - 1, s - 1] = rho ** abs(t - s) * var_m
    return Sigma


# ══════════════════════════════════════════════════════════════════════════════
#  Single Monte Carlo replication
# ══════════════════════════════════════════════════════════════════════════════

@njit
def one_replication(
    N: int,
    T: int,
    beta: float,
    rho: float,
    sigma2: float,
    Pi: np.ndarray,       # (r,)
    alpha: float,
    D: np.ndarray,        # (T-1, T) — precomputed
    Omega_inv: np.ndarray,# (T-1, T-1) — (DΣD')^{-1}, precomputed for given ρ
) -> tuple[float, float, float]:
    
    r = len(Pi)

    # 1. Initialize empty arrays (avoids Numba tuple-shape issues)
    zi = np.empty((N, T, r))
    eps = np.empty((N, T))
    u = np.empty((N, T))
    eta = np.empty(N)
    
    # 2. Fill arrays safely
    for n in range(N):
        eta[n] = np.random.randn()
        for t in range(T):
            eps[n, t] = np.random.randn() * np.sqrt(sigma2)
            u[n, t] = np.random.randn() * np.sqrt(sigma2)
            for k in range(r):
                zi[n, t, k] = np.random.randn()

    # 3. Construct Data Generating Process (DGP)
    vi = np.empty((N, T))
    xi = np.empty((N, T))
    yi = np.empty((N, T))
    
    for n in range(N):
        vi[n, 0] = eps[n, 0]
        for t in range(1, T):
            vi[n, t] = rho * vi[n, t - 1] + eps[n, t]
            
        # zi[n] is (T, r), Pi is (r,). Result is (T,)
        xi[n] = zi[n] @ Pi + alpha * eps[n] + np.sqrt(1.0 - alpha**2) * u[n]
        yi[n] = beta * xi[n] + eta[n] + vi[n]

    # 4. First-Difference (FD) Transformation & Estimator
    Dyi = np.empty((N, T - 1))
    Dxi = np.empty((N, T - 1))
    Dzi = np.empty((N, T - 1, r))
    
    Sxz_FD = np.zeros(r)
    Szz_FD = np.zeros((r, r))
    Szy_FD = np.zeros(r)
    
    for n in range(N):
        Dyi[n] = D @ yi[n]
        Dxi[n] = D @ xi[n]
        Dzi[n] = D @ zi[n]
        
        # Accumulate sums (T-1, r).T @ (T-1,) -> (r,)
        Sxz_FD += Dzi[n].T @ Dxi[n]
        Szz_FD += Dzi[n].T @ Dzi[n]
        Szy_FD += Dzi[n].T @ Dyi[n]
        
    Szz_FD_inv = np.linalg.inv(Szz_FD)
    A_FD = Sxz_FD @ Szz_FD_inv
    beta_FD = (A_FD @ Szy_FD) / (A_FD @ Sxz_FD)

    # 5. Within-Group (WG) Transformation & Estimator
    ty = np.empty((N, T))
    tx = np.empty((N, T))
    tz = np.empty((N, T, r))
    
    Sxz_WG = np.zeros(r)
    Szz_WG = np.zeros((r, r))
    Szy_WG = np.zeros(r)
    
    for n in range(N):
        # Bulletproof 1D de-meaning
        ty[n] = yi[n] - np.mean(yi[n])
        tx[n] = xi[n] - np.mean(xi[n])
        for k in range(r):
            tz[n, :, k] = zi[n, :, k] - np.mean(zi[n, :, k])
            
        Sxz_WG += tz[n].T @ tx[n]
        Szz_WG += tz[n].T @ tz[n]
        Szy_WG += tz[n].T @ ty[n]

    Szz_WG_inv = np.linalg.inv(Szz_WG)
    A_WG = Sxz_WG @ Szz_WG_inv
    beta_WG = (A_WG @ Szy_WG) / (A_WG @ Sxz_WG)

    # 6. Optimal IV Estimator
    num_IV = 0.0
    den_IV = 0.0
    
    for n in range(N):
        opt_n = Dzi[n] @ Pi          # (T-1,) optimal instrument
        Wopt_n = opt_n @ Omega_inv   # (T-1,) weighted instrument
        
        num_IV += Wopt_n @ Dyi[n]
        den_IV += Wopt_n @ Dxi[n]

    beta_IV = num_IV / den_IV

    return beta_FD, beta_WG, beta_IV

# ══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo driver
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def run_monte_carlo(nMC, N, T, beta, rho, sigma2, Pi, alpha, D, Omega_inv):
    # 1. Pre-allocate arrays to hold the results of each replication
    ests_FD = np.empty(nMC)
    ests_WG = np.empty(nMC)
    ests_IV = np.empty(nMC)
    
    # 2. Use prange instead of range to distribute the workload across CPU threads
    for m in prange(nMC):
        fd, wg, iv = one_replication(N, T, beta, rho, sigma2, Pi, alpha, D, Omega_inv)
        
        ests_FD[m] = fd
        ests_WG[m] = wg
        ests_IV[m] = iv
        
    return ests_FD, ests_WG, ests_IV

def sweep_rho(
    rho_values: np.ndarray,
    nMC: int,
    N: int,
    T: int,
    beta: float,
    sigma2: float,
    Pi: np.ndarray,
    alpha: float,
    base_seed: int = 0, # Kept for signature compatibility, but not strictly needed for Numba prange
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run run_monte_carlo for every ρ in rho_values.
    Returns three arrays: (var_FD, var_WG, var_IV).
    """
    # 1. Precompute D (it only depends on T, not rho)
    D = make_D(T)
    
    results = []
    for k, rho in enumerate(rho_values):
        t0 = time.perf_counter()
        
        # 2. Recompute Sigma and Omega_inv for the CURRENT rho
        Sigma = make_Sigma(rho, sigma2, T)
        Omega = D @ Sigma @ D.T
        Omega_inv = np.linalg.inv(Omega)
        
        # 3. Pass the newly computed D and Omega_inv into the parallel Monte Carlo
        v  = run_monte_carlo(nMC, N, T, beta, rho, sigma2, Pi, alpha, D, Omega_inv)
        
        elapsed = time.perf_counter() - t0
        tag = f"[{label}]  " if label else ""
        
        # Calculate N * Var(beta)
        var_FD = N * np.var(v[0])
        var_WG = N * np.var(v[1])
        var_IV = N * np.var(v[2])
        
        print(f"  {tag}ρ={rho:+.2f}  "
              f"N·Var_FD={var_FD:.4f}  N·Var_WG={var_WG:.4f}  N·Var_IV={var_IV:.4f}  "
              f"({elapsed:.1f}s)")
              
        results.append((var_FD, var_WG, var_IV))

    arr = np.array(results)          # (len(rho_values), 3)
    return arr[:, 0], arr[:, 1], arr[:, 2]

# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

_COLOURS = {"FD": "#2874A6", "WG": "#C0392B", "IV": "#1E8449"}
_MARKERS  = {"FD": "o",      "WG": "s",       "IV": "^"}

def _add_lines(ax, rho_vals, vFD, vWG, vIV, suffix="", ls="-", alpha=1.0):
    kw = dict(linestyle=ls, linewidth=1.8, markersize=5, alpha=alpha)
    ax.plot(rho_vals, vFD, marker=_MARKERS["FD"], color=_COLOURS["FD"],
            label=rf"$\hat\beta_{{FD}}${suffix}", **kw)
    ax.plot(rho_vals, vWG, marker=_MARKERS["WG"], color=_COLOURS["WG"],
            label=rf"$\hat\beta_{{WG}}${suffix}", **kw)
    ax.plot(rho_vals, vIV, marker=_MARKERS["IV"], color=_COLOURS["IV"],
            label=rf"$\hat\beta_{{IV}}$ (opt.){suffix}", **kw)


def _decorate(ax, title=""):
    ax.set_xlabel(r"$\rho$", fontsize=12)
    ax.set_ylabel(r"$N \times \widehat{\mathrm{Var}}(\hat\beta)$", fontsize=11)
    ax.set_title(title, fontsize=11, pad=6)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Baseline parameters (problem set specification) ────────────────────
    nMC    = 10_000
    N      = 2_000
    T      = 7
    beta   = 1.0
    sigma2 = 1.0
    Pi     = np.array([1.0, 1.0])
    alpha  = 0.4

    # ρ ∈ {−0.2, 0, 0.2, …, 1.0}  (problem set specification)
    rho_values = np.round(np.arange(-0.2, 1.01, 0.2), 10)

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 1 — Baseline (required figure)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("Figure 1: Baseline")
    print("═" * 60)

    vFD1, vWG1, vIV1 = sweep_rho(rho_values, nMC, N, T, beta, sigma2, Pi, alpha,
                                  base_seed=0)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    _add_lines(ax1, rho_values, vFD1, vWG1, vIV1)
    _decorate(ax1, title=(
        r"Sampling Variances — Baseline"
        "\n"
        r"($\Pi=(1,1)^\prime$,  $\sigma^2=1$,  $\alpha=0.4$,  $N=2000$,  $T=7$)"
    ))
    # Annotate efficiency equivalences (FD optimal at ρ=1, WG at ρ=0)
    for x, lbl in [(0.0, "WG optimal\n(ρ=0)"), (1.0, "FD optimal\n(ρ=1)")]:
        ax1.axvline(x, color="grey", linewidth=0.8, linestyle=":")
        ax1.text(x + 0.02, ax1.get_ylim()[1] * 0.97, lbl,
                 fontsize=7, va="top", color="grey")
    fig1.tight_layout()
    fig1.savefig("fig1_baseline.png", dpi=150, bbox_inches="tight")
    print("  → fig1_baseline.png")

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 2 — Varying α  (endogeneity strength)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("Figure 2: Varying α  (endogeneity strength)")
    print("═" * 60)

    alpha_vals = [0.1, 0.4, 0.8]
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, a in zip(axes2, alpha_vals):
        print(f"\n  α = {a}")
        vFD, vWG, vIV = sweep_rho(rho_values, nMC, N, T, beta, sigma2, Pi, a,
                                   base_seed=100 + int(a * 100),
                                   label=f"α={a}")
        _add_lines(ax, rho_values, vFD, vWG, vIV)
        _decorate(ax, title=rf"$\alpha = {a}$")

    fig2.suptitle("Effect of Endogeneity Strength α on Sampling Variances", fontsize=12)
    fig2.tight_layout()
    fig2.savefig("fig2_alpha.png", dpi=150, bbox_inches="tight")
    print("  → fig2_alpha.png")

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 3 — Varying σ²
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("Figure 3: Varying σ²")
    print("═" * 60)

    sigma2_vals = [0.5, 1.0, 2.0]
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, s2 in zip(axes3, sigma2_vals):
        print(f"\n  σ² = {s2}")
        vFD, vWG, vIV = sweep_rho(rho_values, nMC, N, T, beta, s2, Pi, alpha,
                                   base_seed=200 + int(s2 * 10),
                                   label=f"σ²={s2}")
        _add_lines(ax, rho_values, vFD, vWG, vIV)
        _decorate(ax, title=rf"$\sigma^2 = {s2}$")

    fig3.suptitle(r"Effect of Error Variance $\sigma^2$ on Sampling Variances",
                  fontsize=12)
    fig3.tight_layout()
    fig3.savefig("fig3_sigma2.png", dpi=150, bbox_inches="tight")
    print("  → fig3_sigma2.png")

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 4 — Varying Π  (first-stage strength)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("Figure 4: Varying Π  (instrument strength)")
    print("═" * 60)

    Pi_configs = [
        (np.array([1.0, 1.0]),  r"$\Pi=(1,1)^\prime$  [baseline]"),
        (np.array([1.0, 0.1]),  r"$\Pi=(1,0.1)^\prime$  [weak $z_2$]"),
        (np.array([2.0, 2.0]),  r"$\Pi=(2,2)^\prime$  [stronger]"),
    ]
    fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (Pi_c, lbl) in zip(axes4, Pi_configs):
        print(f"\n  Π = {Pi_c}")
        vFD, vWG, vIV = sweep_rho(rho_values, nMC, N, T, beta, sigma2, Pi_c, alpha,
                                   base_seed=300 + int(Pi_c[1] * 10),
                                   label=f"Π={Pi_c}")
                                   
        _add_lines(ax, rho_values, vFD, vWG, vIV)
        _decorate(ax, title=lbl)

    fig4.suptitle(r"Effect of First-Stage Coefficient $\Pi$ on Sampling Variances",
                  fontsize=12)
    fig4.tight_layout()
    fig4.savefig("fig4_Pi.png", dpi=150, bbox_inches="tight")
    print("  → fig4_Pi.png")

    # ══════════════════════════════════════════════════════════════════════
    plt.show()
    print("\nDone.  All figures saved.")