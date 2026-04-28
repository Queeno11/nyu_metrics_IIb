# Econometrics II (Panel Data) — Graded Problem Set, April 2026
**NYU Department of Economics · Instructor: Martín Almuzara · TA: Rafael Lincoln**

This problem set studies efficiency comparisons across panel data estimators (Ex. 1), income and consumption dynamics via covariance structure estimation (Ex. 2), and testing for correlated random coefficients (Ex. 3). The empirical exercises (2 and 3) use PSID household panel data (`ps1_data.csv`, N=792, T=6 biennial waves 1999–2009).

---

## Index

### `Problem_Set_-_Excercise_1.py` — Python (NumPy / Numba)
Implements the Monte Carlo simulation for Exercise 1(c), comparing the First-Difference (FD), Within-Group (WG), and unfeasible optimal IV estimators across values of ρ. Replications are parallelised with Numba's `prange`.

| Output | Description |
|---|---|
| `fig1_baseline.png` | Baseline sampling variances vs ρ (Π=(1,1)′, σ²=1, α=0.4) |
| `fig2_alpha.png` | Sensitivity to endogeneity strength α ∈ {0.1, 0.4, 0.8} |
| `fig3_sigma2.png` | Sensitivity to error variance σ² ∈ {0.5, 1.0, 2.0} |
| `fig4_Pi.png` | Sensitivity to first-stage coefficient Π ∈ {(1,1)′, (1,0.1)′, (2,2)′} |

---

### `Problem_Set_-_Excercise_2.do` — Stata
Estimates the Hall-Mishkin income/consumption model via GMM on the PSID data. Constructs moment product variables for lags 0–2 of income and consumption growth, then runs one-step (W=I) and two-step (optimal W) GMM.

| Output | Description |
|---|---|
| Table (c) — income-only GMM | Estimates and SEs for (σ²η, ψε, σ²ε); printed to Stata output |
| Scalar `ratio_c` | Permanent-to-transitory ratio σ²η/σ²ε; printed to Stata output |
| Covariance matrix display | Sample covariance matrix of income growth (`correlate dy*, covariance`) for heuristic fit comparison |
| Table (d) — full GMM | Estimates and SEs for all θ = (σ²η, ψε, σ²ε, α, β, ψc, σ²c); printed to Stata output |

---

### `Problem_Set_-_Excercise_3.do` — Stata
Implements the correlated random coefficients test of Exercise 3(c), using income as yit, time index t as xit, and age squared as zit.

| Output | Description |
|---|---|
| Table — pooled FD-OLS | Baseline estimates of (β̄, γ) ignoring slope heterogeneity; printed to Stata output |
| Table — augmented FD-OLS | Estimates including auxiliary regressors r_x, r_z; printed to Stata output |
| Wald test (`test r_z`) | F-test for H0: λ1z = 0 (no correlated random slopes); printed to Stata output |
| Scalars `mean_slope_test`, `mean_slope_class` | Mean slope E[βi] from the parametric (test-supported) and non-parametric (FE) approaches; printed to Stata output |
| Figure 1 (combined graph) | Kernel density of individual slopes βi: parametric linear projection (blue) vs. non-parametric fixed effects (red, dashed), stacked vertically |

---

## Dependencies

| Script | Requirements |
|---|---|
| `Excercise_1.py` | Python ≥ 3.10, `numpy`, `numba`, `matplotlib`, `tqdm` |
| `Excercise_2.do` | Stata ≥ 16, `ps1_data.csv` in working directory |
| `Excercise_3.do` | Stata ≥ 16, `ps1_data.csv` in working directory |

The working directory in both `.do` files is currently hardcoded to `E:\PhD\Econometria IIb`. Update the `cd` command at the top of each file before running.
