* ===========================================================================
* Problem Set - Exercise 3
* First-*fference Estimator under Heterogeneous Slopes
*
* Model:
*   y_it = α_i + β_i*x_it + γ*z_it + v_it
*   E[v_it | β_i, w_i] = 0  (strict exogeneity)
*
*   After FD (t = 2,...,T):
*   Δy_it = β_i*Δx_it + γ*Δz_it + Δv_it
*
* (c. i)  Pooled FD-OLS ignoring slope heterogeneity
* (c.ii)  Variable Addition Test: H0: λ_1 = 0_{k_w×1} (no correlated slopes)
*         Augmented regression: Δy_it = λ_0 Δx_it + γ Δz_it + λ_1' r_it + u_it
*         Wald statistic ~ χ²(k_w) under H0
* ===========================================================================

cd "E:\PhD\Econometria IIb"

* ---------------------------------------------------------------------------
* 1. Load data and reshape to long format
* ---------------------------------------------------------------------------
import delimited ps1_data.csv, clear

gen id = _n

keep id income_1999 income_2001 income_2003 income_2005 income_2007 income_2009 ///
        age_1999    age_2001    age_2003    age_2005    age_2007    age_2009

reshape long income_ age_, i(id) j(year) string
rename income_ income
rename age_    age

* Numeric time index t = 1,...,6
gen t = .
replace t = 1 if year == "1999"
replace t = 2 if year == "2001"
replace t = 3 if year == "2003"
replace t = 4 if year == "2005"
replace t = 5 if year == "2007"
replace t = 6 if year == "2009"
drop year

sort id t
xtset id t

* ---------------------------------------------------------------------------
* 2. Construct model variables
* ---------------------------------------------------------------------------
* y_it = income,  x_it = t,  z_it = age^2
gen age2 = age^2

* First differences
gen dy  = D.income
gen dx  = D.t           // = 1 always (after FD, t=2,...,6)
gen dz  = D.age2        

* ===========================================================================
* PART (c.i): Pooled FD-OLS
* ===========================================================================
regress dy dx dz, vce(cluster id) nocons

* ===========================================================================
* PART (c.ii): Variable Addition Test for Correlated Random Slopes
* ===========================================================================

* --- Compute g(w_i) ---

egen g_x=sum(dx*dx), by(id)
egen g_z=sum(dx*dz), by(id)

* --- Auxiliary regressors ---
gen r_x = g_x * dx         
gen r_z = g_z * dx       


* --- Step 3: Augmented POLS with cluster-robust SE ---

regress dy dx dz r_x r_z, vce(cluster id) nocons

test r_z

* Store the coefficients
scalar lambda_1z = _b[r_z]
scalar lambda_1x = _b[r_x]
scalar intercept = _b[r_x] * 5

* Get the sample mean of the g(w_i) variable for age squared
* (We use tag so we only count each individual's g_age_sq once)
egen tag_id = tag(id)
sum g_z if tag_id == 1
scalar mean_g_z = r(mean)
sum g_x if tag_id == 1
scalar mean_g_x = r(mean)

* Calculate the mean slope
scalar mean_slope_test = intercept + (lambda_1z * mean_g_z)
display "Mean Slope (Test-Supported): " mean_slope_test


* "==================================================================="
* "                          β̂/λ̂_0       γ̂"
* "  Standard  FD-POLS :  " bhat "    " ghat
* "  Augmented FD-POLS :  " lambda0 "    " gamma_a
* ""
* "If H0 is rejected, bias from correlated slopes (eq. 37) is non-zero:"
* "  plim(β̂) ≠ β̄  and  plim(γ̂) ≠ γ"
* "The inconsistency in β bleeds into γ through the inverse Hessian matrix."


* ===========================================================================
* PART (c.iii): Individual Slope Coefficients (Arellano-Bonhomme like)
* ===========================================================================

xtreg dy dz, fe 

scalar mean_slope_class = _b[_cons]
display "Mean Slope (Class Procedure): " mean_slope_class


* ===========================================================================
* PLOTTING THE DISTRIBUTIONS OF INDIVIDUAL SLOPES
* ===========================================================================

* 1. Calculate the Parametric Slopes (Test-Supported Linear Projection)
* ---------------------------------------------------------------------------
cap drop beta_param
* Use the saved scalars so we don't have to rely on silent regressions
gen beta_param = intercept + (lambda_1z * g_z)

* 2. Calculate the Non-Parametric Slopes (Class Procedure / Fixed Effects)
* ---------------------------------------------------------------------------
cap drop u_i beta_nonparam
predict u_i, u
* The slope for person i is the overall average (_cons) + their specific deviation
gen beta_nonparam = mean_slope_class + u_i

* 3. Plot the Distributions — two panels, stacked vertically, x-axis aligned
* ---------------------------------------------------------------------------

* Compute a common x-axis range spanning both distributions
quietly summarize beta_param    if t == 2
scalar xlo = r(min)
scalar xhi = r(max)
quietly summarize beta_nonparam if t == 2
scalar xlo = min(xlo, r(min))
scalar xhi = max(xhi, r(max))
* Add 10% margin on each side
scalar xlo = xlo - 0.1 * (xhi - xlo)
scalar xhi = xhi + 0.1 * (xhi - xlo)

* Panel 1: Parametric slopes
twoway (kdensity beta_param if t == 2, lcolor(blue) lwidth(medthick)), ///
    title("Parametric (Linear Projection)", size(medsmall))                           ///
    xtitle("") ytitle("Density")                                                      ///
    xscale(range(`=scalar(xlo)' `=scalar(xhi)'))                                      ///
    xlabel(`=scalar(xlo)'(0.25)`=scalar(xhi)', format(%4.2f))                         ///
    graphregion(color(white)) plotregion(margin(zero))                                ///
    name(g_param, replace) nodraw

* Panel 2: Non-parametric slopes
twoway (kdensity beta_nonparam if t == 2, lcolor(red) lpattern(dash) lwidth(medthick)), ///
    title("Non-Parametric (Fixed Effects)", size(medsmall))                             ///
    xtitle("Estimated Individual Slope ({&beta}{sub:i})") ytitle("Density")             ///
    xscale(range(`=scalar(xlo)' `=scalar(xhi)'))                                        ///
    xlabel(`=scalar(xlo)'(0.25)`=scalar(xhi)', format(%4.2f))                           ///
    graphregion(color(white)) plotregion(margin(zero))                                  ///
    name(g_nonparam, replace) nodraw

* Combine: one column → stacked vertically, x-axes automatically aligned
graph combine g_param g_nonparam,                                              ///
    cols(1)                                                                    ///
    title("Distribution of Heterogeneous Slopes ({bf:{&beta}}{sub:i})",       ///
          size(medium))                                                        ///
    graphregion(color(white)) imargin(0 0 2 2)