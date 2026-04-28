* ===========================================================================
* Problem Set - Exercise 2
* GMM Estimation of Income/Consumption Covariance Structure
*
* Model:
*   Δy_it = u_it + v_it + (ψ_ε - 1)v_{t-1} - ψ_ε v_{t-2}         [MA(2)]
*   Δc_it = α u_it + αβ v_it + e_it + ψ_c e_{t-1}                 [MA(1)]
*
* Parameters: θ = (σ²_η, ψ_ε, σ²_ε, α, β, ψ_c, σ²_c)
*
* (c) Income-only: 12 equations (5 var + 4 ac1 + 3 ac2), 3 parameters
* (d) Full GMM:    33 equations across all 8 moment types, 7 parameters
* ===========================================================================

cd "E:\PhD\Econometria IIb"

* ---------------------------------------------------------------------------
* Preprocess data
* ---------------------------------------------------------------------------
import delimited ps1_data.csv, clear

gen id = _n

keep id income_1999 income_2001 income_2003 income_2005 income_2007 income_2009 ///
        consumption_1999 consumption_2001 consumption_2003 consumption_2005 ///
        consumption_2007 consumption_2009

* demean each period's level
foreach var of varlist income_* consumption_* {
    quietly summarize `var'
    replace `var' = `var' - r(mean)
}

* First differences in wide format (t=2,...,6 map to biennial periods)
gen dy2 = income_2001 - income_1999
gen dy3 = income_2003 - income_2001
gen dy4 = income_2005 - income_2003
gen dy5 = income_2007 - income_2005
gen dy6 = income_2009 - income_2007

gen dc2 = consumption_2001 - consumption_1999
gen dc3 = consumption_2003 - consumption_2001
gen dc4 = consumption_2005 - consumption_2003
gen dc5 = consumption_2007 - consumption_2005
gen dc6 = consumption_2009 - consumption_2007

* ---------------------------------------------------------------------------
* Moment product variables 
* ---------------------------------------------------------------------------

* --- Lag 0: Variances (t = 2,...,6) ---
gen yy0_t2 = dy2*dy2 
gen yy0_t3 = dy3*dy3 
gen yy0_t4 = dy4*dy4
gen yy0_t5 = dy5*dy5 
gen yy0_t6 = dy6*dy6

gen cc0_t2 = dc2*dc2 
gen cc0_t3 = dc3*dc3 
gen cc0_t4 = dc4*dc4
gen cc0_t5 = dc5*dc5 
gen cc0_t6 = dc6*dc6

gen yc0_t2 = dy2*dc2 
gen yc0_t3 = dy3*dc3 
gen yc0_t4 = dy4*dc4
gen yc0_t5 = dy5*dc5 
gen yc0_t6 = dy6*dc6

* --- Lag 1: First-order autocovariances (t = 3,...,6) ---
gen yy1_t3 = dy3*dy2 
gen yy1_t4 = dy4*dy3 
gen yy1_t5 = dy5*dy4 
gen yy1_t6 = dy6*dy5
gen cc1_t3 = dc3*dc2 
gen cc1_t4 = dc4*dc3 
gen cc1_t5 = dc5*dc4 
gen cc1_t6 = dc6*dc5
gen yc1_t3 = dy3*dc2 
gen yc1_t4 = dy4*dc3 
gen yc1_t5 = dy5*dc4 
gen yc1_t6 = dy6*dc5

* --- Lag 2: Second-order autocovariances (t = 4,...,6) ---
gen yy2_t4 = dy4*dy2 
gen yy2_t5 = dy5*dy3 
gen yy2_t6 = dy6*dy4
gen yc2_t4 = dy4*dc2 
gen yc2_t5 = dy5*dc3 
gen yc2_t6 = dy6*dc4

* ---------------------------------------------------------------------------
* Display sample moments
* ---------------------------------------------------------------------------
di "=== Sample Moments ==="

* Var(Δy): average across t=2,...,6
quietly {
    foreach t in 2 3 4 5 6 { 
        summarize yy0_t`t', meanonly 
        scalar syy_`t' = r(mean) 
    }
    scalar syy0 = (syy_2 + syy_3 + syy_4 + syy_5 + syy_6) / 5
    foreach t in 3 4 5 6 { 
        summarize yy1_t`t', meanonly 
        scalar syy1_`t' = r(mean) 
    }
    scalar syy1 = (syy1_3 + syy1_4 + syy1_5 + syy1_6) / 4
    foreach t in 4 5 6 { 
        summarize yy2_t`t', meanonly 
        scalar syy2_`t' = r(mean) 
    }
    scalar syy2 = (syy2_4 + syy2_5 + syy2_6) / 3
    foreach t in 2 3 4 5 6 { 
        summarize cc0_t`t', meanonly 
        scalar scc_`t' = r(mean) 
    }
    scalar scc0 = (scc_2 + scc_3 + scc_4 + scc_5 + scc_6) / 5
    foreach t in 3 4 5 6 { 
        summarize cc1_t`t', meanonly 
        scalar scc1_`t' = r(mean) 
    }
    scalar scc1 = (scc1_3 + scc1_4 + scc1_5 + scc1_6) / 4
    foreach t in 2 3 4 5 6 { 
        summarize yc0_t`t', meanonly 
        scalar syc_`t' = r(mean) 
    }
    scalar syc0 = (syc_2 + syc_3 + syc_4 + syc_5 + syc_6) / 5
    foreach t in 3 4 5 6 { 
        summarize yc1_t`t', meanonly 
        scalar syc1_`t' = r(mean) 
    }
    scalar syc1 = (syc1_3 + syc1_4 + syc1_5 + syc1_6) / 4
    foreach t in 4 5 6 { 
        summarize yc2_t`t', meanonly 
        scalar syc2_`t' = r(mean) 
    }
    scalar syc2 = (syc2_4 + syc2_5 + syc2_6) / 3
}

di "Var(Δy)            [avg over t=2..6] = " syy0
di "Var(Δc)            [avg over t=2..6] = " scc0
di "Cov(Δy,Δc)         [avg over t=2..6] = " syc0
di "Cov(Δy_t,Δy_{t-1}) [avg over t=3..6] = " syy1
di "Cov(Δc_t,Δc_{t-1}) [avg over t=3..6] = " scc1
di "Cov(Δy_t,Δc_{t-1}) [avg over t=3..6] = " syc1
di "Cov(Δy_t,Δy_{t-2}) [avg over t=4..6] = " syy2
di "Cov(Δy_t,Δc_{t-2}) [avg over t=4..6] = " syc2

* ---------------------------------------------------------------------------
* (c) Income-Only GMM
*
* Uses only income autocovariances to estimate (σ²_η, ψ_ε, σ²_ε).
* 12 equations (5 var + 4 lag-1 + 3 lag-2) for 3 parameters → 9 overid.
* ---------------------------------------------------------------------------
di ""
di "=== (c) INCOME-ONLY GMM: one-step (W=I) ==="

gmm                                                                           ///
    (yy0_2: yy0_t2 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_3: yy0_t3 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_4: yy0_t4 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_5: yy0_t5 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_6: yy0_t6 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy1_3: yy1_t3 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_4: yy1_t4 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_5: yy1_t5 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_6: yy1_t6 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy2_4: yy2_t4 - (-{psi_e}*{sig2e}))                                     ///
    (yy2_5: yy2_t5 - (-{psi_e}*{sig2e}))                                     ///
    (yy2_6: yy2_t6 - (-{psi_e}*{sig2e})),                                    ///
    instruments(yy0_2:) instruments(yy0_3:) instruments(yy0_4:)               ///
    instruments(yy0_5:) instruments(yy0_6:)                                   ///
    instruments(yy1_3:) instruments(yy1_4:)                                   ///
    instruments(yy1_5:) instruments(yy1_6:)                                   ///
    instruments(yy2_4:) instruments(yy2_5:) instruments(yy2_6:)               ///
    winitial(identity)                                                        ///
    from(sig2eta=0.5 psi_e=0.5 sig2e=0.5)                                    ///
    onestep vce(robust)
matrix b_inc = e(b)

di ""
di "=== (c) INCOME-ONLY GMM: two-step (optimal W) ==="

gmm                                                                           ///
    (yy0_2: yy0_t2 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_3: yy0_t3 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_4: yy0_t4 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_5: yy0_t5 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy0_6: yy0_t6 - ({sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}))         ///
    (yy1_3: yy1_t3 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_4: yy1_t4 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_5: yy1_t5 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy1_6: yy1_t6 - (-(1-{psi_e})^2*{sig2e}))                               ///
    (yy2_4: yy2_t4 - (-{psi_e}*{sig2e}))                                     ///
    (yy2_5: yy2_t5 - (-{psi_e}*{sig2e}))                                     ///
    (yy2_6: yy2_t6 - (-{psi_e}*{sig2e})),                                    ///
    instruments(yy0_2:) instruments(yy0_3:) instruments(yy0_4:)               ///
    instruments(yy0_5:) instruments(yy0_6:)                                   ///
    instruments(yy1_3:) instruments(yy1_4:)                                   ///
    instruments(yy1_5:) instruments(yy1_6:)                                   ///
    instruments(yy2_4:) instruments(yy2_5:) instruments(yy2_6:)               ///
    winitial(unadjusted, independent) twostep                                 ///
    from(b_inc) vce(robust)

scalar ratio_c = _b[/sig2eta] / _b[/sig2e]
di ""
di "σ²_η / σ²_ε (permanent-to-transitory ratio) = " ratio_c


* Heurisitic comparison
correlate dy2 dy3 dy4 dy5 dy6, covariance

di ""

* ---------------------------------------------------------------------------
* (d) Full GMM — Income + Consumption Jointly
*
* 33 equations for 7 parameters
*   Lag 0: Var(Δy)×5, Var(Δc)×5, Cov(Δy,Δc)×5           = 15 eqs
*   Lag 1: Cov(Δy,Δy_{-1})×4, Cov(Δc,Δc_{-1})×4,
*          Cov(Δy,Δc_{-1})×4                              = 12 eqs
*   Lag 2: Cov(Δy,Δy_{-2})×3, Cov(Δy,Δc_{-2})×3         =  6 eqs
* ---------------------------------------------------------------------------

* Shorthand macro for the 7 theoretical moment expressions
local var_y   {sig2eta} + 2*(1-{psi_e}+{psi_e}^2)*{sig2e}
local var_c   {alpha}^2*{sig2eta} + {alpha}^2*{beta}^2*{sig2e} + (1+{psi_c}^2)*{sig2c}
local cov_yc  {alpha}*{sig2eta} + {alpha}*{beta}*{sig2e}
local ac1_yy  -(1-{psi_e})^2*{sig2e}
local ac1_cc  {psi_c}*{sig2c}
local ac1_yc  {alpha}*{beta}*({psi_e}-1)*{sig2e}
local ac2_yy  -{psi_e}*{sig2e}
local ac2_yc  -{alpha}*{beta}*{psi_e}*{sig2e}

di ""
di "=== (d) FULL GMM: one-step (W=I) ==="

gmm                                                              ///
    (yy0_2: yy0_t2 - (`var_y'))                                  ///
    (yy0_3: yy0_t3 - (`var_y'))                                  ///
    (yy0_4: yy0_t4 - (`var_y'))                                  ///
    (yy0_5: yy0_t5 - (`var_y'))                                  ///
    (yy0_6: yy0_t6 - (`var_y'))                                  ///
    (cc0_2: cc0_t2 - (`var_c'))                                  ///
    (cc0_3: cc0_t3 - (`var_c'))                                  ///
    (cc0_4: cc0_t4 - (`var_c'))                                  ///
    (cc0_5: cc0_t5 - (`var_c'))                                  ///
    (cc0_6: cc0_t6 - (`var_c'))                                  ///
    (yc0_2: yc0_t2 - (`cov_yc'))                                 ///
    (yc0_3: yc0_t3 - (`cov_yc'))                                 ///
    (yc0_4: yc0_t4 - (`cov_yc'))                                 ///
    (yc0_5: yc0_t5 - (`cov_yc'))                                 ///
    (yc0_6: yc0_t6 - (`cov_yc'))                                 ///
    (yy1_3: yy1_t3 - (`ac1_yy'))                                 ///
    (yy1_4: yy1_t4 - (`ac1_yy'))                                 ///
    (yy1_5: yy1_t5 - (`ac1_yy'))                                 ///
    (yy1_6: yy1_t6 - (`ac1_yy'))                                 ///
    (cc1_3: cc1_t3 - (`ac1_cc'))                                 ///
    (cc1_4: cc1_t4 - (`ac1_cc'))                                 ///
    (cc1_5: cc1_t5 - (`ac1_cc'))                                 ///
    (cc1_6: cc1_t6 - (`ac1_cc'))                                 ///
    (yc1_3: yc1_t3 - (`ac1_yc'))                                 ///
    (yc1_4: yc1_t4 - (`ac1_yc'))                                 ///
    (yc1_5: yc1_t5 - (`ac1_yc'))                                 ///
    (yc1_6: yc1_t6 - (`ac1_yc'))                                 ///
    (yy2_4: yy2_t4 - (`ac2_yy'))                                 ///
    (yy2_5: yy2_t5 - (`ac2_yy'))                                 ///
    (yy2_6: yy2_t6 - (`ac2_yy'))                                 ///
    (yc2_4: yc2_t4 - (`ac2_yc'))                                 ///
    (yc2_5: yc2_t5 - (`ac2_yc'))                                 ///
    (yc2_6: yc2_t6 - (`ac2_yc')),                                ///
    instruments(yy0_2:) instruments(yy0_3:) instruments(yy0_4:)  ///
    instruments(yy0_5:) instruments(yy0_6:)                      ///
    instruments(cc0_2:) instruments(cc0_3:) instruments(cc0_4:)  ///
    instruments(cc0_5:) instruments(cc0_6:)                      ///
    instruments(yc0_2:) instruments(yc0_3:) instruments(yc0_4:)  ///
    instruments(yc0_5:) instruments(yc0_6:)                      ///
    instruments(yy1_3:) instruments(yy1_4:)                      ///
    instruments(yy1_5:) instruments(yy1_6:)                      ///
    instruments(cc1_3:) instruments(cc1_4:)                      ///
    instruments(cc1_5:) instruments(cc1_6:)                      ///
    instruments(yc1_3:) instruments(yc1_4:)                      ///
    instruments(yc1_5:) instruments(yc1_6:)                      ///
    instruments(yy2_4:) instruments(yy2_5:) instruments(yy2_6:)  ///
    instruments(yc2_4:) instruments(yc2_5:) instruments(yc2_6:)  ///
    winitial(identity)                                           ///
    from(sig2eta=0.1 psi_e=0.5 sig2e=0.1                        ///
         alpha=0.6 beta=0.5 psi_c=0.3 sig2c=0.1)               ///
    onestep vce(robust)


di ""
di "=== (d) FULL GMM: two-step (optimal W) ==="

gmm                                                              ///
    (yy0_2: yy0_t2 - (`var_y'))                                  ///
    (yy0_3: yy0_t3 - (`var_y'))                                  ///
    (yy0_4: yy0_t4 - (`var_y'))                                  ///
    (yy0_5: yy0_t5 - (`var_y'))                                  ///
    (yy0_6: yy0_t6 - (`var_y'))                                  ///
    (cc0_2: cc0_t2 - (`var_c'))                                  ///
    (cc0_3: cc0_t3 - (`var_c'))                                  ///
    (cc0_4: cc0_t4 - (`var_c'))                                  ///
    (cc0_5: cc0_t5 - (`var_c'))                                  ///
    (cc0_6: cc0_t6 - (`var_c'))                                  ///
    (yc0_2: yc0_t2 - (`cov_yc'))                                 ///
    (yc0_3: yc0_t3 - (`cov_yc'))                                 ///
    (yc0_4: yc0_t4 - (`cov_yc'))                                 ///
    (yc0_5: yc0_t5 - (`cov_yc'))                                 ///
    (yc0_6: yc0_t6 - (`cov_yc'))                                 ///
    (yy1_3: yy1_t3 - (`ac1_yy'))                                 ///
    (yy1_4: yy1_t4 - (`ac1_yy'))                                 ///
    (yy1_5: yy1_t5 - (`ac1_yy'))                                 ///
    (yy1_6: yy1_t6 - (`ac1_yy'))                                 ///
    (cc1_3: cc1_t3 - (`ac1_cc'))                                 ///
    (cc1_4: cc1_t4 - (`ac1_cc'))                                 ///
    (cc1_5: cc1_t5 - (`ac1_cc'))                                 ///
    (cc1_6: cc1_t6 - (`ac1_cc'))                                 ///
    (yc1_3: yc1_t3 - (`ac1_yc'))                                 ///
    (yc1_4: yc1_t4 - (`ac1_yc'))                                 ///
    (yc1_5: yc1_t5 - (`ac1_yc'))                                 ///
    (yc1_6: yc1_t6 - (`ac1_yc'))                                 ///
    (yy2_4: yy2_t4 - (`ac2_yy'))                                 ///
    (yy2_5: yy2_t5 - (`ac2_yy'))                                 ///
    (yy2_6: yy2_t6 - (`ac2_yy'))                                 ///
    (yc2_4: yc2_t4 - (`ac2_yc'))                                 ///
    (yc2_5: yc2_t5 - (`ac2_yc'))                                 ///
    (yc2_6: yc2_t6 - (`ac2_yc')),                                ///
    instruments(yy0_2:) instruments(yy0_3:) instruments(yy0_4:)  ///
    instruments(yy0_5:) instruments(yy0_6:)                      ///
    instruments(cc0_2:) instruments(cc0_3:) instruments(cc0_4:)  ///
    instruments(cc0_5:) instruments(cc0_6:)                      ///
    instruments(yc0_2:) instruments(yc0_3:) instruments(yc0_4:)  ///
    instruments(yc0_5:) instruments(yc0_6:)                      ///
    instruments(yy1_3:) instruments(yy1_4:)                      ///
    instruments(yy1_5:) instruments(yy1_6:)                      ///
    instruments(cc1_3:) instruments(cc1_4:)                      ///
    instruments(cc1_5:) instruments(cc1_6:)                      ///
    instruments(yc1_3:) instruments(yc1_4:)                      ///
    instruments(yc1_5:) instruments(yc1_6:)                      ///
    instruments(yy2_4:) instruments(yy2_5:) instruments(yy2_6:)  ///
    instruments(yc2_4:) instruments(yc2_5:) instruments(yc2_6:)  ///
    winitial(unadjusted, independent) twostep                    ///
    from(b_full) vce(robust)


