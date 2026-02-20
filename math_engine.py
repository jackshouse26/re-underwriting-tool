import pandas as pd
import numpy_financial as npf
import numpy as np


def run_model_engine(assumptions, gpr_total, p_price_override=None, e_cap_override=None):
    # Unpack assumptions (use overrides for sensitivity matrix)
    p_price = p_price_override if p_price_override is not None else assumptions['purchase_price']
    e_cap = e_cap_override if e_cap_override is not None else assumptions['exit_cap_rate']

    hold_period_yrs = assumptions['hold_period_yrs']
    capex_budget = assumptions['capex_budget']
    const_ltv = assumptions['const_ltv']
    const_months = assumptions['const_months']
    income_growth = assumptions['income_growth']
    year_1_opex = assumptions['year_1_opex']
    expense_growth = assumptions['expense_growth']
    has_abatement = assumptions['has_abatement']
    abatement_years = assumptions['abatement_years']
    abatement_savings = assumptions['abatement_savings']
    const_rate = assumptions['const_rate']
    refi_month = assumptions['refi_month']
    perm_ltv = assumptions['perm_ltv']
    perm_rate = assumptions['perm_rate']
    closing_costs_pct = assumptions.get('closing_costs_pct', 0.015)
    loan_orig_fee_pct = assumptions.get('loan_orig_fee_pct', 0.01)
    exit_costs_pct    = assumptions.get('exit_costs_pct', 0.02)

    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    cols = ['CapEx', 'Unlevered_CF', 'Const_Draw', 'Const_Balance', 'Const_Interest', 'Perm_Balance', 'Perm_Interest', 'Levered_CF', 'Debt_Service']
    for c in cols: df[c] = 0.0

    total_cost = p_price + capex_budget
    loan_max = total_cost * const_ltv
    acq_costs = p_price * closing_costs_pct
    orig_fee  = loan_max * loan_orig_fee_pct
    initial_equity = total_cost - loan_max + acq_costs + orig_fee

    df.loc[0, 'Unlevered_CF'] = -(total_cost + acq_costs)
    df.loc[0, 'Levered_CF'] = -initial_equity

    current_bal = p_price * const_ltv
    m_draw = (loan_max - (p_price * const_ltv)) / const_months if const_months > 0 else 0

    last_noi = 0
    perm_payment = 0.0  # fixed monthly P&I, set at refi; reused in subsequent months

    for m in range(1, total_months + 1):
        noi = ((gpr_total * ((1 + income_growth)**(m/12))) - (year_1_opex * ((1 + expense_growth)**(m/12)))) / 12
        if has_abatement and m <= (abatement_years * 12): noi += (abatement_savings / 12)

        last_noi = noi
        df.loc[m, 'Unlevered_CF'] = noi

        if m <= refi_month:
            ds = current_bal * (const_rate / 12)
            draw = m_draw if m <= const_months else 0
            current_bal += draw
            df.loc[m, 'Debt_Service'] = ds
            df.loc[m, 'Levered_CF'] = noi + draw - ds
        elif m == refi_month + 1:
            p_val = (noi * 12) / e_cap
            p_amt = p_val * perm_ltv
            # 30-year amortizing schedule; npf.pmt returns a negative number for a payment
            perm_payment = float(-npf.pmt(perm_rate / 12, 360, p_amt))
            interest = p_amt * perm_rate / 12
            new_bal = p_amt - (perm_payment - interest)
            df.loc[m, 'Levered_CF'] = noi + (p_amt - current_bal) - perm_payment
            df.loc[m, 'Perm_Balance'] = new_bal
            df.loc[m, 'Debt_Service'] = perm_payment
            current_bal = 0
        else:
            p_bal = df.loc[m-1, 'Perm_Balance']
            interest = p_bal * perm_rate / 12
            principal = perm_payment - interest
            new_bal = max(0.0, p_bal - principal)
            df.loc[m, 'Perm_Balance'] = new_bal
            df.loc[m, 'Debt_Service'] = perm_payment
            df.loc[m, 'Levered_CF'] = noi - perm_payment

    exit_noi = ((gpr_total * ((1 + income_growth)**((total_months+1)/12))) - (year_1_opex * ((1 + expense_growth)**((total_months+1)/12))))
    exit_val = (exit_noi / e_cap) * (1 - exit_costs_pct)

    df.loc[total_months, 'Unlevered_CF'] += exit_val
    df.loc[total_months, 'Levered_CF'] += (exit_val - df.loc[total_months, 'Perm_Balance'])

    try:
        irr = (1 + npf.irr(df['Levered_CF']))**12 - 1
        if pd.isna(irr) or irr < -0.99: irr = -1.0
    except: irr = -1.0

    try:
        unlev_irr = (1 + npf.irr(df['Unlevered_CF']))**12 - 1
        if pd.isna(unlev_irr) or unlev_irr < -0.99: unlev_irr = -1.0
    except: unlev_irr = -1.0

    dscr = (last_noi * 12) / (df['Debt_Service'].max() * 12) if df['Debt_Service'].max() > 0 else 0
    return irr, dscr, df, initial_equity, total_cost, unlev_irr


def compute_gp_lp_waterfall(levered_cf, initial_equity, hold_period_yrs,
                              gp_commit_pct=0.10, pref_rate=0.08,
                              promote_rate=0.20, hurdle_irr=0.15):
    """
    American-style 4-tier GP/LP waterfall:
      T1 — Return of capital (pro-rata LP/GP by commitment)
      T2 — LP preferred return (compound cumulative at pref_rate on LP capital)
      T3 — GP catch-up (100% to GP until GP share of T2+T3 == promote_rate)
      T4 — Above-hurdle promote: (1-promote) LP / promote GP if IRR > hurdle; else pro-rata
    Operating cash flows (pre-exit months) split pro-rata; exit proceeds run through all tiers.
    """
    cf = np.array(levered_cf, dtype=float)
    n = len(cf)
    lp_pct = 1.0 - gp_commit_pct
    gp_invested = initial_equity * gp_commit_pct
    lp_invested  = initial_equity * lp_pct

    lp_cf = np.zeros(n)
    gp_cf = np.zeros(n)

    # Month 0: equity contributions (outflows)
    lp_cf[0] = -lp_invested
    gp_cf[0] = -gp_invested

    # Operating months (1 to n-2): pro-rata split of NOI-less-debt-service
    for i in range(1, n - 1):
        lp_cf[i] = cf[i] * lp_pct
        gp_cf[i] = cf[i] * gp_commit_pct

    # Exit month (n-1): full waterfall tiers
    remaining = float(cf[n - 1])
    lp_exit = gp_exit = 0.0

    # T1 — Return of capital (pro-rata)
    roc = min(remaining, initial_equity)
    lp_exit += roc * lp_pct
    gp_exit += roc * gp_commit_pct
    remaining -= roc

    # T2 — LP preferred return (compound, net of positive operating distributions already received)
    lp_pref_owed = lp_invested * ((1.0 + pref_rate) ** hold_period_yrs - 1.0)
    lp_ops_received = float(np.sum(np.maximum(lp_cf[1:n-1], 0.0)))
    lp_pref_remaining = max(0.0, lp_pref_owed - lp_ops_received)
    lp_pref_paid = min(remaining, lp_pref_remaining)
    lp_exit   += lp_pref_paid
    remaining -= lp_pref_paid

    # T3 — GP catch-up (100% GP until GP / (LP_pref + GP_catchup) == promote_rate)
    gp_catchup_owed = lp_pref_paid * promote_rate / (1.0 - promote_rate)
    gp_catchup_paid = min(remaining, gp_catchup_owed)
    gp_exit   += gp_catchup_paid
    remaining -= gp_catchup_paid

    # T4 — Promote split (check overall levered IRR vs. hurdle)
    try:
        overall_irr = (1.0 + npf.irr(cf)) ** 12 - 1.0
        if np.isnan(overall_irr): overall_irr = 0.0
    except:
        overall_irr = 0.0

    t4_lp = t4_gp = 0.0
    if remaining > 0:
        if overall_irr > hurdle_irr:
            t4_lp = remaining * (1.0 - promote_rate)
            t4_gp = remaining * promote_rate
        else:
            # Below hurdle — revert to pro-rata (no promote earned)
            t4_lp = remaining * lp_pct
            t4_gp = remaining * gp_commit_pct
        lp_exit += t4_lp
        gp_exit += t4_gp

    lp_cf[n - 1] = lp_exit
    gp_cf[n - 1] = gp_exit

    def safe_irr(cfs):
        try:
            r = (1.0 + npf.irr(cfs)) ** 12 - 1.0
            return float(r) if not np.isnan(r) and r > -0.99 else -1.0
        except:
            return -1.0

    lp_pos = float(np.sum(lp_cf[lp_cf > 0]))
    gp_pos = float(np.sum(gp_cf[gp_cf > 0]))

    # GP promote = excess GP profit above what a pure pro-rata split would have yielded
    lp_profit = lp_pos - lp_invested
    gp_profit  = gp_pos  - gp_invested
    pro_rata_gp_profit = lp_profit * (gp_commit_pct / lp_pct) if lp_pct > 0 else 0.0
    gp_promote_earned  = max(0.0, gp_profit - pro_rata_gp_profit)

    return {
        'lp_irr':            safe_irr(lp_cf),
        'gp_irr':            safe_irr(gp_cf),
        'lp_moic':           lp_pos / lp_invested if lp_invested > 0 else 0.0,
        'gp_moic':           gp_pos  / gp_invested if gp_invested > 0 else 0.0,
        'lp_invested':       lp_invested,
        'gp_invested':       gp_invested,
        'lp_total_return':   lp_pos,
        'gp_total_return':   gp_pos,
        'gp_promote_earned': gp_promote_earned,
        'pref_fully_paid':   lp_pref_paid >= lp_pref_remaining - 0.01,
        'tier_roc':          roc,
        'tier_lp_pref':      lp_pref_paid,
        'tier_gp_catchup':   gp_catchup_paid,
        'tier_promote_lp':   t4_lp,
        'tier_promote_gp':   t4_gp,
        'hurdle_cleared':    overall_irr > hurdle_irr,
        'lp_cf':             lp_cf,
        'gp_cf':             gp_cf,
    }


def run_monte_carlo(assumptions, gpr_total, iterations=250):
    base_cap    = assumptions['exit_cap_rate']
    base_rate   = assumptions['perm_rate']
    base_growth = assumptions['income_growth']

    # Correlation matrix: [exit_cap_rate, perm_rate, income_growth]
    # Rising rates push cap rates up; both compress rent growth.
    # Cholesky decomposition ensures draws respect this covariance structure.
    corr = np.array([
        [ 1.00,  0.70, -0.50],   # cap rate
        [ 0.70,  1.00, -0.30],   # perm rate
        [-0.50, -0.30,  1.00],   # income growth
    ])
    L = np.linalg.cholesky(corr)  # lower-triangular Cholesky factor

    # Independent standard normals → correlated shocks via L
    Z          = np.random.normal(0, 1, (3, iterations))
    correlated = L @ Z  # shape: (3, iterations)

    cap_rates      = base_cap    + correlated[0] * 0.005  # σ = 50 bps
    perm_rates     = base_rate   + correlated[1] * 0.005  # σ = 50 bps
    income_growths = base_growth + correlated[2] * 0.010  # σ = 100 bps

    results = []
    for i in range(iterations):
        sim = assumptions.copy()
        sim['exit_cap_rate'] = cap_rates[i]
        sim['perm_rate']     = perm_rates[i]
        sim['income_growth'] = income_growths[i]
        irr, _, _, _, _, _  = run_model_engine(sim, gpr_total)
        results.append(irr)

    return [r for r in results if r > -0.99]
