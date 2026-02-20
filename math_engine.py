import pandas as pd
import numpy_financial as npf
import numpy as np # <-- Moved to the top!

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
        # THE FIX: Removed guess=0.1
        irr = (1 + npf.irr(df['Levered_CF']))**12 - 1
        if pd.isna(irr) or irr < -0.99: irr = -1.0
    except: irr = -1.0

    try:
        unlev_irr = (1 + npf.irr(df['Unlevered_CF']))**12 - 1
        if pd.isna(unlev_irr) or unlev_irr < -0.99: unlev_irr = -1.0
    except: unlev_irr = -1.0

    dscr = (last_noi * 12) / (df['Debt_Service'].max() * 12) if df['Debt_Service'].max() > 0 else 0
    return irr, dscr, df, initial_equity, total_cost, unlev_irr


def run_monte_carlo(assumptions, gpr_total, iterations=250):
    base_cap = assumptions['exit_cap_rate']
    base_rate = assumptions['perm_rate']
    base_growth = assumptions['income_growth']
    
    # Generate 250 random variations of our core assumptions
    cap_rates = np.random.normal(loc=base_cap, scale=0.005, size=iterations) # +/- 50 bps
    perm_rates = np.random.normal(loc=base_rate, scale=0.005, size=iterations) # +/- 50 bps
    income_growths = np.random.normal(loc=base_growth, scale=0.01, size=iterations) # +/- 1%
    
    results = []
    
    # Loop the deal 250 times with the new random variables
    for i in range(iterations):
        sim_assumptions = assumptions.copy()
        sim_assumptions['exit_cap_rate'] = cap_rates[i]
        sim_assumptions['perm_rate'] = perm_rates[i]
        sim_assumptions['income_growth'] = income_growths[i]
        
        # Run the core engine using the mutated assumptions
        irr, _, _, _, _, _ = run_model_engine(sim_assumptions, gpr_total)
        results.append(irr)
        
    return [r for r in results if r > -0.99] # Return all non-total-loss scenarios
