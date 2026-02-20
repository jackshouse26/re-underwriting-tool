import pandas as pd
import numpy_financial as npf

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

    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    cols = ['CapEx', 'Unlevered_CF', 'Const_Draw', 'Const_Balance', 'Const_Interest', 'Perm_Balance', 'Perm_Interest', 'Levered_CF', 'Debt_Service']
    for c in cols: df[c] = 0.0
        
    total_cost = p_price + capex_budget
    loan_max = total_cost * const_ltv
    initial_equity = total_cost - loan_max
    
    df.loc[0, 'Unlevered_CF'] = -total_cost
    df.loc[0, 'Levered_CF'] = -initial_equity
    
    current_bal = p_price * const_ltv
    m_draw = (loan_max - (p_price * const_ltv)) / const_months if const_months > 0 else 0
    
    last_noi = 0
    
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
            ds = p_amt * perm_rate / 12
            df.loc[m, 'Levered_CF'] = noi + (p_amt - current_bal) - ds
            df.loc[m, 'Perm_Balance'], df.loc[m, 'Debt_Service'] = p_amt, ds
            current_bal = 0
        else:
            p_bal = df.loc[m-1, 'Perm_Balance']
            ds = p_bal * perm_rate / 12
            df.loc[m, 'Perm_Balance'], df.loc[m, 'Debt_Service'] = p_bal, ds
            df.loc[m, 'Levered_CF'] = noi - ds

    exit_noi = ((gpr_total * ((1 + income_growth)**((total_months+1)/12))) - (year_1_opex * ((1 + expense_growth)**((total_months+1)/12))))
    exit_val = (exit_noi / e_cap) * 0.98
    
    df.loc[total_months, 'Unlevered_CF'] += exit_val
    df.loc[total_months, 'Levered_CF'] += (exit_val - df.loc[total_months, 'Perm_Balance'])
    
    try:
        irr = (1 + npf.irr(df['Levered_CF'], guess=0.1))**12 - 1
        if pd.isna(irr) or irr < -0.99: irr = -1.0
    except: irr = -1.0
    
    dscr = (last_noi * 12) / (df['Debt_Service'].max() * 12) if df['Debt_Service'].max() > 0 else 0
    return irr, dscr, df, initial_equity, total_cost
