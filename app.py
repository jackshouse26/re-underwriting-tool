import streamlit as st
import pandas as pd
import numpy_financial as npf

st.set_page_config(layout="wide") # Makes the dashboard wider for the big tables

# --- 1. UI: THE SIDEBAR (INPUTS) ---
st.sidebar.header("1. Deal Assumptions")
purchase_price = st.sidebar.number_input("Purchase Price ($)", value=5000000, step=100000)
capex_budget = st.sidebar.number_input("Construction / CapEx ($)", value=1200000, step=50000)
const_months = st.sidebar.slider("Construction Duration (Months)", 0, 24, 12)
hold_period_yrs = st.sidebar.slider("Total Hold Period (Years)", 2, 10, 5)

st.sidebar.header("2. Operations")
year_1_gpr = st.sidebar.number_input("Year 1 Gross Rent ($)", value=500000, step=10000)
income_growth = st.sidebar.slider("Annual Income Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
year_1_opex = st.sidebar.number_input("Year 1 OpEx ($)", value=150000, step=5000)
expense_growth = st.sidebar.slider("Annual Expense Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100

st.sidebar.subheader("Tax Abatement")
has_abatement = st.sidebar.checkbox("Apply Tax Abatement?")
abatement_savings = 0
abatement_years = 0
if has_abatement:
    abatement_savings = st.sidebar.number_input("Annual Tax Savings ($)", value=50000, step=5000)
    abatement_years = st.sidebar.slider("Abatement Duration (Years)", 1, 10, 5)
exit_cap_rate = st.sidebar.slider("Exit Cap Rate (%)", 4.0, 10.0, 5.5, 0.1) / 100

st.sidebar.header("3. Debt Financing")
st.sidebar.subheader("Construction Loan")
const_ltv = st.sidebar.slider("Const. Loan-to-Cost (%)", 0.0, 85.0, 65.0, 1.0) / 100
const_rate = st.sidebar.slider("Const. Interest Rate (%)", 4.0, 12.0, 8.0, 0.1) / 100

st.sidebar.subheader("Permanent Loan (Refi)")
refi_month = st.sidebar.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
perm_ltv = st.sidebar.slider("Perm Loan-to-Value (%)", 0.0, 80.0, 65.0, 1.0) / 100
perm_rate = st.sidebar.slider("Perm Interest Rate (%)", 3.0, 10.0, 5.5, 0.1) / 100

st.sidebar.header("4. Equity Waterfall")
lp_contrib = st.sidebar.slider("LP Equity Contribution (%)", 50, 100, 90) / 100
gp_contrib = 1.0 - lp_contrib
tier_1_hurdle = st.sidebar.slider("Tier 1 Hurdle (Pref %)", 5.0, 15.0, 8.0, 0.5) / 100
tier_2_hurdle = st.sidebar.slider("Tier 2 Hurdle (%)", 10.0, 25.0, 15.0, 0.5) / 100
tier_2_gp_split = st.sidebar.slider("Tier 2 GP Promote (%)", 10, 50, 20, 5) / 100
tier_3_gp_split = st.sidebar.slider("Tier 3 GP Promote (%)", 20, 60, 40, 5) / 100

# --- 2. ENGINE: MONTHLY CASH FLOW & FINANCING ---
def run_monthly_model():
    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    
    # 0. Initialize columns
    cols = ['CapEx', 'Unlevered_CF', 'Const_Draw', 'Const_Balance', 'Const_Interest', 
            'Perm_Balance', 'Perm_Interest', 'Levered_CF']
    for c in cols: df[c] = 0.0
        
    # 1. Day 1 Acquisition
    total_cost = purchase_price + capex_budget
    const_loan_max = total_cost * const_ltv
    initial_equity = total_cost - const_loan_max
    
    # Check: Sources vs Uses
    total_sources = const_loan_max + initial_equity
    sources_uses_match = abs(total_sources - total_cost) < 1
    
    df.loc[0, 'Unlevered_CF'] = -purchase_price
    df.loc[0, 'Levered_CF'] = -initial_equity
    
    # 2. Construction Draws & CapEx
    monthly_capex = capex_budget / const_months if const_months > 0 else 0
    monthly_const_draw = (const_loan_max - (purchase_price * const_ltv)) / const_months if const_months > 0 else 0
    
    current_const_balance = purchase_price * const_ltv
    df.loc[0, 'Const_Balance'] = current_const_balance
    
    # 3. Monthly Loop (Operations & Debt)
    for m in range(1, total_months + 1):
        # Operations
        current_gpr = year_1_gpr * ((1 + income_growth) ** (m/12))
        current_opex = year_1_opex * ((1 + expense_growth) ** (m/12))
        
        if has_abatement and m <= (abatement_years * 12):
            current_opex -= abatement_savings
            
        noi = (current_gpr - current_opex) / 12
        
        # CapEx Draw
        capex = monthly_capex if m <= const_months else 0
        df.loc[m, 'CapEx'] = -capex
        df.loc[m, 'Unlevered_CF'] = noi - capex
        
        # Debt Logic
        if m <= refi_month:
            # Construction Phase
            interest = current_const_balance * (const_rate / 12)
            draw = monthly_const_draw if m <= const_months else 0
            current_const_balance += draw
            
            df.loc[m, 'Const_Draw'] = draw
            df.loc[m, 'Const_Interest'] = -interest
            df.loc[m, 'Const_Balance'] = current_const_balance
            df.loc[m, 'Levered_CF'] = noi - capex + draw - interest
        
        elif m == refi_month + 1:
            # The Refinance Event!
            forward_12m_noi = sum([ (year_1_gpr * ((1 + income_growth) ** ((m+i)/12)) - 
                                   (year_1_opex * ((1 + expense_growth) ** ((m+i)/12)))) / 12 for i in range(12)])
            property_value = forward_12m_noi / exit_cap_rate
            perm_loan_amount = property_value * perm_ltv
            
            # Payoff construction loan, keep leftover cash
            net_refi_proceeds = perm_loan_amount - current_const_balance
            
            df.loc[m, 'Perm_Balance'] = perm_loan_amount
            interest = perm_loan_amount * (perm_rate / 12)
            df.loc[m, 'Perm_Interest'] = -interest
            
            df.loc[m, 'Levered_CF'] = noi + net_refi_proceeds - interest
            current_const_balance = 0 # Const loan is gone
        else:
            # Stabilized Perm Phase
            perm_loan_amount = df.loc[m-1, 'Perm_Balance']
            interest = perm_loan_amount * (perm_rate / 12)
            df.loc[m, 'Perm_Balance'] = perm_loan_amount
            df.loc[m, 'Perm_Interest'] = -interest
            df.loc[m, 'Levered_CF'] = noi - interest
            
    # 4. Exit Sale
    exit_m = total_months
    forward_12m_noi = sum([ (year_1_gpr * ((1 + income_growth) ** ((exit_m+i)/12)) - 
                           (year_1_opex * ((1 + expense_growth) ** ((exit_m+i)/12)))) / 12 for i in range(12)])
    gross_sales_price = forward_12m_noi / exit_cap_rate
    net_proceeds = gross_sales_price * 0.98 # 2% broker fee
    
    df.loc[exit_m, 'Unlevered_CF'] += net_proceeds
    df.loc[exit_m, 'Levered_CF'] += (net_proceeds - df.loc[exit_m, 'Perm_Balance'])
    
    return df, initial_equity, total_cost, total_sources, sources_uses_match

# --- 3. ENGINE: MONTHLY WATERFALL ---
def run_monthly_waterfall(df, total_equity):
    df['LP_Cash_Flow'] = 0.0
    df['GP_Cash_Flow'] = 0.0
    
    lp_invest = total_equity * lp_contrib
    gp_invest = total_equity * gp_contrib
    
    df.loc[0, 'LP_Cash_Flow'] = -lp_invest
    df.loc[0, 'GP_Cash_Flow'] = -gp_invest
    
    t1_bal = total_equity
    t2_bal = total_equity
    
    # Convert annual hurdles to monthly compounding rates
    m_hurdle_1 = (1 + tier_1_hurdle)**(1/12) - 1
    m_hurdle_2 = (1 + tier_2_hurdle)**(1/12) - 1

    for m in range(1, len(df)):
        cash = df.loc[m, 'Levered_CF']
        if cash <= 0: continue
            
        t1_bal = t1_bal * (1 + m_hurdle_1)
        if cash <= t1_bal:
            t1_dist = cash; cash = 0
        else:
            t1_dist = t1_bal; cash -= t1_bal
        t1_bal -= t1_dist
        
        lp_t1 = t1_dist * lp_contrib; gp_t1 = t1_dist * gp_contrib
        
        t2_bal = t2_bal * (1 + m_hurdle_2) - t1_dist
        if t2_bal > 0 and cash > 0:
            if cash <= t2_bal:
                t2_dist = cash; cash = 0
            else:
                t2_dist = t2_bal; cash -= t2_bal
            t2_bal -= t2_dist
        else:
            t2_dist = 0
            
        lp_t2 = t2_dist * (1 - tier_2_gp_split); gp_t2 = t2_dist * tier_2_gp_split
        
        t3_dist = cash
        lp_t3 = t3_dist * (1 - tier_3_gp_split); gp_t3 = t3_dist * tier_3_gp_split
        
        df.loc[m, 'LP_Cash_Flow'] = lp_t1 + lp_t2 + lp_t3
        df.loc[m, 'GP_Cash_Flow'] = gp_t1 + gp_t2 + gp_t3

    return df, lp_invest, gp_invest

# --- 4. EXECUTION & UI DASHBOARD ---
st.title("Institutional Real Estate Model (Monthly)")

# Run Math
df_model, initial_equity, total_uses, total_sources, is_balanced = run_monthly_model()
df_wf, lp_invest, gp_invest = run_monthly_waterfall(df_model, initial_equity)

# Data Sanity Check Alert
if is_balanced:
    st.success(f"✅ Sources & Uses Balanced! Total Capital: ${total_uses:,.0f}")
else:
    st.error(f"❌ Warning: Sources (${total_sources:,.0f}) do not match Uses (${total_uses:,.0f}).")

# Calculate IRRs (Convert Monthly IRR to Annualized IRR)
try:
    unlev_irr = (1 + npf.irr(df_wf['Unlevered_CF']))**12 - 1
    lev_irr = (1 + npf.irr(df_wf['Levered_CF']))**12 - 1
    lp_irr = (1 + npf.irr(df_wf['LP_Cash_Flow']))**12 - 1
    gp_irr = (1 + npf.irr(df_wf['GP_Cash_Flow']))**12 - 1
except:
    unlev_irr, lev_irr, lp_irr, gp_irr = 0, 0, 0, 0

# Calculate MoIC
unlev_moic = df_wf['Unlevered_CF'][df_wf['Unlevered_CF'] > 0].sum() / total_uses
lev_moic = df_wf['Levered_CF'][df_wf['Levered_CF'] > 0].sum() / initial_equity
lp_moic = df_wf['LP_Cash_Flow'][df_wf['LP_Cash_Flow'] > 0].sum() / lp_invest if lp_invest > 0 else 0
gp_moic = df_wf['GP_Cash_Flow'][df_wf['GP_Cash_Flow'] > 0].sum() / gp_invest if gp_invest > 0 else 0

st.subheader("Return Matrix")
# The Breakout Grid your friend asked for
col1, col2, col3, col4 = st.columns(4)
col1.metric("1. Unlevered Deal", f"{unlev_irr:.2%}", f"{unlev_moic:.2f}x MoIC", delta_color="off")
col2.metric("2. Levered Deal", f"{lev_irr:.2%}", f"{lev_moic:.2f}x MoIC", delta_color="off")
col3.metric("3. LP Returns", f"{lp_irr:.2%}", f"{lp_moic:.2f}x MoIC", delta_color="off")
col4.metric("4. GP Returns", f"{gp_irr:.2%}", f"{gp_moic:.2f}x MoIC", delta_color="off")

st.divider()

st.subheader("Monthly Cash Flow Timeline")
st.dataframe(df_wf.style.format("${:,.0f}"))
