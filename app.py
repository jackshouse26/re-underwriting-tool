import streamlit as st
import pandas as pd
import numpy_financial as npf
from geopy.geocoders import Nominatim
import google.generativeai as genai
from datetime import datetime
import PyPDF2
import json

# --- UI CONFIG ---
st.set_page_config(page_title="Real Estate Underwriting Pro", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1A1C24; border: 1px solid #2D303E;
        padding: 15px; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

if "rent_roll_data" not in st.session_state:
    st.session_state.rent_roll_data = pd.DataFrame([
        {"Unit Type": "1 Bed / 1 Bath", "Count": 10, "Sq Ft": 750, "Monthly Rent ($)": 1200},
        {"Unit Type": "2 Bed / 2 Bath", "Count": 5, "Sq Ft": 1000, "Monthly Rent ($)": 1600}
    ])

# --- SIDEBAR ---
st.sidebar.title("🏢 Deal Assumptions")
address = st.sidebar.text_input("Property Address", "11 Wall Street, New York, NY")

with st.sidebar.expander("🏗️ 1. Acquisition & CapEx", expanded=True):
    purchase_price = st.number_input("Purchase Price ($)", value=5000000, step=100000)
    capex_budget = st.number_input("Construction / CapEx ($)", value=1200000, step=50000)
    const_months = st.slider("Construction Duration (Months)", 0, 24, 12)
    hold_period_yrs = st.slider("Total Hold Period (Years)", 2, 10, 5)

with st.sidebar.expander("🏢 2. Operations & Exit", expanded=False):
    income_growth = st.slider("Annual Income Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    year_1_opex = st.number_input("Year 1 OpEx ($)", value=150000, step=5000)
    expense_growth = st.slider("Annual Expense Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    has_abatement = st.checkbox("Apply Tax Abatement?")
    abatement_savings = st.number_input("Annual Tax Savings ($)", value=50000) if has_abatement else 0
    abatement_yrs = st.slider("Abatement Years", 1, 10, 5) if has_abatement else 0
    exit_cap_rate = st.slider("Exit Cap Rate (%)", 4.0, 10.0, 5.5, 0.1) / 100

with st.sidebar.expander("🏦 3. Debt Financing", expanded=False):
    const_ltv = st.slider("Const. LTC (%)", 0.0, 85.0, 65.0, 1.0) / 100
    const_rate = st.slider("Const. Rate (%)", 4.0, 12.0, 8.0, 0.1) / 100
    refi_month = st.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
    perm_ltv = st.slider("Perm LTV (%)", 0.0, 80.0, 65.0, 1.0) / 100
    perm_rate = st.slider("Perm Rate (%)", 3.0, 10.0, 5.5, 0.1) / 100

# --- ENGINE ---
def run_model(p_price, e_cap, gpr):
    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    for c in ['Unlevered_CF', 'Levered_CF', 'Const_Balance', 'Perm_Balance', 'Debt_Service']: df[c] = 0.0
    
    total_cost = p_price + capex_budget
    loan_max = total_cost * const_ltv
    df.loc[0, 'Levered_CF'] = -(total_cost - loan_max)
    
    curr_bal = p_price * const_ltv
    for m in range(1, total_months + 1):
        noi = ((gpr * ((1+income_growth)**(m/12))) - (year_1_opex * ((1+expense_growth)**(m/12)))) / 12
        if has_abatement and m <= (abatement_yrs * 12): noi += (abatement_savings / 12)
        
        if m <= refi_month:
            ds = curr_bal * (const_rate / 12)
            draw = (loan_max - (p_price * const_ltv)) / const_months if m <= const_months else 0
            curr_bal += draw
            df.loc[m, 'Levered_CF'] = noi - (capex_budget/const_months if m <= const_months else 0) + draw - ds
        elif m == refi_month + 1:
            p_amt = ((noi * 12) / e_cap) * perm_ltv
            ds = p_amt * perm_rate / 12
            df.loc[m, 'Levered_CF'] = noi + (p_amt - curr_bal) - ds
            df.loc[m, 'Perm_Balance'], df.loc[m, 'Debt_Service'], curr_bal = p_amt, ds, 0
        else:
            p_bal = df.loc[m-1, 'Perm_Balance']
            ds = p_bal * perm_rate / 12
            df.loc[m, 'Perm_Balance'], df.loc[m, 'Debt_Service'] = p_bal, ds
            df.loc[m, 'Levered_CF'] = noi - ds

    exit_val = (noi * 12 / e_cap) * 0.98
    df.loc[total_months, 'Levered_CF'] += (exit_val - df.loc[total_months, 'Perm_Balance'])
    
    try:
        irr = (1 + npf.irr(df['Levered_CF']))**12 - 1
        if pd.isna(irr): return 0.0, 0.0
    except: return 0.0, 0.0
    
    dscr = (noi * 12) / (df['Debt_Service'].max() * 12) if df['Debt_Service'].max() > 0 else 0
    return irr, dscr

# --- UI ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🗓️ Pro Forma", "🔑 Rent Roll", "📈 Sensitivity"])

with tab3:
    edited_rr = st.data_editor(st.session_state.rent_roll_data, num_rows="dynamic", use_container_width=True)
    dynamic_gpr = (edited_rr["Count"] * edited_rr["Monthly Rent ($)"] * 12).sum()

irr, dscr = run_model(purchase_price, exit_cap_rate, dynamic_gpr)

with tab1:
    st.subheader("Key Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Levered IRR", f"{irr:.2%}")
    c2.metric("Year 1 DSCR", f"{dscr:.2f}x", delta="Target: 1.25x")
    
    if st.button("✨ Generate Memo"):
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Write 3-para IC Memo for {address}. Date: {datetime.now().strftime('%B %d, %Y')}. IRR: {irr:.2%}. DSCR: {dscr:.2f}. Use USD, no $ symbols."
            st.info(model.generate_content(prompt).text)

with tab4:
    st.subheader("Levered IRR Sensitivity Analysis")
    prices = [purchase_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    caps = [exit_cap_rate + offset for offset in [-0.01, -0.005, 0, 0.005, 0.01]]
    matrix = [[f"{run_model(p, c, dynamic_gpr)[0]:.2%}" if run_model(p, c, dynamic_gpr)[0] > -0.99 else "Neg." for c in caps] for p in prices]
    st.table(pd.DataFrame(matrix, index=[f"USD {p:,.0f}" for p in prices], columns=[f"{c:.2%}" for c in caps]))
