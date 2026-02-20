import streamlit as st
import pandas as pd
import numpy_financial as npf
from geopy.geocoders import Nominatim
import google.generativeai as genai
from datetime import datetime
import PyPDF2
import json

# --- UI UPGRADE: Page Config & Custom CSS ---
st.set_page_config(page_title="Real Estate Underwriting Pro", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1A1C24; 
        border: 1px solid #2D303E;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "rent_roll_data" not in st.session_state:
    st.session_state.rent_roll_data = pd.DataFrame([
        {"Unit Type": "1 Bed / 1 Bath", "Count": 10, "Sq Ft": 750, "Monthly Rent ($)": 1200},
        {"Unit Type": "2 Bed / 2 Bath", "Count": 5, "Sq Ft": 1000, "Monthly Rent ($)": 1600},
        {"Unit Type": "Studio", "Count": 2, "Sq Ft": 500, "Monthly Rent ($)": 900}
    ])

# --- 1. UI: THE SIDEBAR (INPUTS) ---
st.sidebar.title("🏢 Deal Assumptions")
address = st.sidebar.text_input("Property Address", "11 Wall Street, New York, NY")

with st.sidebar.expander("🏗️ 1. Acquisition & CapEx", expanded=True):
    purchase_price = st.number_input("Purchase Price ($)", value=5000000, step=100000)
    capex_budget = st.number_input("Construction / CapEx ($)", value=1200000, step=50000)
    const_months = st.slider("Construction Duration (Months)", 0, 24, 12)
    hold_period_yrs = st.slider("Total Hold Period (Years)", 2, 10, 5)

with st.sidebar.expander("🏢 2. Operations & Exit", expanded=False):
    st.info("💡 Base Rent is now calculated from the Rent Roll tab!")
    income_growth = st.slider("Annual Income Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    year_1_opex = st.number_input("Year 1 OpEx ($)", value=150000, step=5000)
    expense_growth = st.slider("Annual Expense Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    
    st.divider()
    has_abatement = st.checkbox("Apply Tax Abatement?")
    abatement_savings, abatement_years = 0, 0
    if has_abatement:
        abatement_savings = st.number_input("Annual Tax Savings ($)", value=50000, step=5000)
        abatement_years = st.slider("Abatement Duration (Years)", 1, 10, 5)
    
    st.divider()
    exit_cap_rate = st.sidebar.slider("Exit Cap Rate (%)", 4.0, 10.0, 5.5, 0.1) / 100

with st.sidebar.expander("🏦 3. Debt Financing", expanded=False):
    st.write("**Construction Loan**")
    const_ltv = st.slider("Const. Loan-to-Cost (%)", 0.0, 85.0, 65.0, 1.0) / 100
    const_rate = st.slider("Const. Interest Rate (%)", 4.0, 12.0, 8.0, 0.1) / 100
    st.divider()
    st.write("**Permanent Loan (Refi)**")
    refi_month = st.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
    perm_ltv = st.slider("Perm Loan-to-Value (%)", 0.0, 80.0, 65.0, 1.0) / 100
    perm_rate = st.slider("Perm Interest Rate (%)", 3.0, 10.0, 5.5, 0.1) / 100

with st.sidebar.expander("🌊 4. Equity Waterfall", expanded=False):
    lp_contrib = st.slider("LP Equity Contribution (%)", 50, 100, 90) / 100
    gp_contrib = 1.0 - lp_contrib
    tier_1_hurdle = st.slider("Tier 1 Hurdle (Pref %)", 5.0, 15.0, 8.0, 0.5) / 100
    tier_2_gp_split = st.slider("Tier 2 GP Promote (%)", 10, 50, 20, 5) / 100
    tier_3_gp_split = st.slider("Tier 3 GP Promote (%)", 20, 60, 40, 5) / 100

# --- 2. THE REUSABLE MATH ENGINE ---
def run_model_engine(p_price, e_cap, gpr_total):
    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    cols = ['CapEx', 'Unlevered_CF', 'Const_Draw', 'Const_Balance', 'Const_Interest', 'Perm_Balance', 'Perm_Interest', 'Levered_CF', 'Debt_Service']
    for c in cols: df[c] = 0.0
        
    total_cost = p_price + capex_budget
    loan_max = total_cost * const_ltv
    initial_equity = total_cost - loan_max
    df.loc[0, 'Unlevered_CF'] = -p_price
    df.loc[0, 'Levered_CF'] = -initial_equity
    
    current_bal = p_price * const_ltv
    m_capex = capex_budget / const_months if const_months > 0 else 0
    m_draw = (loan_max - (p_price * const_ltv)) / const_months if const_months > 0 else 0
    
    for m in range(1, total_months + 1):
        noi = ((gpr_total * ((1 + income_growth)**(m/12))) - (year_1_opex * ((1 + expense_growth)**(m/12)))) / 12
        if has_abatement and m <= (abatement_years * 12): noi += (abatement_savings / 12)
        
        capex = m_capex if m <= const_months else 0
        df.loc[m, 'Unlevered_CF'] = noi - capex
        
        if m <= refi_month:
            ds = current_bal * (const_rate / 12)
            draw = m_draw if m <= const_months else 0
            current_bal += draw
            df.loc[m, 'Const_Interest'] = -ds
            df.loc[m, 'Debt_Service'] = ds
            df.loc[m, 'Levered_CF'] = noi - capex + draw - ds
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

    exit_val = (df.loc[total_months, 'Unlevered_CF'] * 12 / e_cap) * 0.98
    df.loc[total_months, 'Unlevered_CF'] += exit_val
    df.loc[total_months, 'Levered_CF'] += (exit_val - df.loc[total_months, 'Perm_Balance'])
    
    try:
        irr = (1 + npf.irr(df['Levered_CF']))**12 - 1
        if pd.isna(irr): irr = 0.0
    except: irr = 0.0
    
    dscr = (noi * 12) / (df['Debt_Service'].max() * 12) if df['Debt_Service'].max() > 0 else 0
    return irr, dscr, df, initial_equity, total_cost

# --- 3. UI TABS & EXECUTION ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Dashboard", "🗓️ Monthly Pro Forma", "🔑 Rent Roll", "📈 Sensitivity"])

with tab3:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("📄 Rent Roll Extraction")
        uploaded_file = st.file_uploader("Upload PDF Rent Roll", type="pdf")
        if uploaded_file and st.button("Extract Data with AI", type="primary"):
            if "GEMINI_API_KEY" in st.secrets:
                with st.spinner("Extracting..."):
                    reader = PyPDF2.PdfReader(uploaded_file); pdf_text = "\n".join([p.extract_text() for p in reader.pages])
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"Extract rent roll to JSON: Unit Type, Count, Sq Ft, Monthly Rent ($). Text: {pdf_text}"
                    response = model.generate_content(prompt)
                    raw_json = response.text.replace("```json", "").replace("```", "").strip()
                    st.session_state.rent_roll_data = pd.DataFrame(json.loads(raw_json)); st.rerun()
    with col_b:
        edited_rr = st.data_editor(st.session_state.rent_roll_data, num_rows="dynamic", use_container_width=True)
        dynamic_gpr = (edited_rr["Count"] * edited_rr["Monthly Rent ($)"] * 12).sum()

# Run deal math for current inputs
lev_irr, dscr, df_wf, init_eq, tot_cost = run_model_engine(purchase_price, exit_cap_rate, dynamic_gpr)

with tab1:
    st.subheader("Key Deal Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Levered IRR", f"{lev_irr:.2%}")
    c2.metric("Year 1 DSCR", f"{dscr:.2f}x", delta="Lender Target: 1.25x")
    c3.metric("Initial Equity", f"USD {init_eq:,.0f}")
    
    st.divider()
    col_chart, col_map = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_wf['Levered_CF'])
    with col_map:
        try:
            geolocator = Nominatim(user_agent="re_pro")
            loc = geolocator.geocode(address)
            if loc: st.map(pd.DataFrame({'lat':[loc.latitude], 'lon':[loc.longitude]}), zoom=14)
        except: st.info("Map unavailable.")

    st.divider()
    if st.button("✨ Generate Investment Memo", type="primary"):
        if "GEMINI_API_KEY" in st.secrets:
            with st.spinner("Jack is analyzing your deal..."):
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Write 3-para IC Memo for {address}. Date: {datetime.now().strftime('%B %d, %Y')}. IRR: {lev_irr:.2%}. DSCR: {dscr:.2f}. No $ symbols, use USD."
                st.info(model.generate_content(prompt).text)

with tab4:
    st.subheader("📈 Levered IRR Sensitivity Analysis")
    st.write("Cross-tab of Purchase Price vs Exit Cap Rate")
    prices = [purchase_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    caps = [exit_cap_rate + offset for offset in [-0.01, -0.005, 0, 0.005, 0.01]]
    matrix = [[f"{run_model_engine(p, c, dynamic_gpr)[0]:.2%}" if run_model_engine(p, c, dynamic_gpr)[0] > -0.99 else "Neg." for c in caps] for p in prices]
    st.table(pd.DataFrame(matrix, index=[f"USD {p:,.0f}" for p in prices], columns=[f"{c:.2%}" for c in caps]))

with tab2:
    st.dataframe(df_wf.style.format("USD {:,.0f}"), height=600)
