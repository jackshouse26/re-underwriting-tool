import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim

# Import our custom modules
import math_engine
import ai_agent
import pdf_generator
import live_data 

st.set_page_config(page_title="Real Estate Underwriting Pro", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1A1C24; border: 1px solid #2D303E;
        padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# --- STATE INIT ---
if "rent_roll_data" not in st.session_state:
    st.session_state.rent_roll_data = pd.DataFrame([
        {"Unit Type": "1 Bed / 1 Bath", "Count": 10, "Sq Ft": 750, "Current Rent ($)": 1200, "Market Rent ($)": 1400},
        {"Unit Type": "2 Bed / 2 Bath", "Count": 5, "Sq Ft": 1000, "Current Rent ($)": 1600, "Market Rent ($)": 1850}
    ])
else:
    if "Monthly Rent ($)" in st.session_state.rent_roll_data.columns:
        st.session_state.rent_roll_data.rename(columns={"Monthly Rent ($)": "Current Rent ($)"}, inplace=True)
        if "Market Rent ($)" not in st.session_state.rent_roll_data.columns:
            st.session_state.rent_roll_data["Market Rent ($)"] = st.session_state.rent_roll_data["Current Rent ($)"] * 1.15

if "saved_deals" not in st.session_state: st.session_state.saved_deals = []
if "memo_text" not in st.session_state: st.session_state.memo_text = ""

API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None

# --- SIDEBAR INPUTS ---
st.sidebar.title("🏢 Deal Assumptions")
address = st.sidebar.text_input("Property Address", "11 Wall Street, New York, NY")

with st.sidebar.expander("🏗️ 1. Acquisition & CapEx", expanded=True):
    purchase_price = st.number_input("Purchase Price ($)", value=5000000, step=100000)
    capex_budget = st.number_input("Construction / CapEx ($)", value=1200000, step=50000)
    const_months = st.slider("Construction Duration (Months)", 0, 24, 12)
    hold_period_yrs = st.slider("Total Hold Period (Years)", 2, 10, 5)

with st.sidebar.expander("🏢 2. Operations & Exit", expanded=False):
    use_market_rents = st.checkbox("📈 Underwrite to Market Rents?", value=False)
    income_growth = st.slider("Annual Income Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    year_1_opex = st.number_input("Year 1 OpEx ($)", value=150000, step=5000)
    expense_growth = st.slider("Annual Expense Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    has_abatement = st.checkbox("Apply Tax Abatement?")
    abatement_savings = st.number_input("Annual Tax Savings ($)", value=50000, step=5000) if has_abatement else 0
    abatement_years = st.slider("Abatement Duration (Years)", 1, 10, 5) if has_abatement else 0
    exit_cap_rate = st.sidebar.slider("Exit Cap Rate (%)", 4.0, 10.0, 5.5, 0.1) / 100

# Fetch live rates before drawing the sliders!
live_const, live_perm = live_data.get_live_rates()

with st.sidebar.expander("🏦 3. Debt Financing", expanded=False):
    st.write("**Construction Loan**")
    const_ltv = st.slider("Const. Loan-to-Cost (%)", 0.0, 85.0, 65.0, 1.0) / 100
    # Notice the default value is now `live_const`
    const_rate = st.slider("Const. Interest Rate (%)", 4.0, 15.0, live_const, 0.1) / 100
    
    st.divider()
    has_mezz = st.checkbox("Add Mezzanine Debt?")
    mezz_ltc = st.slider("Mezz Loan-to-Cost (%)", 0.0, 20.0, 10.0, 1.0) / 100 if has_mezz else 0.0
    mezz_rate = st.slider("Mezz Interest Rate (%)", 5.0, 15.0, 10.0, 0.1) / 100 if has_mezz else 0.0
    
    st.divider()
    st.write("**Permanent Loan (Refi)**")
    refi_month = st.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
    perm_ltv = st.slider("Perm Loan-to-Value (%)", 0.0, 80.0, 65.0, 1.0) / 100
    # Notice the default value is now `live_perm`
    perm_rate = st.slider("Perm Interest Rate (%)", 3.0, 12.0, live_perm, 0.1) / 100

# Pack all inputs into a dictionary for our new module
assumptions = {
    "purchase_price": purchase_price, "capex_budget": capex_budget, "const_months": const_months,
    "hold_period_yrs": hold_period_yrs, "income_growth": income_growth, "year_1_opex": year_1_opex,
    "expense_growth": expense_growth, "has_abatement": has_abatement, "abatement_savings": abatement_savings,
    "abatement_years": abatement_years, "exit_cap_rate": exit_cap_rate, "const_ltv": const_ltv,
    "const_rate": const_rate, "refi_month": refi_month, "perm_ltv": perm_ltv, "perm_rate": perm_rate
}

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🗓️ Pro Forma", "🔑 Rent Roll", "📈 Sensitivity"])

with tab3:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("📄 AI Rent Roll Extraction")
        uploaded_file = st.file_uploader("Upload PDF Rent Roll", type="pdf")
        if uploaded_file and st.button("Extract Data with AI", type="primary"):
            if API_KEY:
                with st.spinner("Extracting..."):
                    st.session_state.rent_roll_data = ai_agent.extract_rent_roll_from_pdf(uploaded_file, API_KEY)
                    st.rerun()
            else: st.warning("Add GEMINI_API_KEY to secrets.")
    with col_b:
        edited_rr = st.data_editor(st.session_state.rent_roll_data, num_rows="dynamic", use_container_width=True)
        rent_col = "Market Rent ($)" if use_market_rents and "Market Rent ($)" in edited_rr.columns else "Current Rent ($)"
        dynamic_gpr = (edited_rr["Count"] * edited_rr[rent_col] * 12).sum()
        st.info(f"**Calculated Year 1 GPR:** \${dynamic_gpr:,.0f}")

# RUN ENGINE VIA MODULE
lev_irr, dscr, df_wf, init_eq, tot_cost = math_engine.run_model_engine(assumptions, dynamic_gpr)
max_debt_service = df_wf['Debt_Service'].max() * 12
breakeven_occ = (max_debt_service + year_1_opex) / dynamic_gpr if dynamic_gpr > 0 else 0

with tab1:
    st.subheader("Key Deal Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Levered IRR", f"{lev_irr:.2%}" if lev_irr > -1 else "Loss")
    c2.metric("Year 1 DSCR", f"{dscr:.2f}x", delta="Target: 1.25x")
    c3.metric("Breakeven Occ.", f"{breakeven_occ:.1%}", delta="Risk Metric", delta_color="inverse")
    c4.metric("Equity Required", f"USD {init_eq:,.0f}")
    
    st.divider()
    col_chart, col_map = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_wf['Levered_CF'])
    with col_map:
        try:
            loc = Nominatim(user_agent="re_pro").geocode(address)
            if loc: st.map(pd.DataFrame({'lat':[loc.latitude], 'lon':[loc.longitude]}), zoom=14)
        except: pass
        if st.button("🏙️ AI Neighborhood Context") and API_KEY:
             with st.spinner("Researching..."):
                 st.info(ai_agent.get_neighborhood_context(address, API_KEY))

    st.divider()
    col_memo, col_pdf = st.columns([3, 1])
    with col_memo:
        if st.button("✨ Generate Memo", type="primary") and API_KEY:
            with st.spinner("Analyzing..."):
                r_type = "Market Rents" if use_market_rents else "Current In-Place Rents"
                st.session_state.memo_text = ai_agent.generate_investment_memo(address, r_type, lev_irr, dscr, API_KEY)
                st.rerun()

    if st.session_state.memo_text:
        st.info(st.session_state.memo_text)
        with col_pdf:
            pdf_bytes = pdf_generator.create_pdf_report(address, purchase_price, init_eq, lev_irr, dscr, breakeven_occ, st.session_state.memo_text)
            st.download_button("📥 Download PDF", data=pdf_bytes, file_name=f"IC_Memo_{address.replace(' ', '_')}.pdf", mime="application/pdf", type="primary")

with tab4:
    st.subheader("📈 Levered IRR Sensitivity Analysis")
    st.write("Cross-tab of Purchase Price vs Exit Cap Rate")
    prices = [purchase_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    caps = [exit_cap_rate + offset for offset in [-0.01, -0.005, 0, 0.005, 0.01]]
    matrix_data = [[f"{math_engine.run_model_engine(assumptions, dynamic_gpr, p_price_override=p, e_cap_override=c)[0]:.2%}" if math_engine.run_model_engine(assumptions, dynamic_gpr, p_price_override=p, e_cap_override=c)[0] > -0.99 else "Loss" for c in caps] for p in prices]
    st.table(pd.DataFrame(matrix_data, index=[f"USD {p:,.0f}" for p in prices], columns=[f"{c:.2%}" for c in caps]))

    st.divider()
    
    # --- NEW MONTE CARLO SECTION ---
    st.subheader("🎲 Monte Carlo Risk Simulation")
    st.write("Run 250 parallel universes of your deal by introducing historical volatility into your Exit Cap Rate, Permanent Interest Rate, and Rent Growth assumptions.")
    
    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running 250 deal simulations..."):
            import numpy as np
            sim_results = math_engine.run_monte_carlo(assumptions, dynamic_gpr, iterations=250)
            
            if len(sim_results) > 0:
                avg_irr = sum(sim_results) / len(sim_results)
                success_rate = len([r for r in sim_results if r >= 0.15]) / len(sim_results)
                loss_rate = len([r for r in sim_results if r <= 0.0]) / len(sim_results)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Average Expected IRR", f"{avg_irr:.2%}")
                c2.metric("Probability of hitting 15% IRR", f"{success_rate:.0%}")
                c3.metric("Probability of losing Equity", f"{loss_rate:.0%}")
                
                # Create the Bell Curve Histogram
                hist_data = pd.DataFrame(sim_results, columns=['IRR'])
                hist_data['IRR (%)'] = hist_data['IRR'] * 100
                counts, bins = np.histogram(hist_data['IRR (%)'], bins=20)
                bin_labels = [f"{b:.1f}%" for b in bins[:-1]]
                chart_df = pd.DataFrame({'Frequency': counts}, index=bin_labels)
                
                st.write("**IRR Distribution Curve**")
                st.bar_chart(chart_df)
            else:
                st.error("Simulation resulted in total losses across all scenarios. Lower purchase price.")

with tab2:
    st.dataframe(df_wf.style.format("USD {:,.0f}"), height=600)
