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

# NEW: Master GPR state and God Mode states
if "active_gpr" not in st.session_state: st.session_state.active_gpr = 950000 # Healthy default for a $5M building
if "om_address" not in st.session_state: st.session_state.om_address = "11 Wall Street, New York, NY"
if "om_price" not in st.session_state: st.session_state.om_price = 5000000
if "om_capex" not in st.session_state: st.session_state.om_capex = 1200000
if "om_opex" not in st.session_state: st.session_state.om_opex = 150000
if "om_cap" not in st.session_state: st.session_state.om_cap = 5.5

API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None

# Live Treasury-based rate defaults (cached hourly)
_live_const_pct, _live_perm_pct = live_data.get_live_rates()
live_const_default = max(4.0, min(15.0, round(_live_const_pct, 1)))
live_perm_default  = max(3.0, min(12.0, round(_live_perm_pct,  1)))

# --- SIDEBAR INPUTS ---
st.sidebar.title("🏢 Deal Assumptions")

st.sidebar.subheader("🤖 God Mode: Auto-Fill OM")
om_file = st.sidebar.file_uploader("Upload Broker OM (PDF)", type="pdf")
if om_file and st.sidebar.button("Extract Deal Data", type="primary"):
    if API_KEY:
        with st.spinner("Jack is reading the OM and underwriting the deal..."):
            try:
                extracted = ai_agent.extract_om_data(om_file, API_KEY)
                raw_price = extracted.get("purchase_price", st.session_state.om_price)
                raw_cap   = float(extracted.get("exit_cap_rate", st.session_state.om_cap))
                raw_capex = extracted.get("capex_budget", st.session_state.om_capex)
                raw_opex  = extracted.get("year_1_opex", st.session_state.om_opex)

                # Sanity-check extracted values before populating sliders
                flags = []
                if not (500_000 <= int(raw_price) <= 500_000_000):
                    flags.append(f"Purchase price ${int(raw_price):,} is unusual (expected $500K–$500M)")
                if not (3.0 <= raw_cap <= 12.0):
                    flags.append(f"Cap rate {raw_cap:.2f}% is unusual (expected 3–12%)")
                if int(raw_capex) <= 0:
                    flags.append(f"CapEx ${int(raw_capex):,} is zero or negative — may be missing from OM")

                st.session_state.om_address = extracted.get("address", st.session_state.om_address)
                st.session_state.om_price = int(raw_price)
                st.session_state.om_capex = int(raw_capex)
                st.session_state.om_opex  = int(raw_opex)
                st.session_state.om_cap   = raw_cap

                if flags:
                    st.sidebar.warning("⚠️ **Review before proceeding:**\n" + "\n".join(f"• {f}" for f in flags))
                else:
                    st.sidebar.success("OM Extracted! Sliders Updated.")
                st.rerun()
            except Exception as e:
                st.sidebar.error("Could not cleanly parse the OM. Ensure it contains underwriting text.")
    else:
        st.sidebar.warning("Add GEMINI_API_KEY to secrets.")

st.sidebar.divider()

address = st.sidebar.text_input("Property Address", value=st.session_state.om_address)

with st.sidebar.expander("🏗️ 1. Acquisition & CapEx", expanded=True):
    purchase_price = st.number_input("Purchase Price ($)", value=st.session_state.om_price, step=100000)
    capex_budget = st.number_input("Construction / CapEx ($)", value=st.session_state.om_capex, step=50000)
    const_months = st.slider("Construction Duration (Months)", 0, 24, 12)
    hold_period_yrs = st.slider("Total Hold Period (Years)", 2, 10, 5)

with st.sidebar.expander("🏢 2. Operations & Exit", expanded=False):
    st.info("💡 Edit this manually, OR send data here from the Rent Roll tab!")
    # NEW: The Master GPR Input (tied to our session state)
    active_gpr = st.number_input("Year 1 Gross Potential Rent ($)", key="active_gpr", step=10000)
    
    use_market_rents = st.checkbox("📈 Underwrite to Market Rents?", value=False)
    income_growth = st.slider("Annual Income Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    year_1_opex = st.number_input("Year 1 OpEx ($)", value=st.session_state.om_opex, step=5000)
    expense_growth = st.slider("Annual Expense Growth (%)", 1.0, 10.0, 3.0, 0.5) / 100
    has_abatement = st.checkbox("Apply Tax Abatement?")
    abatement_savings = st.number_input("Annual Tax Savings ($)", value=50000, step=5000) if has_abatement else 0
    abatement_years = st.slider("Abatement Duration (Years)", 1, 10, 5) if has_abatement else 0
    exit_cap_rate = st.sidebar.slider("Exit Cap Rate (%)", 4.0, 10.0, float(st.session_state.om_cap), 0.1) / 100

with st.sidebar.expander("🏦 3. Debt Financing", expanded=False):
    st.caption(f"⚡ Live: SOFR+350bps = **{live_const_default:.1f}%** const | 10yr+200bps = **{live_perm_default:.1f}%** perm")
    const_ltv = st.slider("Const. Loan-to-Cost (%)", 0.0, 85.0, 65.0, 1.0) / 100
    const_rate = st.slider("Const. Interest Rate (%)", 4.0, 15.0, live_const_default, 0.1) / 100
    refi_month = st.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
    perm_ltv = st.slider("Perm Loan-to-Value (%)", 0.0, 80.0, 65.0, 1.0) / 100
    perm_rate = st.slider("Perm Interest Rate (%)", 3.0, 12.0, live_perm_default, 0.1) / 100
    st.divider()
    closing_costs_pct = st.slider("Acquisition Closing Costs (%)", 0.0, 3.0, 1.5, 0.1) / 100
    loan_orig_fee_pct = st.slider("Loan Origination Fee (%)", 0.0, 3.0, 1.0, 0.1) / 100
    exit_costs_pct    = st.slider("Exit Transaction Costs (%)", 0.5, 4.0, 2.0, 0.1) / 100

assumptions = {
    "purchase_price": purchase_price, "capex_budget": capex_budget, "const_months": const_months,
    "hold_period_yrs": hold_period_yrs, "income_growth": income_growth, "year_1_opex": year_1_opex,
    "expense_growth": expense_growth, "has_abatement": has_abatement, "abatement_savings": abatement_savings,
    "abatement_years": abatement_years, "exit_cap_rate": exit_cap_rate, "const_ltv": const_ltv,
    "const_rate": const_rate, "refi_month": refi_month, "perm_ltv": perm_ltv, "perm_rate": perm_rate,
    "closing_costs_pct": closing_costs_pct, "loan_orig_fee_pct": loan_orig_fee_pct, "exit_costs_pct": exit_costs_pct
}

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🗓️ Pro Forma", "🔑 Rent Roll", "📈 Sensitivity", "📋 Scenarios"])

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
        st.info(f"**Calculated Rent Roll GPR:** \${dynamic_gpr:,.0f}")
        
        # NEW: The Magic Button to push Rent Roll data to the math engine!
        if st.button("🚀 Apply Rent Roll to Deal", type="primary"):
            st.session_state.active_gpr = int(dynamic_gpr)
            st.rerun()

# RUN ENGINE VIA MODULE (Now using the sidebar's active_gpr!)
lev_irr, dscr, df_wf, init_eq, tot_cost, unlev_irr = math_engine.run_model_engine(assumptions, st.session_state.active_gpr)
max_debt_service = df_wf['Debt_Service'].max() * 12
breakeven_occ = (max_debt_service + year_1_opex) / st.session_state.active_gpr if st.session_state.active_gpr > 0 else 0

# MOIC: total equity returned / total equity invested
_lev = df_wf['Levered_CF']
_invested = _lev[_lev < 0].abs().sum()
_returned = _lev[_lev > 0].sum()
moic = _returned / _invested if _invested > 0 else 0

year_1_noi_annual = st.session_state.active_gpr - year_1_opex
going_in_cap = year_1_noi_annual / tot_cost if tot_cost > 0 else 0

# ── Pre-compute sensitivity matrix (used in tab4 and PDF) ────────────────────
sens_prices = [purchase_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
sens_caps   = [exit_cap_rate + d for d in [-0.01, -0.005, 0.0, 0.005, 0.01]]
sens_matrix = []
for _p in sens_prices:
    _row = []
    for _c in sens_caps:
        _r = math_engine.run_model_engine(assumptions, st.session_state.active_gpr, p_price_override=_p, e_cap_override=_c)[0]
        _row.append(f"{_r:.2%}" if _r > -0.99 else "Loss")
    sens_matrix.append(_row)

# ── Pre-compute Bear/Base/Bull with plain keys for PDF ───────────────────────
_pdf_scen_defs = {
    "Bear": {"price_delta":  0.05, "cap_delta":  0.0075, "growth_delta": -0.01, "rate_delta":  0.0075},
    "Base": {"price_delta":  0.0,  "cap_delta":  0.0,    "growth_delta":  0.0,  "rate_delta":  0.0},
    "Bull": {"price_delta": -0.05, "cap_delta": -0.005,  "growth_delta":  0.01, "rate_delta": -0.005},
}
pdf_scenario_results = {}
for _sname, _d in _pdf_scen_defs.items():
    _a = assumptions.copy()
    _a['purchase_price'] = purchase_price * (1 + _d['price_delta'])
    _a['exit_cap_rate']  = exit_cap_rate  + _d['cap_delta']
    _a['income_growth']  = income_growth  + _d['growth_delta']
    _a['perm_rate']      = perm_rate      + _d['rate_delta']
    _si, _sd, _sdf, _seq, _, _su = math_engine.run_model_engine(_a, st.session_state.active_gpr)
    _scf = _sdf['Levered_CF']
    _sm  = _scf[_scf > 0].sum() / _scf[_scf < 0].abs().sum() if _scf[_scf < 0].abs().sum() > 0 else 0
    pdf_scenario_results[_sname] = {
        "Levered IRR":   f"{_si:.2%}"  if _si  > -1 else "Loss",
        "Unlevered IRR": f"{_su:.2%}"  if _su  > -1 else "Loss",
        "MOIC":          f"{_sm:.2f}x",
        "DSCR":          f"{_sd:.2f}x",
        "Equity Req'd":  f"${_seq:,.0f}",
    }

with tab1:
    st.title("Wes's Secret Underwriting Tool")
    st.subheader("Key Deal Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Levered IRR",   f"{lev_irr:.2%}"   if lev_irr   > -1 else "Loss")
    c2.metric("Unlevered IRR", f"{unlev_irr:.2%}" if unlev_irr > -1 else "Loss")
    c3.metric("MOIC", f"{moic:.2f}x")

    c4, c5, c6 = st.columns(3)
    c4.metric("Year 1 DSCR", f"{dscr:.2f}x", delta="Target: 1.25x")
    c5.metric("Breakeven Occ.", f"{breakeven_occ:.1%}", delta="Risk Metric", delta_color="inverse")
    c6.metric("Equity Required", f"USD {init_eq:,.0f}")

    st.divider()
    st.subheader("Deal Health Check")
    h1, h2, h3, h4, h5 = st.columns(5)

    if lev_irr >= 0.15:
        h1.success(f"✅ **IRR**\n\n{lev_irr:.2%} ≥ 15%")
    else:
        h1.error(f"❌ **IRR**\n\n{lev_irr:.2%} < 15% hurdle")

    if dscr >= 1.25:
        h2.success(f"✅ **DSCR**\n\n{dscr:.2f}x ≥ 1.25x")
    else:
        h2.error(f"❌ **DSCR**\n\n{dscr:.2f}x < 1.25x")

    if exit_cap_rate >= going_in_cap:
        h3.success(f"✅ **Cap Rate**\n\nNo compression assumed")
    else:
        h3.warning(f"⚠️ **Cap Compression**\n\nExit {exit_cap_rate:.2%} < In {going_in_cap:.2%}")

    max_ltv = max(const_ltv, perm_ltv)
    if max_ltv <= 0.75:
        h4.success(f"✅ **Max LTV**\n\n{max_ltv:.0%} ≤ 75%")
    else:
        h4.warning(f"⚠️ **Max LTV**\n\n{max_ltv:.0%} > 75%")

    if breakeven_occ <= 0.85:
        h5.success(f"✅ **Breakeven Occ.**\n\n{breakeven_occ:.1%}")
    else:
        h5.warning(f"⚠️ **Breakeven Occ.**\n\n{breakeven_occ:.1%} > 85%")

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
            pdf_bytes = pdf_generator.create_pdf_report(
                address, purchase_price, init_eq, lev_irr, dscr, breakeven_occ,
                st.session_state.memo_text,
                unlev_irr=unlev_irr, moic=moic, total_cost=tot_cost, const_ltv=const_ltv,
                scenario_results=pdf_scenario_results,
                sensitivity_data={"prices": sens_prices, "caps": sens_caps, "matrix": sens_matrix},
            )
            st.download_button("📥 Download PDF", data=pdf_bytes, file_name=f"IC_Memo_{address.replace(' ', '_')}.pdf", mime="application/pdf", type="primary")

with tab4:
    st.subheader("📈 Levered IRR Sensitivity Analysis")
    st.table(pd.DataFrame(sens_matrix,
                          index=[f"USD {p:,.0f}" for p in sens_prices],
                          columns=[f"{c:.2%}" for c in sens_caps]))

    st.divider()
    st.subheader("🎲 Monte Carlo Risk Simulation")
    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running 250 deal simulations..."):
            import numpy as np
            sim_results = math_engine.run_monte_carlo(assumptions, st.session_state.active_gpr, iterations=250)
            
            if len(sim_results) > 0:
                avg_irr = sum(sim_results) / len(sim_results)
                success_rate = len([r for r in sim_results if r >= 0.15]) / len(sim_results)
                loss_rate = len([r for r in sim_results if r <= 0.0]) / len(sim_results)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Average Expected IRR", f"{avg_irr:.2%}")
                c2.metric("Probability of hitting 15% IRR", f"{success_rate:.0%}")
                c3.metric("Probability of losing Equity", f"{loss_rate:.0%}")
                
                hist_data = pd.DataFrame(sim_results, columns=['IRR'])
                hist_data['IRR (%)'] = hist_data['IRR'] * 100
                counts, bins = np.histogram(hist_data['IRR (%)'], bins=20)
                bin_labels = [f"{b:.1f}%" for b in bins[:-1]]
                chart_df = pd.DataFrame({'Frequency': counts}, index=bin_labels)
                st.bar_chart(chart_df)
            else:
                st.error("Simulation resulted in total losses. Adjust your baseline inputs.")

with tab2:
    st.dataframe(df_wf.style.format("USD {:,.0f}"), height=600)

with tab5:
    st.subheader("Bear / Base / Bull Scenarios")
    st.caption("Fixed adjustments applied on top of your base assumptions. Tweak the sidebar to shift all three scenarios together.")

    scenario_defs = {
        "🐻 Bear": {"price_delta":  0.05, "cap_delta":  0.0075, "growth_delta": -0.01, "rate_delta":  0.0075},
        "📊 Base": {"price_delta":  0.0,  "cap_delta":  0.0,    "growth_delta":  0.0,  "rate_delta":  0.0},
        "🐂 Bull": {"price_delta": -0.05, "cap_delta": -0.005,  "growth_delta":  0.01, "rate_delta": -0.005},
    }

    assumption_rows = {
        "Purchase Price": [f"USD {purchase_price * (1 + d['price_delta']):,.0f}" for d in scenario_defs.values()],
        "Exit Cap Rate":  [f"{exit_cap_rate  + d['cap_delta']:.2%}"              for d in scenario_defs.values()],
        "Income Growth":  [f"{income_growth  + d['growth_delta']:.1%}"           for d in scenario_defs.values()],
        "Perm Rate":      [f"{perm_rate      + d['rate_delta']:.2%}"             for d in scenario_defs.values()],
    }
    st.caption("**Scenario Assumptions**")
    st.table(pd.DataFrame(assumption_rows, index=list(scenario_defs.keys())).T)

    st.divider()

    results = {}
    for name, deltas in scenario_defs.items():
        s_assum = assumptions.copy()
        s_assum['purchase_price'] = purchase_price * (1 + deltas['price_delta'])
        s_assum['exit_cap_rate']  = exit_cap_rate  + deltas['cap_delta']
        s_assum['income_growth']  = income_growth  + deltas['growth_delta']
        s_assum['perm_rate']      = perm_rate      + deltas['rate_delta']
        s_irr, s_dscr, s_df, s_eq, _, s_unlev = math_engine.run_model_engine(s_assum, st.session_state.active_gpr)
        s_cf = s_df['Levered_CF']
        s_moic = s_cf[s_cf > 0].sum() / s_cf[s_cf < 0].abs().sum() if s_cf[s_cf < 0].abs().sum() > 0 else 0
        results[name] = {
            "Levered IRR":    f"{s_irr:.2%}"   if s_irr   > -1 else "Loss",
            "Unlevered IRR":  f"{s_unlev:.2%}" if s_unlev > -1 else "Loss",
            "MOIC":           f"{s_moic:.2f}x",
            "DSCR":           f"{s_dscr:.2f}x",
            "Equity Required": f"USD {s_eq:,.0f}",
        }

    st.caption("**Scenario Results**")
    st.table(pd.DataFrame(results))
