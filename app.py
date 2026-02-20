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

if "saved_deals" not in st.session_state:
    st.session_state.saved_deals = []

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
    
    # NEW: Phase 1 Mezzanine Debt
    st.divider()
    has_mezz = st.checkbox("Add Mezzanine Debt?")
    mezz_ltc = st.slider("Mezz Loan-to-Cost (%)", 0.0, 20.0, 10.0, 1.0) / 100 if has_mezz else 0.0
    mezz_rate = st.slider("Mezz Interest Rate (%)", 5.0, 15.0, 10.0, 0.1) / 100 if has_mezz else 0.0
    
    st.divider()
    st.write("**Permanent Loan (Refi)**")
    refi_month = st.slider("Refinance Month", const_months, hold_period_yrs * 12, const_months)
    perm_ltv = st.slider("Perm Loan-to-Value (%)", 0.0, 80.0, 65.0, 1.0) / 100
    perm_rate = st.slider("Perm Interest Rate (%)", 3.0, 10.0, 5.5, 0.1) / 100

with st.sidebar.expander("🌊 4. Equity Waterfall", expanded=False):
    lp_contrib = st.slider("LP Equity Contribution (%)", 50, 100, 90) / 100
    gp_contrib = 1.0 - lp_contrib
    
    # NEW: Phase 1 GP Catch-up Toggle
    has_catchup = st.checkbox("Enable 100% GP Catch-Up (Tier 2)")
    
    tier_1_hurdle = st.slider("Tier 1 Hurdle (Pref %)", 5.0, 15.0, 8.0, 0.5) / 100
    tier_2_hurdle = st.slider("Tier 2 Hurdle (%)", 10.0, 25.0, 15.0, 0.5) / 100
    tier_2_gp_split = st.slider("Tier 2 GP Promote (%)", 10, 50, 20, 5) / 100
    tier_3_gp_split = st.slider("Tier 3 GP Promote (%)", 20, 60, 40, 5) / 100


# --- 2. ENGINE: MONTHLY CASH FLOW & FINANCING ---
def run_monthly_model(calc_year_1_gpr, p_price=None, e_cap=None):
    if p_price is None: p_price = purchase_price
    if e_cap is None: e_cap = exit_cap_rate
    
    total_months = hold_period_yrs * 12
    df = pd.DataFrame(index=range(0, total_months + 1))
    
    cols = ['CapEx', 'Unlevered_CF', 'Const_Draw', 'Const_Balance', 'Const_Interest', 
            'Mezz_Balance', 'Mezz_Interest', 'Perm_Balance', 'Perm_Interest', 'Levered_CF', 'Debt_Service']
    for c in cols: df[c] = 0.0
        
    total_cost = p_price + capex_budget
    const_loan_max = total_cost * const_ltv
    mezz_loan_max = total_cost * mezz_ltc # NEW: Mezzanine Loan Calc
    
    initial_equity = total_cost - const_loan_max - mezz_loan_max
    
    total_sources = const_loan_max + mezz_loan_max + initial_equity
    sources_uses_match = abs(total_sources - total_cost) < 1
    
    df.loc[0, 'Unlevered_CF'] = -p_price
    df.loc[0, 'Levered_CF'] = -initial_equity
    
    monthly_capex = capex_budget / const_months if const_months > 0 else 0
    monthly_const_draw = (const_loan_max - (p_price * const_ltv)) / const_months if const_months > 0 else 0
    
    current_const_balance = p_price * const_ltv
    current_mezz_balance = mezz_loan_max # Funds at closing
    
    df.loc[0, 'Const_Balance'] = current_const_balance
    df.loc[0, 'Mezz_Balance'] = current_mezz_balance
    
    for m in range(1, total_months + 1):
        current_gpr = calc_year_1_gpr * ((1 + income_growth) ** (m/12))
        current_opex = year_1_opex * ((1 + expense_growth) ** (m/12))
        
        if has_abatement and m <= (abatement_years * 12):
            current_opex -= abatement_savings
            
        noi = (current_gpr - current_opex) / 12
        
        capex = monthly_capex if m <= const_months else 0
        df.loc[m, 'CapEx'] = -capex
        df.loc[m, 'Unlevered_CF'] = noi - capex
        
        if m <= refi_month:
            interest = current_const_balance * (const_rate / 12)
            mezz_int = current_mezz_balance * (mezz_rate / 12) # NEW: Mezz Interest
            
            draw = monthly_const_draw if m <= const_months else 0
            current_const_balance += draw
            
            df.loc[m, 'Const_Draw'] = draw
            df.loc[m, 'Const_Interest'] = -interest
            df.loc[m, 'Mezz_Interest'] = -mezz_int
            df.loc[m, 'Const_Balance'] = current_const_balance
            df.loc[m, 'Mezz_Balance'] = current_mezz_balance
            df.loc[m, 'Debt_Service'] = interest + mezz_int
            df.loc[m, 'Levered_CF'] = noi - capex + draw - interest - mezz_int
        
        elif m == refi_month + 1:
            forward_12m_noi = sum([ (calc_year_1_gpr * ((1 + income_growth) ** ((m+i)/12)) - 
                                   (year_1_opex * ((1 + expense_growth) ** ((m+i)/12)))) / 12 for i in range(12)])
            property_value = forward_12m_noi / e_cap
            perm_loan_amount = property_value * perm_ltv
            
            # Refi pays off senior AND mezzanine
            net_refi_proceeds = perm_loan_amount - current_const_balance - current_mezz_balance
            
            df.loc[m, 'Perm_Balance'] = perm_loan_amount
            interest = perm_loan_amount * (perm_rate / 12)
            mezz_int = current_mezz_balance * (mezz_rate / 12)
            
            df.loc[m, 'Perm_Interest'] = -interest
            df.loc[m, 'Mezz_Interest'] = -mezz_int
            df.loc[m, 'Debt_Service'] = interest + mezz_int
            df.loc[m, 'Levered_CF'] = noi + net_refi_proceeds - interest - mezz_int
            
            current_const_balance = 0 
            current_mezz_balance = 0
        else:
            perm_loan_amount = df.loc[m-1, 'Perm_Balance']
            interest = perm_loan_amount * (perm_rate / 12)
            df.loc[m, 'Perm_Balance'] = perm_loan_amount
            df.loc[m, 'Perm_Interest'] = -interest
            df.loc[m, 'Debt_Service'] = interest
            df.loc[m, 'Levered_CF'] = noi - interest
            
    exit_m = total_months
    # Forward Year NOI for accurate Terminal Value
    forward_12m_noi = sum([ (calc_year_1_gpr * ((1 + income_growth) ** ((exit_m+i)/12)) - 
                           (year_1_opex * ((1 + expense_growth) ** ((exit_m+i)/12)))) / 12 for i in range(1, 13)])
    gross_sales_price = forward_12m_noi / e_cap
    net_proceeds = gross_sales_price * 0.98 
    
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
            
        # NEW: GP Catch-up overwrites the Tier 2 split if checked
        eff_tier_2_gp = 1.0 if has_catchup else tier_2_gp_split
        lp_t2 = t2_dist * (1 - eff_tier_2_gp); gp_t2 = t2_dist * eff_tier_2_gp
        
        t3_dist = cash
        lp_t3 = t3_dist * (1 - tier_3_gp_split); gp_t3 = t3_dist * tier_3_gp_split
        
        df.loc[m, 'LP_Cash_Flow'] = lp_t1 + lp_t2 + lp_t3
        df.loc[m, 'GP_Cash_Flow'] = gp_t1 + gp_t2 + gp_t3

    return df, lp_invest, gp_invest

# --- 4. THE UI DASHBOARD & EXECUTION ---
st.title("🏢 Wes's Secret Underwriting Tool")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Dashboard", "🗓️ Monthly Pro Forma", "🔑 Rent Roll", "📈 Sensitivity"])

with tab3:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("📄 Rent Roll Extraction")
        st.write("Upload a broker PDF. Jack will extract, group, and average the unit data automatically.")
        uploaded_file = st.file_uploader("Upload PDF Rent Roll", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Extract Data", type="primary"):
                if "GEMINI_API_KEY" in st.secrets:
                    with st.spinner("Reading PDF and extracting data..."):
                        reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text() + "\n"
                        
                        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""
                        You are a real estate analyst. Extract the rent roll data from the following PDF text. 
                        Group the units by "Unit Type" (e.g., 1 Bed / 1 Bath, Studio, 2 Bed). 
                        For each group, provide the "Count" (total number of units of that type), the average "Sq Ft", and the average "Monthly Rent ($)".
                        Return EXACTLY a raw JSON array of objects. Do not include markdown blocks like ```json.
                        Example format: [{{"Unit Type": "1 Bed", "Count": 10, "Sq Ft": 750, "Monthly Rent ($)": 1500}}]
                        Text:
                        {pdf_text}
                        """
                        try:
                            response = model.generate_content(prompt)
                            raw_json = response.text.strip()
                            if raw_json.startswith("```json"): raw_json = raw_json[7:-3]
                            elif raw_json.startswith("```"): raw_json = raw_json[3:-3]
                            st.session_state.rent_roll_data = pd.DataFrame(json.loads(raw_json))
                            st.rerun()
                        except Exception as e:
                            st.error("Failed to parse the PDF cleanly. Ensure the document is readable text.")
                else:
                    st.warning("⚠️ Add GEMINI_API_KEY to Streamlit Secrets to use AI Extraction.")

    with col_b:
        st.subheader("🔑 Interactive Rent Roll")
        edited_rr = st.data_editor(st.session_state.rent_roll_data, num_rows="dynamic", use_container_width=True)
        dynamic_gpr = (edited_rr["Count"] * edited_rr["Monthly Rent ($)"] * 12).sum()
        st.info(f"**Total Calculated Year 1 Gross Potential Rent (GPR):** \${dynamic_gpr:,.0f}")

# RUN THE MATH ENGINE
df_model, initial_equity, total_uses, total_sources, is_balanced = run_monthly_model(dynamic_gpr)
df_wf, lp_invest, gp_invest = run_monthly_waterfall(df_model, initial_equity)

try:
    unlev_irr = (1 + npf.irr(df_wf['Unlevered_CF'], guess=0.1))**12 - 1
    lev_irr = (1 + npf.irr(df_wf['Levered_CF'], guess=0.1))**12 - 1
    lp_irr = (1 + npf.irr(df_wf['LP_Cash_Flow'], guess=0.1))**12 - 1
    gp_irr = (1 + npf.irr(df_wf['GP_Cash_Flow'], guess=0.1))**12 - 1
except:
    unlev_irr, lev_irr, lp_irr, gp_irr = 0, 0, 0, 0

unlev_moic = df_wf['Unlevered_CF'][df_wf['Unlevered_CF'] > 0].sum() / total_uses
lev_moic = df_wf['Levered_CF'][df_wf['Levered_CF'] > 0].sum() / initial_equity
lp_moic = df_wf['LP_Cash_Flow'][df_wf['LP_Cash_Flow'] > 0].sum() / lp_invest if lp_invest > 0 else 0
gp_moic = df_wf['GP_Cash_Flow'][df_wf['GP_Cash_Flow'] > 0].sum() / gp_invest if gp_invest > 0 else 0

max_ds = df_model['Debt_Service'].max()
dscr = ((dynamic_gpr - year_1_opex) / (max_ds * 12)) if max_ds > 0 else 0

with tab1:
    # NEW: Phase 1 OER Benchmarking Warning
    oer = year_1_opex / dynamic_gpr if dynamic_gpr > 0 else 0
    if oer < 0.35 and dynamic_gpr > 0:
        st.warning(f"⚠️ **High Risk Assumption:** Your Operating Expense Ratio (OER) is {oer:.1%}. Industry standard is 35%-45%. Lenders may discount your NOI.")

    if is_balanced:
        st.success(f"✅ Capital Stack Balanced | Total Sources: ${total_sources:,.0f} | Total Uses: ${total_uses:,.0f}")
    else:
        st.error(f"❌ Warning: Sources (${total_sources:,.0f}) do not match Uses (${total_uses:,.0f}).")
        
    st.subheader("Return Matrix")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1. Unlevered Deal", f"{unlev_irr:.2%}" if unlev_irr > -0.99 else "Loss", f"{unlev_moic:.2f}x MoIC")
    col2.metric("2. Levered Deal", f"{lev_irr:.2%}" if lev_irr > -0.99 else "Loss", f"{lev_moic:.2f}x MoIC")
    col3.metric("3. LP Returns", f"{lp_irr:.2%}" if lp_irr > -0.99 else "Loss", f"{lp_moic:.2f}x MoIC")
    col4.metric("4. GP Returns", f"{gp_irr:.2%}" if gp_irr > -0.99 else "Loss", f"{gp_moic:.2f}x MoIC")

    st.divider()
    
    # DEAL COMPARISON
    col_save, col_clear = st.columns([1, 4])
    with col_save:
        if st.button("💾 Bookmark Deal"):
            st.session_state.saved_deals.append({
                "Address": address, "Price": f"USD {purchase_price:,.0f}",
                "IRR": f"{lev_irr:.2%}", "DSCR": f"{dscr:.2f}x", "Equity": f"USD {initial_equity:,.0f}"
            })
            st.toast("Deal bookmarked for comparison!")

    if st.session_state.saved_deals:
        st.write("### ⚖️ Side-by-Side Comparison")
        st.table(pd.DataFrame(st.session_state.saved_deals))
        if st.button("🗑️ Clear All Saved Deals"):
            st.session_state.saved_deals = []
            st.rerun()
            
    st.divider()
    
    col_chart, col_map = st.columns([2, 1])
    with col_chart:
        st.write("**LP vs GP Cash Flow Timeline**")
        st.bar_chart(df_wf[['LP_Cash_Flow', 'GP_Cash_Flow']])
    
    with col_map:
        st.write("**Property Location**")
        try:
            geolocator = Nominatim(user_agent="re_underwriting_app")
            location = geolocator.geocode(address)
            if location:
                map_data = pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]})
                st.map(map_data, zoom=14)
        except:
            st.info("Map unavailable. Check address.")

    st.divider()
    
    st.subheader("Investment Memo")
    if "GEMINI_API_KEY" in st.secrets:
        if st.button("Generate Investment Committee Memo", type="primary"):
            with st.spinner("Jack is analyzing the underwriting and writing the memo..."):
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-2.5-flash')
                current_date = datetime.now().strftime("%B %d, %Y")
                abatement_text = f"The deal includes a tax abatement saving ${abatement_savings:,.0f}/year for {abatement_years} years." if has_abatement else "No tax abatements are modeled."
                
                prompt = f"""
                Act as a Principal at a Private Equity Real Estate firm. Write a highly professional, 3-paragraph Investment Committee (IC) Memo for a property located at {address}. The date is {current_date}.
                
                Data: Purchase: ${purchase_price:,.0f} | CapEx: ${capex_budget:,.0f} | Capitalization: ${total_sources:,.0f} | Const Loan: {const_ltv*100}% LTC at {const_rate*100}% | Refi: Month {refi_month} at {perm_ltv*100}% LTV and {perm_rate*100}%. {abatement_text}
                
                Returns: Deal Levered IRR: {lev_irr:.2%} | LP IRR: {lp_irr:.2%} | GP IRR: {gp_irr:.2%} | Year 1 DSCR: {dscr:.2f}x
                
                Format:
                **1. Executive Summary & Capital Stack:** Context, location, and capitalization.
                **2. Business Plan & Financing:** Transition from construction to permanent financing, plus abatement impact.
                **3. Return Profile & Recommendation:** LP vs GP alignment, leverage impact, and Go/No-Go recommendation.
                
                CRITICAL INSTRUCTION: Do NOT use the "$" symbol anywhere in your response. Always use "USD" instead.
                """
                
                response = model.generate_content(prompt)
                st.info(response.text)
    else:
        st.warning("⚠️ Add GEMINI_API_KEY to Streamlit Secrets to use AI.")

with tab4:
    st.subheader("📈 Levered IRR Sensitivity Analysis")
    st.write("Cross-tab of Purchase Price vs Exit Cap Rate")
    
    def get_irr_for_sens(p, c):
        df_m, eq, _, _, _ = run_monthly_model(dynamic_gpr, p_price=p, e_cap=c)
        df_w, _, _ = run_monthly_waterfall(df_m, eq)
        try:
            irr = (1 + npf.irr(df_w['Levered_CF'], guess=0.1))**12 - 1
            return irr if (not pd.isna(irr) and irr > -0.99) else -1.0
        except:
            return -1.0

    prices = [purchase_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    caps = [exit_cap_rate + offset for offset in [-0.01, -0.005, 0, 0.005, 0.01]]
    matrix = [[f"{get_irr_for_sens(p, c):.2%}" if get_irr_for_sens(p, c) > -0.99 else "Loss" for c in caps] for p in prices]
    st.table(pd.DataFrame(matrix, index=[f"USD {p:,.0f}" for p in prices], columns=[f"{c:.2%}" for c in caps]))

with tab2:
    st.subheader("🗓️ Complete Monthly Pro Forma")
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=True).encode('utf-8')

    csv_data = convert_df(df_wf)
    st.download_button(label="📥 Export to Excel (CSV)", data=csv_data, file_name='Deal_Underwriting_Export.csv', mime='text/csv')
    
    st.dataframe(df_wf.style.format("${:,.0f}"), height=600)
