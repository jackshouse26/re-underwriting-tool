import streamlit as st
import pandas as pd
import numpy_financial as npf
from geopy.geocoders import Nominatim

# --- 1. UI: THE SIDEBAR (INPUTS) ---
st.sidebar.header("Property Details")
address = st.sidebar.text_input("Property Address", "11 Wall Street, New York, NY")

st.sidebar.header("Deal Assumptions")
purchase_price = st.sidebar.number_input("Purchase Price ($)", value=5000000, step=100000)
capex_budget = st.sidebar.number_input("CapEx Budget ($)", value=350000, step=50000)
year_1_noi = st.sidebar.number_input("Year 1 NOI ($)", value=300000, step=10000)
exit_cap_rate = st.sidebar.slider("Exit Cap Rate (%)", min_value=3.0, max_value=10.0, value=5.5, step=0.1) / 100
hold_period = st.sidebar.slider("Hold Period (Years)", min_value=1, max_value=10, value=5)

st.sidebar.header("Capital Stack")
ltv = st.sidebar.slider("Loan-to-Value (%)", min_value=0.0, max_value=80.0, value=65.0, step=1.0) / 100
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=2.0, max_value=10.0, value=6.0, step=0.1) / 100

st.sidebar.header("Equity Waterfall")
lp_contrib = st.sidebar.slider("LP Equity Contribution (%)", 50, 100, 90) / 100
gp_contrib = 1.0 - lp_contrib
tier_1_hurdle = st.sidebar.slider("Tier 1 Hurdle (Pref %)", 5.0, 12.0, 8.0, 0.5) / 100
tier_2_hurdle = st.sidebar.slider("Tier 2 Hurdle (%)", 10.0, 20.0, 15.0, 0.5) / 100
tier_2_gp_split = st.sidebar.slider("Tier 2 GP Promote (%)", 10, 50, 20, 5) / 100
tier_3_gp_split = st.sidebar.slider("Tier 3 GP Promote (%)", 20, 60, 40, 5) / 100

# --- 2. THE ENGINE (LOGIC) ---
def run_model():
    loan_amount = purchase_price * ltv
    total_cost = purchase_price + capex_budget
    total_equity = total_cost - loan_amount
    
    years = list(range(0, hold_period + 1))
    df = pd.DataFrame(index=years)
    
    df.loc[0, 'Unlevered_CF'] = -total_cost
    df.loc[0, 'Levered_CF'] = -total_equity
    
    annual_interest = loan_amount * interest_rate
    
    for year in range(1, hold_period + 1):
        noi = year_1_noi * ((1 + 0.03) ** (year - 1))
        df.loc[year, 'NOI'] = noi
        df.loc[year, 'Unlevered_CF'] = noi
        df.loc[year, 'Levered_CF'] = noi - annual_interest
        
    exit_noi = df.loc[hold_period, 'NOI'] * (1 + 0.03)
    net_sales_proceeds = (exit_noi / exit_cap_rate) * 0.98
    
    df.loc[hold_period, 'Unlevered_CF'] += net_sales_proceeds
    df.loc[hold_period, 'Levered_CF'] += (net_sales_proceeds - loan_amount)
    
    df.fillna(0, inplace=True)
    return df, total_equity, annual_interest

def calculate_waterfall(df, total_equity):
    df['LP_Cash_Flow'] = 0.0
    df['GP_Cash_Flow'] = 0.0
    
    lp_contribution_amt = total_equity * lp_contrib
    gp_contribution_amt = total_equity * gp_contrib
    
    df.loc[0, 'LP_Cash_Flow'] = -lp_contribution_amt
    df.loc[0, 'GP_Cash_Flow'] = -gp_contribution_amt
    
    t1_balance = total_equity
    t2_balance = total_equity

    for year in range(1, len(df)):
        cash = df.loc[year, 'Levered_CF']
        if cash <= 0:
            continue
            
        # TIER 1
        t1_balance = t1_balance * (1 + tier_1_hurdle)
        if cash <= t1_balance:
            t1_dist = cash; cash = 0
        else:
            t1_dist = t1_balance; cash -= t1_balance
        t1_balance -= t1_dist
        
        lp_t1 = t1_dist * lp_contrib; gp_t1 = t1_dist * gp_contrib
        
        # TIER 2
        t2_balance = t2_balance * (1 + tier_2_hurdle) - t1_dist
        if t2_balance > 0 and cash > 0:
            if cash <= t2_balance:
                t2_dist = cash; cash = 0
            else:
                t2_dist = t2_balance; cash -= t2_balance
            t2_balance -= t2_dist
        else:
            t2_dist = 0
            
        lp_t2 = t2_dist * (1 - tier_2_gp_split); gp_t2 = t2_dist * tier_2_gp_split
        
        # TIER 3
        t3_dist = cash
        lp_t3 = t3_dist * (1 - tier_3_gp_split); gp_t3 = t3_dist * tier_3_gp_split
        
        df.loc[year, 'LP_Cash_Flow'] = lp_t1 + lp_t2 + lp_t3
        df.loc[year, 'GP_Cash_Flow'] = gp_t1 + gp_t2 + gp_t3

    return df, lp_contribution_amt, gp_contribution_amt

# --- 3. RUNNING THE LOGIC ---
df_model, tot_equity, ann_interest = run_model()
df_waterfall, lp_invest, gp_invest = calculate_waterfall(df_model, tot_equity)

deal_irr = npf.irr(df_waterfall['Levered_CF'])
lp_irr = npf.irr(df_waterfall['LP_Cash_Flow'])
gp_irr = npf.irr(df_waterfall['GP_Cash_Flow'])

lp_moic = df_waterfall['LP_Cash_Flow'][df_waterfall['LP_Cash_Flow'] > 0].sum() / lp_invest if lp_invest > 0 else 0
gp_moic = df_waterfall['GP_Cash_Flow'][df_waterfall['GP_Cash_Flow'] > 0].sum() / gp_invest if gp_invest > 0 else 0

# DSCR Calculation
year_1_dscr = year_1_noi / ann_interest if ann_interest > 0 else 0
dscr_indicator = "🟢" if year_1_dscr >= 1.25 else "🔴"

# --- 4. UI: THE DASHBOARD (OUTPUTS) ---
st.title("Real Estate Underwriting Platform")

# Map Integration
try:
    geolocator = Nominatim(user_agent="re_underwriting_app")
    location = geolocator.geocode(address)
    if location:
        map_data = pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]})
        st.map(map_data, zoom=15)
except:
    st.info("Enter a valid address in the sidebar to view the property map.")

st.subheader("Return & Risk Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Project IRR", f"{deal_irr:.2%}")
col2.metric("LP IRR", f"{lp_irr:.2%}", f"{lp_moic:.2f}x MoIC")
col3.metric("GP IRR", f"{gp_irr:.2%}", f"{gp_moic:.2f}x MoIC")
col4.metric("Year 1 DSCR", f"{year_1_dscr:.2f}x", f"{dscr_indicator} Target: 1.25x")

st.divider()

st.subheader("Waterfall Cash Flows")
# Format for viewing
display_df = df_waterfall[['Levered_CF', 'LP_Cash_Flow', 'GP_Cash_Flow']].style.format("${:,.0f}")
st.dataframe(display_df)

# Download to CSV Functionality
@st.cache_data
def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')

csv_data = convert_df(df_waterfall)
st.download_button(
    label="📥 Download Pro Forma (CSV)",
    data=csv_data,
    file_name='Deal_Underwriting_Export.csv',
    mime='text/csv',
)

st.divider()

st.subheader("LP vs GP Distributions Over Time")
chart_data = df_waterfall[['LP_Cash_Flow', 'GP_Cash_Flow']]
st.bar_chart(chart_data)
