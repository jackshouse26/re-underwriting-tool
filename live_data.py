import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600) # Caches the data for 1 hour so your app doesn't slow down
def get_live_rates():
    try:
        # Pull live 10-Year Treasury Yield (^TNX)
        tnx = yf.Ticker("^TNX")
        ten_year_yield = tnx.history(period="1d")['Close'].iloc[-1]
        
        # Pull live 13-Week Treasury Bill (^IRX) - proxy for short-term/SOFR rates
        irx = yf.Ticker("^IRX")
        short_term_rate = irx.history(period="1d")['Close'].iloc[-1]
        
        # Calculate realistic lender rates (Base + Spread)
        # Perm rate = 10-Year + 200 bps spread
        live_perm_rate = ten_year_yield + 2.0  
        
        # Const rate = Short-term + 350 bps spread
        live_const_rate = short_term_rate + 3.5 
        
        return float(live_const_rate), float(live_perm_rate)
        
    except Exception as e:
        # Bulletproof fallback if Yahoo Finance is ever down
        return 8.0, 5.5
