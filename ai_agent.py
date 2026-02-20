import google.generativeai as genai
import PyPDF2
import json
import pandas as pd
from datetime import datetime

def extract_rent_roll_from_pdf(uploaded_file, api_key):
    reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = "\n".join([p.extract_text() for p in reader.pages])
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Extract rent roll to JSON: Unit Type, Count, Sq Ft, Current Rent ($), Market Rent ($). Text: {pdf_text}"
    response = model.generate_content(prompt)
    raw_json = response.text.replace("```json", "").replace("```", "").strip()
    return pd.DataFrame(json.loads(raw_json))

def generate_investment_memo(address, rent_type, lev_irr, dscr, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Write 3-para IC Memo for {address}. Date: {datetime.now().strftime('%B %d, %Y')}. Underwritten to: {rent_type}. IRR: {lev_irr:.2%}. DSCR: {dscr:.2f}. No $ symbols, use USD. Plain text only, no markdown bolding."
    response = model.generate_content(prompt)
    return response.text.replace("*", "")

def get_neighborhood_context(address, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    context_prompt = f"Act as a real estate acquisitions analyst. Provide a brief 2-paragraph neighborhood analysis for {address}. Highlight potential economic drivers, transit access, and typical zoning/development trends in this specific submarket."
    return model.generate_content(context_prompt).text

def extract_om_data(uploaded_file, api_key):
    reader = PyPDF2.PdfReader(uploaded_file)
    # Extract text from every page of the OM
    pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    genai.configure(api_key=api_key)
    # Gemini 2.5 Flash has a massive context window, easily handling 60+ page PDFs
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Act as an elite Real Estate Private Equity Acquisitions Analyst. 
    Read this Offering Memorandum (OM) and extract the key underwriting assumptions.
    Return EXACTLY a raw JSON object with these exact keys. If a value isn't explicitly stated, make your best educated guess based on the market or document context. Do not use commas in numbers.
    {{
        "address": "Property Address, City, ST",
        "purchase_price": 5000000, 
        "capex_budget": 1200000,
        "year_1_opex": 150000,
        "exit_cap_rate": 5.5
    }}
    Do not include markdown formatting like ```json. Just return the raw JSON.
    
    OM Text:
    {pdf_text}
    """
    response = model.generate_content(prompt)
    raw_json = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw_json)
