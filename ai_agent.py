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
