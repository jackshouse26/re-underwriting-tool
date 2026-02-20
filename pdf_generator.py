from fpdf import FPDF

def create_pdf_report(address, purchase_price, init_eq, lev_irr, dscr, breakeven_occ, memo_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Investment Committee Summary", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Property: {address}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Purchase Price: USD {purchase_price:,.0f}", ln=True)
    pdf.cell(200, 8, txt=f"Equity Required: USD {init_eq:,.0f}", ln=True)
    pdf.cell(200, 8, txt=f"Levered IRR: {lev_irr:.2%}", ln=True)
    pdf.cell(200, 8, txt=f"Year 1 DSCR: {dscr:.2f}x", ln=True)
    pdf.cell(200, 8, txt=f"Breakeven Occupancy: {breakeven_occ:.1%}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Sponsor Rationale & Business Plan:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, txt=memo_text)
    
    try:
        return pdf.output(dest='S').encode('latin-1')
    except AttributeError:
        return pdf.output()
