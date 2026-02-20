from fpdf import FPDF


def _header(pdf, cols, widths):
    pdf.set_fill_color(15, 25, 55)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 9)
    for col, w in zip(cols, widths):
        pdf.cell(w, 7, col, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_text_color(0, 0, 0)


def _row(pdf, values, widths, bold=False, shade=False):
    if shade:
        pdf.set_fill_color(240, 243, 250)
    pdf.set_font("Arial", 'B' if bold else '', 9)
    for val, w in zip(values, widths):
        pdf.cell(w, 6, str(val), border=1, fill=shade, align='C')
    pdf.ln()


def _section(pdf, title):
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(15, 25, 55)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_text_color(0, 0, 0)


def create_pdf_report(address, purchase_price, init_eq, lev_irr, dscr, breakeven_occ, memo_text,
                      unlev_irr=None, moic=None, total_cost=None, const_ltv=None,
                      scenario_results=None, sensitivity_data=None):
    pdf = FPDF()
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # ── Title bar ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(15, 25, 55)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 13, "Investment Committee Memorandum", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"Property: {address}", ln=True)
    pdf.ln(4)

    # ── Key Return Metrics ────────────────────────────────────────────────────────
    _section(pdf, "Key Return Metrics")
    metrics = [
        ("Purchase Price",      f"USD {purchase_price:,.0f}"),
        ("Equity Required",     f"USD {init_eq:,.0f}"),
        ("Levered IRR",         f"{lev_irr:.2%}"   if lev_irr   > -1 else "Loss"),
        ("Unlevered IRR",       f"{unlev_irr:.2%}" if unlev_irr is not None and unlev_irr > -1 else "N/A"),
        ("MOIC",                f"{moic:.2f}x"     if moic      is not None else "N/A"),
        ("Year 1 DSCR",         f"{dscr:.2f}x"),
        ("Breakeven Occupancy", f"{breakeven_occ:.1%}"),
    ]
    pdf.set_font("Arial", '', 10)
    for i, (label, val) in enumerate(metrics):
        shade = (i % 2 == 0)
        if shade:
            pdf.set_fill_color(240, 243, 250)
        pdf.cell(95, 7, label, border=1, fill=shade)
        pdf.cell(95, 7, val,   border=1, fill=shade, ln=True)
    pdf.ln(5)

    # ── Capital Stack ─────────────────────────────────────────────────────────────
    if total_cost is not None and const_ltv is not None:
        _section(pdf, "Capital Stack")
        const_debt = total_cost * const_ltv
        _header(pdf, ["Component", "Amount (USD)", "% of Cost"], [72, 68, 50])
        cap_rows = [
            ("Equity",            init_eq,    init_eq / total_cost),
            ("Construction Debt", const_debt, const_debt / total_cost),
        ]
        for i, (name, amt, pct) in enumerate(cap_rows):
            _row(pdf, [name, f"{amt:,.0f}", f"{pct:.1%}"], [72, 68, 50], shade=(i % 2 == 0))
        _row(pdf, ["Total Project Cost", f"{total_cost:,.0f}", "100.0%"], [72, 68, 50], bold=True)
        pdf.ln(5)

    # ── Bear / Base / Bull ────────────────────────────────────────────────────────
    if scenario_results:
        _section(pdf, "Bear / Base / Bull Scenario Analysis")
        names = list(scenario_results.keys())
        metric_keys = list(next(iter(scenario_results.values())).keys())
        n = len(names)
        lw = 48
        cw = int((190 - lw) / n)
        _header(pdf, ["Metric"] + names, [lw] + [cw] * n)
        for i, key in enumerate(metric_keys):
            vals = [key] + [scenario_results[s].get(key, "-") for s in names]
            _row(pdf, vals, [lw] + [cw] * n, shade=(i % 2 == 0))
        pdf.ln(5)

    # ── Sensitivity Matrix ────────────────────────────────────────────────────────
    if sensitivity_data:
        _section(pdf, "Levered IRR Sensitivity: Price vs. Exit Cap Rate")
        caps   = sensitivity_data["caps"]
        prices = sensitivity_data["prices"]
        matrix = sensitivity_data["matrix"]
        n  = len(caps)
        lw = 36
        cw = int((190 - lw) / n)
        _header(pdf, ["Price \\ Cap"] + [f"{c:.2%}" for c in caps], [lw] + [cw] * n)
        for i, (p, row) in enumerate(zip(prices, matrix)):
            _row(pdf, [f"${p:,.0f}"] + row, [lw] + [cw] * n, shade=(i % 2 == 0))
        pdf.ln(5)

    # ── Memo ──────────────────────────────────────────────────────────────────────
    if memo_text:
        _section(pdf, "Sponsor Rationale & Business Plan")
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, memo_text)

    try:
        return pdf.output(dest='S').encode('latin-1')
    except AttributeError:
        return pdf.output()
