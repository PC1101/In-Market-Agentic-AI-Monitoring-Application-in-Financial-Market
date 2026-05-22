"""
generate_report_pdf.py
Generates an executive summary PDF for the XSectional Cross-Sectional Momentum Model.
Run from the XSectional/ directory: python3 generate_report_pdf.py
"""

from fpdf import FPDF
from datetime import date
import os

OUTPUT_PATH = "XSectional_Executive_Summary.pdf"

# ?? Colour palette ????????????????????????????????????????????????????????????
NAVY   = (15,  40,  80)
TEAL   = (0,  112, 130)
LIGHT  = (240, 245, 250)
WHITE  = (255, 255, 255)
DARK   = (30,  30,  30)
GREY   = (100, 100, 100)
LGREY  = (220, 225, 230)


class PDF(FPDF):

    def header(self):
        # Top colour bar
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 14, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        self.set_y(4)
        self.cell(0, 6, "XSectional  |  Cross-Sectional Momentum Model", align="C")
        self.set_text_color(*DARK)
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*NAVY)
        self.rect(0, 285, 210, 12, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "", 7)
        self.set_y(-10)
        self.cell(0, 6,
                  f"Confidential  |  Prepared {date.today().strftime('%d %B %Y')}  |  Page {self.page_no()}",
                  align="C")
        self.set_text_color(*DARK)

    # ?? Section heading ???????????????????????????????????????????????????????
    def section(self, title: str):
        self.ln(4)
        self.set_fill_color(*TEAL)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"  {title}", fill=True, ln=True)
        self.set_text_color(*DARK)
        self.ln(2)

    # ?? Body text ?????????????????????????????????????????????????????????????
    def body(self, text: str, indent: int = 0):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(10 + indent)
        self.multi_cell(190 - indent, 5.5, text)
        self.ln(1)

    # ?? Bullet point ??????????????????????????????????????????????????????????
    def bullet(self, text: str, indent: int = 5):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(10 + indent)
        self.cell(5, 5.5, "*")
        self.set_x(10 + indent + 5)
        self.multi_cell(185 - indent, 5.5, text)

    # ?? Key-value row (for spec table) ????????????????????????????????????????
    def kv_row(self, key: str, value: str, shaded: bool = False):
        if shaded:
            self.set_fill_color(*LIGHT)
        else:
            self.set_fill_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*NAVY)
        self.cell(55, 7, f"  {key}", fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.cell(135, 7, value, fill=True, ln=True)

    # ?? Metric box ???????????????????????????????????????????????????????????
    def metric_box(self, label: str, value: str, x: float, y: float, w: float = 42):
        self.set_fill_color(*NAVY)
        self.rect(x, y, w, 18, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.set_xy(x, y + 1)
        self.cell(w, 9, value, align="C")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*LGREY)
        self.set_xy(x, y + 10)
        self.cell(w, 6, label, align="C")
        self.set_text_color(*DARK)


# ?????????????????????????????????????????????????????????????????????????????
def build_pdf():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ?? TITLE BLOCK ??????????????????????????????????????????????????????????
    pdf.set_fill_color(*LIGHT)
    pdf.rect(10, 18, 190, 36, "F")

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*NAVY)
    pdf.set_xy(14, 21)
    pdf.cell(0, 10, "XSectional Momentum Model")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*TEAL)
    pdf.set_xy(14, 31)
    pdf.cell(0, 7, "Cross-Sectional 12-1 Momentum Strategy  |  S&P 500  |  2000-2025")

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*GREY)
    pdf.set_xy(14, 42)
    pdf.cell(0, 5, f"Executive Summary  |  {date.today().strftime('%d %B %Y')}")
    pdf.set_text_color(*DARK)
    pdf.set_y(60)

    # ?? 1. EXECUTIVE SUMMARY ?????????????????????????????????????????????????
    pdf.section("1.  Executive Summary")
    pdf.body(
        "XSectional is a quantitative backtesting module that implements the classic Jegadeesh & Titman "
        "(1993) cross-sectional momentum strategy on the S&P 500 universe over a 25-year period "
        "(January 2000 - December 2025). The system ranks all 500 stocks monthly by their prior 12-month "
        "return (skipping the most recent month to avoid short-term reversal), goes long the top 20% "
        "of performers and short the bottom 20%, and rebalances at every month-end. It produces a full "
        "performance tearsheet including equity curve, annual returns, rolling Sharpe ratio, and drawdown analysis."
    )

    # ?? 2. STRATEGY SPECIFICATION ????????????????????????????????????????????
    pdf.section("2.  Strategy Specification")
    rows = [
        ("Universe",           "S&P 500 constituents (current composition via Wikipedia)"),
        ("Backtest Period",     "January 2000 - December 2025  (25 years, 300 monthly periods)"),
        ("Momentum Signal",    "12-1 momentum: cumulative return over t-13 to t-2 (12 months, skip t-1)"),
        ("Long Leg",           "Top 20% of stocks by momentum score  - equally weighted (+1/n each)"),
        ("Short Leg",          "Bottom 20% of stocks by momentum score - equally weighted (-1/n each)"),
        ("Rebalancing",        "Monthly, at month-end close prices"),
        ("Data Source",        "Yahoo Finance via yfinance (adjusted close prices, auto-adjusted)"),
        ("Missing Data",       "Tickers with >20% missing history dropped; min 10 stocks per leg enforced"),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.kv_row(k, v, shaded=(i % 2 == 0))
    pdf.ln(3)

    # ?? 3. TECHNICAL ARCHITECTURE ????????????????????????????????????????????
    pdf.section("3.  Technical Architecture")
    pdf.body("The system is built as a modular Python pipeline. Each module has a single responsibility "
             "and communicates through well-defined DataFrame interfaces:")

    modules = [
        ("config.py",    "Central configuration - all tunable parameters in one place (dates, lookback, quantiles, paths)"),
        ("data.py",      "Downloads and caches S&P 500 adjusted close prices from Yahoo Finance; persists to CSV"),
        ("signals.py",   "Computes monthly returns and 12-1 momentum scores using log-return rolling sums for numerical stability"),
        ("portfolio.py", "Constructs equal-weighted long-short monthly weights from momentum rankings"),
        ("backtest.py",  "Simulates month-by-month portfolio returns; applies weights at t to returns at t+1"),
        ("report.py",    "Computes annualised metrics and renders a 4-panel tearsheet PNG"),
        ("main.py",      "Orchestrates the full pipeline end-to-end with structured logging"),
    ]
    for mod, desc in modules:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*NAVY)
        pdf.set_x(15)
        pdf.cell(32, 5.5, mod)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*DARK)
        pdf.multi_cell(153, 5.5, desc)

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*GREY)
    pdf.body("Data flow:  yfinance  ->  data.py  ->  signals.py  ->  portfolio.py  ->  backtest.py  ->  report.py", indent=5)
    pdf.set_text_color(*DARK)

    # ?? 4. PERFORMANCE METRICS ???????????????????????????????????????????????
    pdf.section("4.  Performance Metrics Reported")
    pdf.body("The tearsheet computes and reports the following risk-adjusted metrics "
             "(risk-free rate assumed = 0% throughout):")

    metrics = [
        ("Annualised Return",     "Geometric mean annualised return over the full backtest period"),
        ("Annualised Volatility", "Standard deviation of monthly returns scaled to annual frequency (x sqrt12)"),
        ("Sharpe Ratio",          "Annualised return divided by annualised volatility"),
        ("Maximum Drawdown",      "Largest peak-to-trough decline in the cumulative equity curve"),
        ("Calmar Ratio",          "Annualised return divided by the absolute maximum drawdown"),
    ]
    for i, (m, d) in enumerate(metrics):
        pdf.kv_row(m, d, shaded=(i % 2 == 0))
    pdf.ln(3)

    pdf.body("The tearsheet PNG contains four panels:")
    for item in [
        "Cumulative equity curve (log scale)",
        "Year-by-year annual return bar chart (blue = positive, red = negative)",
        "Rolling 12-month Sharpe ratio",
        "Drawdown chart (shaded area below zero)",
    ]:
        pdf.bullet(item)

    # ?? 5. ACADEMIC FOUNDATION ???????????????????????????????????????????????
    pdf.section("5.  Academic Foundation")
    pdf.body(
        "This model replicates the core methodology of Jegadeesh & Titman (1993), "
        "the seminal paper establishing that stocks with strong prior returns continue to outperform "
        "over intermediate horizons (3-12 months). The one-month skip period is the standard adjustment "
        "to avoid the well-documented short-term reversal effect. The long-short portfolio structure "
        "isolates the pure momentum factor return, net of market exposure."
    )
    pdf.bullet("Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers: "
               "Implications for Stock Market Efficiency. Journal of Finance, 48(1), 65-91.")
    pdf.bullet("Reference implementation: Fisjo/momentum-strategy-backtest (GitHub)")

    # ?? 6. TECH STACK ????????????????????????????????????????????????????????
    pdf.section("6.  Technology Stack")
    stack = [
        ("Language",       "Python 3.9+"),
        ("Data",           "yfinance, pandas, lxml, requests"),
        ("Computation",    "NumPy (log-return rolling sums for momentum)"),
        ("Visualisation",  "Matplotlib (Agg backend - headless safe)"),
        ("Testing",        "pytest  |  37 unit tests  |  100% pass rate"),
        ("Version Control","Git / GitHub"),
    ]
    for i, (k, v) in enumerate(stack):
        pdf.kv_row(k, v, shaded=(i % 2 == 0))
    pdf.ln(3)

    # ?? 7. KNOWN LIMITATIONS ?????????????????????????????????????????????????
    pdf.section("7.  Known Limitations  &  Future Work")
    pdf.body("The current prototype is designed for research and proof-of-concept. The following "
             "limitations should be disclosed when interpreting backtest results:")

    limitations = [
        "Survivorship bias - yfinance only returns currently listed tickers; delisted stocks are excluded, "
        "which overstates historical returns.",
        "Static universe - uses current S&P 500 composition rather than point-in-time historical membership.",
        "No transaction costs - bid-ask spread, commissions, and market impact are not modelled.",
        "No slippage - assumes execution at exact month-end closing prices.",
    ]
    for lim in limitations:
        pdf.bullet(lim)

    pdf.ln(2)
    pdf.body("Planned enhancements:")
    future = [
        "Integrate survivorship-bias-free data (e.g., CRSP or Compustat via WRDS)",
        "Add point-in-time index composition using historical S&P 500 membership data",
        "Model transaction costs and realistic execution assumptions",
        "Extend to multi-factor model (value + momentum + quality)",
        "Live monitoring dashboard - agent-based alerting on momentum signal shifts",
    ]
    for f in future:
        pdf.bullet(f)

    # ?? 8. HOW TO RUN ????????????????????????????????????????????????????????
    pdf.section("8.  How to Run")
    pdf.body("Prerequisites: Python 3.9+, dependencies installed via pip install -r requirements.txt")
    pdf.ln(1)

    pdf.set_fill_color(30, 30, 40)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Courier", "", 8.5)
    code_lines = [
        "# Navigate to module directory",
        "cd XSectional/",
        "",
        "# Run full pipeline (downloads data on first run, ~2-5 min)",
        "python3 main.py",
        "",
        "# Run unit test suite (37 tests, ~5 seconds)",
        "python3 -m pytest -v",
    ]
    for line in code_lines:
        pdf.set_x(10)
        pdf.cell(190, 5, f"  {line}", fill=True, ln=True)
    pdf.set_text_color(*DARK)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.body("Output: Performance tearsheet saved to XSectional/data/tearsheet.png")

    # ?? SAVE ?????????????????????????????????????????????????????????????????
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved -> {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    build_pdf()
