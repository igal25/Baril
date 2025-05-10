# Baril
# Extracting Structured Data from Israeli Financial Filings (תקנה 21)

This project automates the extraction and parsing of compensation data from periodic financial reports filed by public companies on the [Maya TASE website](https://maya.tase.co.il). It focuses specifically on **Regulation 21 ("תקנה 21")** tables, which include detailed compensation disclosures for senior executives and related parties.

---

## 📌 Features

- 🔍 Detects and downloads relevant PDF reports
- 📄 Locates Regulation 21 pages using text search
- 📐 Reconstructs tables using PDF grid lines (if available)
- 🧠 Dynamically identifies key columns: names, roles, holdings, compensation
- 📊 Aggregates and cleans numerical data (e.g., salaries, bonuses, other rewards)
- 📁 Outputs a clean UTF-8 CSV with company name and year
- 📎 Handles layout variations across companies and filings

---

## 🛠 Technologies Used

- Python 3.10+
- [Playwright (async)](https://playwright.dev/python/)
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`)
- `pandas`
- `re`, `csv`, `tempfile`, `collections`

---

## 🧪 Example Output
A typical row in the final CSV file looks like:

| שנה | חברה | שם מלא | תפקיד | היקף משרה | שיעור החזקה | תגמולים עבור שירותים | תגמולים אחרים | סה"כ |

| 2024 | בוני תיכון | עמרם פרץ | מנכ"ל | 100% | 47.3% | 1740 | 0 | 1740 |

---

## 🚀 Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python main.py
```

3. Output CSV: `תקנה_21.csv` (with year and company columns)

---

## 📂 File Structure
```
.
├── main.py                  # Entry point for running scraping and extraction
├── detect_table_lines.py    # Core PDF table reconstruction logic
├── תקנה_21.csv              # Output data (generated)
└── requirements.txt         # Python package dependencies
```

---

## 📝 Notes
- Some PDFs may not include תקנה 21 or have highly inconsistent layouts; those are skipped with logs.
- Two methods are used to identify tables: grid line detection (preferred) and word grouping heuristics (fallback).

---

## 📬 Contact
For questions or collaboration, feel free to reach out via GitHub Issues or email.

---

## 🧠 Acknowledgments
This project was built for academic and research purposes, enabling easier access to structured compensation data in Israel's public market.

---
