import asyncio
import os
import logging
import tempfile
import csv
from datetime import datetime
from typing import List
from playwright.async_api import async_playwright
import fitz
import os
import re
import pandas as pd

# Suppress PyMuPDF and pdfminer warnings
logging.getLogger("fitz").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Paths and setup
CSV_FILE = "תקנה_21.csv"
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_cell(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value)
    value = value.replace('"', '').replace("=", "").strip()
    return value

def merge_split_title_words(words):
    merged_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith(")"):
            while i < len(words) and not words[i].endswith("("):
                i += 1
            i += 1
            continue
        if (
            i + 2 < len(words)
            and words[i + 1] == '"'
            and re.match(r"^[\u0590-\u05FF]{2,5}$", word)
            and re.match(r"^[\u0590-\u05FF]$", words[i + 2])
        ):
            merged = f'{word}״{words[i + 2]}'
            merged_words.append(merged)
            i += 3
            continue
        word = word.replace('"', '')
        merged_words.append(word)
        i += 1
    return merged_words

def clean_token(token: str) -> str:
    token = re.sub(r"\([^)]*\)", "", token)
    token = token.replace('"', "").replace("=", "")
    return token.strip()

def summarize_amounts(values: List[str]) -> str:
    cleaned = [clean_token(v) for v in values]
    numeric_values = [v for v in cleaned if v and v != "-" and re.match(r'^[-\d,.]+$', v)]

    if not numeric_values:
        return "-"

    try:
        total = sum(float(v.replace(",", "")) for v in numeric_values)
        return str(int(total)) if total.is_integer() else f"{total:.2f}"
    except Exception:
        return "-"

def reverse_hebrew(text):
    words = text.split()
    return ' '.join([word[::-1] if re.search(r'[\u0590-\u05FF]', word) else word for word in words])

def is_table_row(words):
    numericish = lambda w: re.match(r'^[-\d,.%()]+$', w.strip())
    count_numeric = sum(1 for w in words if numericish(w))
    return count_numeric >= max(2, len(words) // 2)

async def set_date(page, selector, date_str):
    await page.wait_for_selector(selector)
    await page.evaluate("""
    ([selector, dateStr]) => {
        const input = document.querySelector(selector);
        input.value = dateStr;
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    """, [selector, date_str])
    logging.info(f"Set date {date_str} in {selector}")

async def fetch_reports_playwright(from_year, to_year, mode="extract"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        logging.info("Navigating to Maya TASE Companies Reports page...")
        await page.goto("https://maya.tase.co.il/he/reports/companies", timeout=60000)

        await set_date(page, 'input[aria-label="searchForm.fromDatePlaceholder"]', f'01.01.{from_year}')
        await set_date(page, 'input[aria-label="searchForm.toDatePlaceholder"]', f'31.12.{to_year}')

        await page.locator('span:has-text("דוחות כספיים")').locator('xpath=..').locator('button.button-expand').click()
        await asyncio.sleep(1)
        await page.get_by_role("treeitem", name="דוח תקופתי ושנתי", exact=True).locator('input[type="checkbox"]').click()
        await asyncio.sleep(1)
        await page.locator('input[placeholder="טקסט חופשי"]').fill("דוח תקופתי")
        await asyncio.sleep(1)
        await page.locator('button.panel-filter-bt', has_text="סינון").first.click()
        logging.info("Waiting for reports list to load...")

        for _ in range(30):
            if await page.locator('a[aria-label*="PDF"]').count() > 0:
                break
            await page.wait_for_timeout(1000)
        else:
            raise Exception("PDF icons not found in time")

        logging.info("Reports list loaded.")

        while True:
            pdf_links = await page.locator('a[aria-label*="PDF"]:not([aria-label*="מתקן"])').all()
            logging.info(f"Found {len(pdf_links)} PDF links on this page.")

            for link in pdf_links:
                href = await link.get_attribute("href")
                if not href:
                    continue

                report_id = href.split("/")[-1].split("?")[0]
                full_url = f"https://maya.tase.co.il{href}"

                try:
                    card_locator = link.locator("xpath=ancestor::maya-report-card")
                    company_name = await card_locator.locator("h3 a").get_attribute("aria-label")
                except Exception:
                    company_name = "לא ידוע"

                await page.evaluate("url => window.open(url, '_blank')", full_url)
                report_page = await context.wait_for_event("page")
                await report_page.wait_for_load_state("load")

                if mode == "extract":
                    await process_report_extract(report_page, report_id, company_name, from_year)
                await report_page.close()

            try:
                next_button = page.locator('button[aria-label="עבור לעמוד הבא"]')
                await next_button.scroll_into_view_if_needed()
                await page.wait_for_timeout(1000)
                if await next_button.is_disabled():
                    logging.info("Next page button is disabled. Stopping.")
                    break
                await next_button.click(force=True)
                await page.wait_for_timeout(3000)
            except Exception as e:
                logging.warning(f"Couldn't click next page: {e}")
                break

        await browser.close()


async def process_report_extract(page, report_id, company, year):
    try:
        await page.wait_for_selector('a.ma-tooltip[aria-label="הורדת מסמך"]', timeout=15000)
        async with page.expect_download() as download_info:
            await page.locator('a.ma-tooltip[aria-label="הורדת מסמך"]').click()
        download = await download_info.value

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            await download.save_as(tmp_file.name)
            tmp_path = tmp_file.name

        rows = detect_table_lines_in_pdf(tmp_path)

        headers = ["שם מלא", "תפקיד", "היקף משרה", "שיעור החזקה", "תגמולים עבור שירותים", "תגמולים אחרים", "סה\"כ"]

        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["שנה", "חברה"] + headers)
            for row in rows:
                writer.writerow([year, company] + row)

    except Exception as e:
        logging.error(f"⚠️ שגיאה ב־process_report_extract עבור {company}: {e}")


def detect_table_lines_in_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Analyzing file: {os.path.basename(pdf_path)}")

    takana_page = None
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if re.search(r"תקנה\s*21\s*[:\-]", text):
            takana_page = doc.load_page(page_num)
            break

    if takana_page is None:
        print("⚠️ תקנה 21 לא נמצאה במסמך.")
        return

    drawings = takana_page.get_drawings()
    words = takana_page.get_text("words")

    horizontal_lines = sorted({round(d['rect'].y0) for d in drawings if abs(d['rect'].y1 - d['rect'].y0) < 1})
    vertical_lines = sorted({round(d['rect'].x0) for d in drawings if abs(d['rect'].x1 - d['rect'].x0) < 1})

    if not horizontal_lines or not vertical_lines:
        print("⚠️ No table grid detected.")
        return

    print(f"📐 Grid size: {len(horizontal_lines) - 1} rows x {len(vertical_lines) - 1} columns")

    table = [[[] for _ in range(len(vertical_lines) - 1)] for _ in range(len(horizontal_lines) - 1)]

    for x0, y0, x1, y1, word, *_ in words:
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2
        row_idx = next((i for i in range(len(horizontal_lines) - 1) if horizontal_lines[i] <= y_center < horizontal_lines[i + 1]), None)
        col_idx = next((j for j in range(len(vertical_lines) - 1) if vertical_lines[j] <= x_center < vertical_lines[j + 1]), None)
        if row_idx is not None and col_idx is not None:
            table[row_idx][col_idx].append(word)

    df_rows = [[" ".join(cell).strip() if cell else "" for cell in row] for row in table]
    raw_df = pd.DataFrame(df_rows)[1:]
    print("\n🧾 Extracted table:")
    print(raw_df.to_string(index=False))

    if raw_df.shape[1] == 1 and raw_df[0].str.contains(r'\d').any():
        def parse_line(line):
            name_match = re.search(r"^([\u0590-\u05FF\s\"']+?)\s+(?=\d{1,3}%|\d{1,3}(,\d{3})?)", line)
            name = name_match.group(1).strip() if name_match else "-"
            numbers = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", line)
            values = [n.replace(",", "") for n in numbers if "%" not in n][:3]
            services = [n.replace(",", "") for n in numbers if "%" not in n][3:-1]
            others = [n.replace(",", "") for n in numbers if "%" not in n][-1:]
            sum_services = sum(float(v) for v in services) if services else 0
            sum_others = sum(float(v) for v in others) if others else 0
            return [name, str(sum_services), str(sum_others), str(sum_services + sum_others)]

        parsed = raw_df[0].apply(parse_line).tolist()
        final_df = pd.DataFrame(parsed, columns=["שם", "תגמולים עבור שירותים", "תגמולים אחרים", "סה\"כ"])
    else:
        header_keywords = ["פרטי", "תגמולים", "סה"]
        max_matches = 0
        header_row_idx = None
        for i, row in raw_df.iterrows():
            match_count = sum(1 for cell in row if any(kw in str(cell) for kw in header_keywords))
            if match_count > max_matches:
                max_matches = match_count
                header_row_idx = i

        if header_row_idx is None:
            print("⚠️ לא נמצאה שורת כותרת.")
            return

        header_row = raw_df.iloc[header_row_idx]
        data_df = raw_df.iloc[header_row_idx + 1:].reset_index(drop=True)
        data_df.columns = header_row[:data_df.shape[1]].tolist()
        print("📌 Columns detected:", list(data_df.columns))

        def find_columns_by_keywords(columns, keywords, exact=False):
            if exact:
                return [col for col in columns if col.strip() in keywords]
            else:
                return [col for col in columns if any(kw in col for kw in keywords)]

        service_cols = find_columns_by_keywords(data_df.columns, ["שכר", "מענק", "מניות", "ניהול", "ייעוץ", "עמלה", "רכב"])
        other_cols = find_columns_by_keywords(data_df.columns, ["ריבית", "שכירות"])
        other_cols.extend(find_columns_by_keywords(data_df.columns, ["אחר"], exact=True))

        def is_number(s):
            return re.match(r'^-?\d+(\.\d+)?$', s.replace(",", "").strip())

        data_df["תגמולים עבור שירותים"] = data_df[service_cols].apply(
            lambda row: sum(float(cell.replace(",", "").replace("-", "0") or 0) for cell in row if is_number(cell)), axis=1)

        data_df["תגמולים אחרים"] = data_df[other_cols].apply(
            lambda row: sum(float(cell.replace(",", "").replace("-", "0") or 0) for cell in row if is_number(cell)), axis=1)

        data_df["סה\"כ"] = data_df["תגמולים עבור שירותים"] + data_df["תגמולים אחרים"]

        required_cols = ["שם", "תפקיד", "היקף משרה", "שיעור"]
        for col in required_cols:
            if col not in data_df.columns:
                found = False
                for existing_col in data_df.columns:
                    if col in existing_col:
                        data_df[col] = data_df[existing_col]
                        found = True
                        break
                if not found:
                    data_df[col] = ""

        final_df = data_df[required_cols + ["תגמולים עבור שירותים", "תגמולים אחרים", "סה\"כ"]]
        final_df = final_df[final_df["שם"].astype(str).str.strip() != ""]
        return final_df.values.tolist()


async def main():
    from_year = 2024
    to_year = 2024
    mode = "extract"

    start_time = datetime.now()
    await fetch_reports_playwright(from_year, to_year, mode=mode)
    end_time = datetime.now()

    logging.info(f"All done in {end_time - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
