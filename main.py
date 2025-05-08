import asyncio
import os
import logging
import tempfile
import re
import csv
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import List

import fitz  # PyMuPDF
from playwright.async_api import async_playwright
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

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
                    await process_report_extract(report_page, report_id, company_name)
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

async def process_report_extract2(page, report_id, company):
    try:
        await page.wait_for_selector('a.ma-tooltip[aria-label="הורדת מסמך"]', timeout=15000)
        async with page.expect_download() as download_info:
            await page.locator('a.ma-tooltip[aria-label="הורדת מסמך"]').click()
        download = await download_info.value

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            await download.save_as(tmp_file.name)
            tmp_path = tmp_file.name

        doc = fitz.open(tmp_path)
        takana_page = None

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            if re.search(r"תקנה\s*21\s*[:\-]", page_text):
                takana_page = doc.load_page(page_num)
                break

        if not takana_page:
            logging.info(f"📄 תקנה 21 לא נמצאה עבור {company} ({report_id})")
            return

        words = takana_page.get_text("words")
        rows_by_y = defaultdict(list)

        for x0, y0, x1, y1, word, *_ in words:
            if not word.strip():
                continue
            y_key = round(y0 / 5) * 5
            rows_by_y[y_key].append((x0, y0, word.strip()))

        header_line = None
        header_keywords = [
            "שם", "תפקיד", "היקף", "החזקה", "שכר", "מענק", "מניות", "ניהול",
            "ייעוץ", "עמלה", "אחר", "רכב", "ריבית", "שכירות", "סה"
        ]
        column_bounds = []
        header_y = None

        for y in sorted(rows_by_y.keys()):
            row = rows_by_y[y]
            text_row = [w for _, _, w in row]
            if sum(1 for h in header_keywords if any(h in w for w in text_row)) >= 3:
                header_line = row
                header_y = y
                break

        if not header_line:
            logging.warning(f"לא נמצאה שורת הכותרות עבור {company}")
            return

        for x, _, word in sorted(header_line, key=lambda t: t[0]):
            cleaned = word.strip().replace('\n', '')
            column_bounds.append((cleaned, x))

        columns = []
        for i in range(len(column_bounds)):
            col_name, x0 = column_bounds[i]
            x1 = column_bounds[i + 1][1] if i + 1 < len(column_bounds) else x0 + 100
            columns.append((col_name, x0, x1))

        def assign_column(x):
            for name, start, end in columns:
                if start <= x < end:
                    return name
            return None

        structured_rows = []
        blank_count = 0
        start_found = False

        for y in sorted(rows_by_y.keys()):
            if rows_by_y[y] == header_line:
                start_found = True
                continue
            if not start_found or y <= header_y:
                continue

            row_words = sorted(rows_by_y[y], key=lambda t: t[0])
            if len(row_words) < 2:
                blank_count += 1
                if blank_count >= 5:
                    break  # יצאנו מהטבלה - יותר מדי שורות ריקות ברצף
                continue
            blank_count = 0

            row_data = defaultdict(list)
            for x, y_val, word in row_words:
                col = assign_column(x)
                if col:
                    row_data[col].append(word)

            clean_text = lambda t: re.sub(r'\([^)]*\)', '', t).replace('=', '').strip()
            get_val = lambda key: clean_text(" ".join(row_data[key])) if key in row_data else "-"

            full_name = get_val("שם")
            title = get_val("תפקיד")
            profile1 = get_val("היקף משרה")
            profile2 = get_val("שיעור החזקה בהון") if "שיעור החזקה בהון" in row_data else get_val("שיעור החזקה")
            profile3 = get_val("שיעור החזקה בהצבעה") if "שיעור החזקה בהצבעה" in row_data else "-"

            if full_name == "-" and title == "-":
                continue

            percent_pattern = re.compile(r"(\d{1,3}(\.\d+)?%)")
            title_words = row_data.get("תפקיד", [])
            name_words = row_data.get("שם", [])
            for word in title_words + name_words:
                matches = percent_pattern.findall(word)
                for match in matches:
                    value = match[0].replace('%', '')
                    try:
                        val = float(value)
                        if val <= 100:
                            if profile2 == "-":
                                profile2 = f"{val}%"
                            elif profile3 == "-":
                                profile3 = f"{val}%"
                    except:
                        continue

            service_keywords = ["שכר", "מענק", "מניות", "ניהול", "ייעוץ", "עמלה", "אחר", "רכב"]
            other_keywords = ["ריבית", "שכירות", "אחר"]

            def sum_vals_by_keywords(keywords):
                total = 0
                for k in row_data:
                    if any(kw in k for kw in keywords):
                        for val in row_data[k]:
                            val = val.replace(",", "").replace("(", "").replace(")", "")
                            if val == "-" or not val:
                                continue
                            try:
                                total += float(val)
                            except:
                                continue
                return str(int(total)) if total else "-"

            tagmul_services = sum_vals_by_keywords(service_keywords)
            tagmul_others = sum_vals_by_keywords(other_keywords)
            try:
                total_combined = str(int(tagmul_services) + int(tagmul_others))
            except:
                total_combined = "-"

            row = [
                company,
                full_name,
                title,
                profile1,
                profile2,
                profile3,
                tagmul_services,
                tagmul_others,
                total_combined
            ]
            structured_rows.append(row)

        if not structured_rows:
            logging.info(f"📄 לא נמצאו שורות טבלה לאחר תקנה 21 עבור {company} ({report_id})")
            return

        headers = [
            "חברה", "שם", "תפקיד", "היקף משרה", "שיעור החזקה בהון",
            "שיעור החזקה בהצבעה", "תגמולים (שירותים)", "תגמולים אחרים", "סה\"כ (בש\"ח)"
        ]

        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(headers)
            writer.writerows(structured_rows)

        logging.info(f"✅ חולצו {len(structured_rows)} שורות תקנה 21 עבור {company} ({report_id})")

    except Exception as e:
        logging.error(f"⚠️ שגיאה ב־process_report_extract עבור {company}: {e}")


async def process_report_extract(page, report_id, company):
    try:
        await page.wait_for_selector('a.ma-tooltip[aria-label="הורדת מסמך"]', timeout=15000)
        async with page.expect_download() as download_info:
            await page.locator('a.ma-tooltip[aria-label="הורדת מסמך"]').click()
        download = await download_info.value

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            await download.save_as(tmp_file.name)
            tmp_path = tmp_file.name

        doc = fitz.open(tmp_path)
        takana_page = None

        rows = detect_table_lines_in_pdf(tmp_path)

        headers = ["שם מלא", "תפקיד", "היקף משרה", "שיעור החזקה", "תגמולים עבור שירותים", "תגמולים אחרים", "סה\"כ"]

        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["חברה"] + headers)
            for row in rows:
                writer.writerow([company] + row)

        """for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            if re.search(r"תקנה\s*21\s*[:\-]", page_text):
                takana_page = doc.load_page(page_num)
                break

        if not takana_page:
            logging.info(f"📄 תקנה 21 לא נמצאה עבור {company} ({report_id})")
            return

        words = takana_page.get_text("words")

        expected_headers = [
            ("פרטי מקבלי התגמולים", ["פרטי", "מקבלי", "התגמולים"]),
            ("תגמולים עבור שירותים", ["תגמולים", "עבור", "שירותים"]),
            ("תגמולים אחרים", ["תגמולים", "אחרים"]),
            ("סה", ["סה"])
        ]

        header_matches = {}
        rows_by_y = defaultdict(list)
        for x0, y0, x1, y1, word, *_ in words:
            if not word.strip():
                continue
            y_key = round(y0 / 5) * 5
            rows_by_y[y_key].append((x0, x1, word.strip()))

        for y, row in rows_by_y.items():
            words_in_row = [w for _, _, w in row]
            for header_name, parts in expected_headers:
                if header_name in header_matches:
                    continue
                if all(part in words_in_row for part in parts):
                    xs = [x0 for x0, _, w in row if w in parts]
                    xe = [x1 for _, x1, w in row if w in parts]
                    if xs and xe:
                        header_matches[header_name] = (min(xs), max(xe))

        if len(header_matches) < 4:
            logging.warning(f"❌ לא נמצאו כל ארבעת בלוקי העמודות עבור {company}")
            return

        sorted_headers = sorted(header_matches.items(), key=lambda t: t[1][0])
        blocks = []
        for i in range(len(sorted_headers)):
            name, (x0, _) = sorted_headers[i]
            x1 = sorted_headers[i + 1][1][0] if i + 1 < len(sorted_headers) else x0 + 200
            blocks.append((name, x0, x1))

        def assign_block(x):
            for name, start, end in blocks:
                if start <= x < end:
                    return name
            return "unknown"

        table_data = defaultdict(lambda: defaultdict(list))
        for x0, y0, x1, y1, word, *_ in words:
            if not word.strip():
                continue
            y_key = round(y0 / 5) * 5
            block = assign_block(x0)
            table_data[y_key][block].append((x0, word.strip()))

        def sum_numeric_values(word_list):
            total = 0
            for _, word in word_list:
                word_clean = word.replace(",", "").replace("-", "0").strip()
                try:
                    total += float(word_clean)
                except:
                    continue
            return int(total) if total else "-"

        structured_rows = []
        header_y = max(y for y in rows_by_y if any("פרטי" in w for _, _, w in rows_by_y[y]))
        y_sorted = sorted([y for y in table_data.keys() if y > header_y + 10])

        previous_y = None
        star_counter = 0
        for y in y_sorted:
            row_words = [w for block in table_data[y].values() for _, w in block]
            if any(word.startswith("(*)") for word in row_words):
                star_counter += 1
                if star_counter == 2:
                    break

            row = table_data[y]
            block1_words = sorted(row.get("פרטי מקבלי התגמולים", []), key=lambda t: t[0])
            full_name = " ".join(word for _, word in block1_words).strip() or "-"

            tagmul_services = sum_numeric_values(row.get("תגמולים עבור שירותים", []))
            tagmul_others = sum_numeric_values(row.get("תגמולים אחרים", []))

            try:
                total = tagmul_services + tagmul_others if tagmul_services != "-" and tagmul_others != "-" else "-"
            except:
                total = "-"

            structured_rows.append([
                company,
                full_name,
                tagmul_services,
                tagmul_others,
                total
            ])

        headers = ["חברה", "פרטי מקבל", "תגמולים בעבור שירותים", "תגמולים אחרים", "סה\"כ"]

        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(headers)
            writer.writerows(structured_rows)

        logging.info(f"✅ טבלה חולצה לפי בלוקים ונותחה בהצלחה עבור {company} ({report_id})")
"""
    except Exception as e:
        logging.error(f"⚠️ שגיאה ב־process_report_extract עבור {company}: {e}")


import fitz  # PyMuPDF
import os
import re
import pandas as pd
from collections import defaultdict


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

    # Detect lines
    horizontal_lines = sorted({round(d['rect'].y0) for d in drawings if abs(d['rect'].y1 - d['rect'].y0) < 1})
    vertical_lines = sorted({round(d['rect'].x0) for d in drawings if abs(d['rect'].x1 - d['rect'].x0) < 1})

    if not horizontal_lines or not vertical_lines:
        print("⚠️ No table grid detected.")
        return

    print(f"📐 Grid size: {len(horizontal_lines) - 1} rows x {len(vertical_lines) - 1} columns")

    # Build cell grid
    table = [[[] for _ in range(len(vertical_lines) - 1)] for _ in range(len(horizontal_lines) - 1)]

    for x0, y0, x1, y1, word, *_ in words:
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2

        row_idx = next((i for i in range(len(horizontal_lines) - 1)
                        if horizontal_lines[i] <= y_center < horizontal_lines[i + 1]), None)
        col_idx = next((j for j in range(len(vertical_lines) - 1)
                        if vertical_lines[j] <= x_center < vertical_lines[j + 1]), None)

        if row_idx is not None and col_idx is not None:
            table[row_idx][col_idx].append(word)

    # Convert to DataFrame
    df_rows = []
    for row in table:
        df_rows.append([" ".join(cell).strip() if cell else "" for cell in row])

    raw_df = pd.DataFrame(df_rows)[1:]
    print("\n🧾 Extracted table:")
    print(raw_df.to_string(index=False))

    # Handle special case: only 1 column and many rows with full text
    if raw_df.shape[1] == 1 and raw_df[0].str.contains(r'\d').any():
        def parse_line(line):
            name_match = re.search(r"^([\u0590-\u05FF\s\"\']+?)\s+(?=\d{1,3}%|\d{1,3}(,\d{3})?)", line)
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
        final_df.insert(0, "חברה", os.path.basename(pdf_path).split("_")[0])
        final_df.to_csv(os.path.splitext(pdf_path)[0] + "_cleaned_table.csv", index=False, encoding="utf-8-sig")
        print("\n💾 Fallback cleaned table saved.")
        return final_df.values.tolist()

    # Try to identify the header row based on keywords
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

    service_cols = find_columns_by_keywords(data_df.columns,
                                            ["שכר", "מענק", "מניות", "ניהול", "ייעוץ", "עמלה", "רכב"])
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
    final_df.insert(0, "חברה", os.path.basename(pdf_path).split("_")[0])
    csv_output_path = os.path.splitext(pdf_path)[0] + "_cleaned_table.csv"
    final_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Cleaned table saved to: {csv_output_path}")

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
