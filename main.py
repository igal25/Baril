import asyncio
import os
import logging
import tempfile
import re
import csv
from collections import defaultdict

from datetime import datetime
from io import BytesIO

import fitz
from playwright.async_api import async_playwright

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# Suppress PyMuPDF and pdfminer warnings
import logging
logging.getLogger("fitz").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Paths and setup
CSV_FILE = "תקנה_21.csv"
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

                # locate the card to extract the company name from aria-label
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


import re
import csv
import tempfile
import fitz  # PyMuPDF
import logging

CSV_FILE = "תקנה_21.csv"

def reverse_hebrew(text):
    # הופך כל מילה עם תווים עבריים — כדי להתמודד עם כיווניות RTL במילים
    words = text.split()
    return ' '.join([word[::-1] if re.search(r'[\u0590-\u05FF]', word) else word for word in words])

import fitz
import tempfile
import csv
import re
import logging

CSV_FILE = "תקנה_21.csv"

def is_table_row(words):
    """בודק אם השורה נראית כמו שורת טבלה (מספרים/אחוזים/שדות ריקים)"""
    numericish = lambda w: re.match(r'^[-\d,.%()]+$', w.strip())
    count_numeric = sum(1 for w in words if numericish(w))
    return count_numeric >= max(2, len(words) // 2)  # לפחות חצי מהשורה "נראית מספרית"

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

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            if re.search(r"תקנה\s*21\s*[:\-]", page_text):
                takana_page = doc.load_page(page_num)
                break

        if not takana_page:
            logging.info(f"📄 תקנה 21 לא נמצאה עבור {company} ({report_id})")
            return

        logging.info(f"📄 מוצאת טבלה מתוך תקנה 21 עבור {company} ({report_id})")

        words = takana_page.get_text("words")
        rows_by_y = defaultdict(list)

        for x0, y0, x1, y1, word, *_ in words:
            y_key = round(y0 / 5) * 5
            rows_by_y[y_key].append((x0, word.strip()))

        structured_rows = []
        start_found = False
        x_threshold = 7

        for y in sorted(rows_by_y.keys()):
            row_words = sorted(rows_by_y[y], key=lambda t: t[0])
            raw_texts = [w for _, w in row_words]
            texts = merge_split_title_words(raw_texts)
            if not texts:
                continue

            numericish = lambda w: re.match(r'^[-\d,.%()״"׳]+$', w)
            num_numeric = sum(1 for w in texts if numericish(w))
            is_table = num_numeric >= max(2, len(texts) // 2)

            if not start_found:
                if is_table:
                    start_found = True
                else:
                    continue

            if not is_table:
                break

            # חלוקה לשם, תפקיד ונתונים
            name_parts = []
            title_parts = []
            data_parts = []

            for token in reversed(texts):
                if numericish(token) or re.match(r'^[\d]+$', token):
                    data_parts.insert(0, token)
                elif len(title_parts) < 2:
                    title_parts.insert(0, token)
                else:
                    name_parts.insert(0, token)

            full_name = " ".join(list(reversed(name_parts))).strip()
            title = " ".join(list(reversed(title_parts))).strip()

            while len(data_parts) < 14:
                data_parts.insert(0, "")

            structured_rows.append(
                [company, title, full_name] + list(reversed(data_parts))
            )

        if not structured_rows:
            logging.info(f"📄 לא נמצאו שורות טבלה לאחר תקנה 21 עבור {company} ({report_id})")
            return

        headers = ["חברה", "שם", "תפקיד", "היקף משרה", "שיעור החזקה בהון", "שכר",
                   "מענק", "תגמול מבוסס מניות", "דמי ניהול", "החזר הוצאות", "עמלה",
                   "אחר - ניהול דירקטורים", "ריבית", "דמי שכירות", "אחר", "סה\"כ (באלפי ש\"ח)"]

        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(headers)
            writer.writerows(structured_rows)

        logging.info(f"✅ חולצו {len(structured_rows)} שורות תקנה 21 עבור {company} ({report_id})")

    except Exception as e:
        logging.error(f"⚠️ שגיאה ב־process_report_extract עבור {company}: {e}")


def merge_split_title_words(words):
    merged_words = []
    i = 0

    while i < len(words):
        word = words[i]

        # דילוג על מילים שמתחילות בסוגריים
        if word.startswith(")"):
            # נמשיך לדלג עד שנמצא מילה שמסתיימת בסוגריים
            while i < len(words) and not words[i].endswith("("):
                i += 1
            i += 1  # נדלג גם על זו שמסתיימת
            continue

        # איחוד מילים מפוצלות כמו מנכ + " + ל => מנכ"ל
        if (
            i + 2 < len(words)
            and words[i + 1] == '"'
            and re.match(r"^[\u0590-\u05FF]{2,5}$", word)  # מילים בעברית
            and re.match(r"^[\u0590-\u05FF]$", words[i + 2])
        ):
            merged = f'{word}״{words[i + 2]}'
            merged_words.append(merged)
            i += 3
            continue

        # ניקוי גרשיים רגילים אם יש
        word = word.replace('"', '')
        merged_words.append(word)
        i += 1

    return merged_words


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
