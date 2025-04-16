import asyncio
import os
import sqlite3
import logging
from datetime import datetime
from io import BytesIO
from PyPDF2 import PdfReader
from playwright.async_api import async_playwright

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
DB_FILE = "maya_filings.db"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE,
            company TEXT,
            report_title TEXT,
            report_date TEXT,
            local_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def extract_text_and_find_keyword(pdf_data: BytesIO) -> bool:
    reader = PdfReader(pdf_data)
    for page in reader.pages:
        text = page.extract_text()
        if text and "תקנה 21" in text:
            return True
    return False

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

async def fetch_reports_playwright(from_year, to_year, mode="download"):
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

                await page.evaluate("url => window.open(url, '_blank')", full_url)
                report_page = await context.wait_for_event("page")
                await report_page.wait_for_load_state("load")

                if mode == "download":
                    await process_report_download(report_page, report_id)
                elif mode == "extract":
                    await process_report_extract(report_page, report_id)

                await report_page.close()

            try:
                next_button = page.locator('div.panel-footer button:has-text("עבור לעמוד הבא")')
                await next_button.scroll_into_view_if_needed()
                if await next_button.is_disabled():
                    logging.info("Next page button is disabled. Stopping.")
                    break
                await next_button.click()
                await page.wait_for_timeout(3000)
            except Exception as e:
                logging.warning(f"Couldn't click next page: {e}")
                break

        await browser.close()

async def process_report_download(page, report_id):
    try:
        await page.wait_for_selector('a.ma-tooltip[aria-label="הורדת מסמך"]', timeout=15000)

        async with page.expect_download() as download_info:
            await page.locator('a.ma-tooltip[aria-label="הורדת מסמך"]').click()
        download = await download_info.value

        local_file_path = os.path.join(DOWNLOAD_DIR, download.suggested_filename)
        await download.save_as(local_file_path)

        logging.info(f"Downloaded to {local_file_path}")

    except Exception as e:
        logging.error(f"Failed to process report: {e}")

async def process_report_extract(page, report_id):
    try:
        await page.wait_for_selector('a.ma-tooltip[aria-label="הורדת מסמך"]', timeout=15000)

        async with page.expect_download() as download_info:
            await page.locator('a.ma-tooltip[aria-label="הורדת מסמך"]').click()
        download = await download_info.value

        temp_path = await download.path()
        with open(temp_path, "rb") as f:
            pdf_data = BytesIO(f.read())

        found = extract_text_and_find_keyword(pdf_data)
        if found:
            logging.info(f"✔️ Report {report_id} contains 'תקנה 21'")
        else:
            logging.info(f"❌ Report {report_id} does NOT contain 'תקנה 21'")

    except Exception as e:
        logging.error(f"Failed to process report: {e}")

async def main():
    init_db()
    from_year = 2024
    to_year = 2024
    mode = "extract"  # Change to "download" for downloading

    start_time = datetime.now()
    await fetch_reports_playwright(from_year, to_year, mode=mode)
    end_time = datetime.now()

    logging.info(f"All done in {end_time - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
