import asyncio
import os
import sqlite3
import logging
from datetime import datetime
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

def save_to_db(report_id, company, title, date, local_file):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO filings (report_id, company, report_title, report_date, local_file)
            VALUES (?, ?, ?, ?, ?)
        ''', (report_id, company, title, date, local_file))
        conn.commit()
    except sqlite3.IntegrityError:
        logging.info(f"Skipping duplicate report ID {report_id}")
    finally:
        conn.close()

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

async def fetch_reports_playwright(from_year, to_year):
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

        await page.locator('input[placeholder="טקסט חופשי"]').fill("דוח תקופתי ושנתי")
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

                await process_report(report_page, report_id)
                await report_page.close()

            try:
                next_button = page.locator('button[aria-label="עבור לעמוד הבא"]')

                await page.wait_for_timeout(1000)  # small pause

                if await next_button.count() == 0:
                    logging.info("Next page button not found. Stopping.")
                    break

                is_disabled = await next_button.get_attribute("disabled")
                if is_disabled is not None:
                    logging.info("Next page button is disabled. Stopping.")
                    break

                await next_button.click(force=True)
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(1)

            except Exception as e:
                logging.warning(f"Couldn't click next page: {e}")
                break
            except Exception as e:
                logging.warning(f"Couldn't click next page: {e}")
                break
            except Exception as e:
                logging.warning(f"Couldn't click next page: {e}")
                break

        await browser.close()

async def process_report(page, report_id):
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

async def main():
    init_db()

    from_year = 2024
    to_year = 2024

    start_time = datetime.now()
    await fetch_reports_playwright(from_year, to_year)
    end_time = datetime.now()

    logging.info(f"All done in {end_time - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
