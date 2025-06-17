from playwright.sync_api import sync_playwright, ViewportSize
import random
from pathlib import Path


def login_to_website(url, username, password):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                # "--window-size=800,600",
                # "--window-position=0,0",
            ],
        )
        # Create context with realistic user agent and viewport
        viewport = ViewportSize(width=1920, height=1080)

        # Create context with realistic user agent and viewport
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport=viewport,
            accept_downloads=True,
        )

        page = context.new_page()

        # Remove webdriver property
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        try:
            # Navigate to login page
            page.goto(url)

            # Random human-like delays
            page.wait_for_timeout(random.randint(2000, 4000))

            # Simulate mouse movements
            # page.mouse.move(random.randint(100, 200), random.randint(100, 200))
            # page.wait_for_timeout(random.randint(500, 1500))

            page.fill('input[name="user_id"]', username)
            page.wait_for_timeout(1000)
            page.fill('input[name="customer_password"]', password)
            page.wait_for_timeout(1000)

            # Click submit button
            page.click('input[name="continue_button"]')

            # Wait for specific h1 text
            page.wait_for_selector('h1:has-text("Account Services")')

            print("Login successful!")

            # ===== Download files here =====

            base_download_url = "https://www.equibase.com/premium/eqpTMResultChartDownload.cfm?tid=82300044&seq="
            start_id = 1
            end_id = 27

            for file_id in range(start_id, end_id + 1):
                try:
                    url = f"{base_download_url}{file_id}"
                    page.goto(url)

                    # Wait for page to load
                    # page.wait_for_timeout(random.randint(500, 750))

                    with page.expect_download() as download_info:
                        page.click("a:has(div:text('XML Format'))")

                    download = download_info.value
                    native_downloads_dir = Path.home() / "Downloads" / download.suggested_filename

                    download.save_as(native_downloads_dir)
                    print(f"Downloaded: {download.suggested_filename}")

                    # Delay between downloads
                    # page.wait_for_timeout(random.randint(500, 1000))

                except Exception as e:
                    print(f"Failed to download ID {file_id}: {e}")
                    continue

            print(f"Finished downloading files!")

        except Exception as e:
            print(f"Login failed: {e}")
        finally:
            browser.close()


if __name__ == "__main__":
    login_to_website(
        "https://www.equibase.com/premium/eebCustomerLogon.cfm?TMP=customeradminmain.cfm&QS=logon%3DY",
        "hikenny@me.com",
        "muffinsoda",
    )
