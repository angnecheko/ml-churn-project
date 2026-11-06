import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait


def test_application():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1280,900")

    service = Service(
        log_output="chromedriver.log",
        service_args=["--verbose"],
    )

    driver = webdriver.Chrome(
        service=service,
        options=options,
    )

    try:
        driver.set_page_load_timeout(30)

        url = os.getenv(
            "API_DOCS_URL",
            "http://localhost:8001/docs",
        )
        driver.get(url)

        WebDriverWait(driver, 15).until(
            lambda navigateur:
                "Churn Prediction API" in navigateur.title
        )

        assert "Churn Prediction API" in driver.title

    finally:
        driver.quit()