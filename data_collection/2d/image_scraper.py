import pathlib

import requests
from fake_http_header import FakeHttpHeader
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

DOWNLOAD_PATH = pathlib.Path(__file__).parent / "google_dataset"

SCRAPE_URL = "https://images.google.com"
THUMBNAIL_CLASS = "YQ4gaf"
IMAGE_CLASS = "sFlh5c.FyHeAf.iPVvYb"

SEARCH_TERM = "weapons"
NUM_IMAGES = 1000


def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.maximize_window()
    driver.implicitly_wait(1)
    return driver


def locate_images(driver, num_images):
    driver.get(SCRAPE_URL)

    # search for keyword
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(SEARCH_TERM)
    search_box.send_keys(Keys.RETURN)

    image_urls = []
    # go until enough images are found
    while len(image_urls) < num_images:
        thumbnails = driver.find_elements(
            By.XPATH, f"//img[@class='{THUMBNAIL_CLASS}']"
        )
        if not thumbnails:
            print("No more images found")
            break
        print(f"Found {len(thumbnails)} thumbnails")

        for thumbnail in thumbnails[len(image_urls) :]:
            try:
                # to get high res img, you have to click on thumbnail, then it shows
                thumbnail.click()

                high_res_image = driver.find_element(
                    By.CSS_SELECTOR, f"img.{IMAGE_CLASS}"
                )
                img_url = high_res_image.get_attribute("src")
                if img_url and img_url.startswith("http"):
                    image_urls.append(img_url)
                    print(f"Image {len(image_urls)}: {img_url}")
            except NoSuchElementException:
                print(f"Couldn't find image with {IMAGE_CLASS}")
            except ElementClickInterceptedException as e:
                # when clicking on thumbnail
                print(e.msg)
            except Exception as e:
                print(e)

            if len(image_urls) >= num_images:
                break

        # scroll to bottom of window to find more thumbnails
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    return image_urls


def download_image(img_url, file_name):
    if img_url and img_url.startswith("http"):
        if not DOWNLOAD_PATH.is_dir():
            DOWNLOAD_PATH.mkdir()

        # this gets around some sites anti scraper detection
        # fake_header = FakeHttpHeader().as_header_dict()
        # response = requests.get(img_url, headers=fake_header, stream=True)
        # response.raise_for_status()

        # with open(DOWNLOAD_PATH / file_name, "wb") as file:
        #     for chunk in response.iter_content(1024):
        #         file.write(chunk)
    else:
        print("Invalid image URL")


def download_images():
    driver = setup_driver()
    image_urls = locate_images(driver, NUM_IMAGES)
    for i, img_url in enumerate(image_urls):
        file_name = f"{SEARCH_TERM}_{i + 1}.jpg"
        download_image(img_url, file_name)

    driver.quit()


if __name__ == "__main__":
    download_images()
