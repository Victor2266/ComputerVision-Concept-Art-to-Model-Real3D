import pathlib
import requests
from PIL import Image
import io
import time
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# Configuration
BASE_PATH = pathlib.Path(__file__).parent
DOWNLOAD_PATH = BASE_PATH / "fruit_drawings_dataset"
SCRAPE_URL = "https://images.google.com"
THUMBNAIL_CLASS = "YQ4gaf"
IMAGE_CLASS = "sFlh5c.FyHeAf.iPVvYb"

# Define fruit list and parameters
FRUITS = [
    "apple", "orange", "banana", "pear", "pineapple",
    "strawberry", "grape", "lemon", "mango", "watermelon"
]
IMAGES_PER_FRUIT = 50

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument("--headless")  # Uncomment for headless mode
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), 
        options=options
    )
    driver.maximize_window()
    driver.implicitly_wait(1)
    return driver

def locate_images(driver, search_term, num_images):
    driver.get(SCRAPE_URL)

    # search for keyword
    search_box = driver.find_element(By.NAME, "q")
    search_term_with_params = f"{search_term} drawing white background simple clean"
    search_box.send_keys(search_term_with_params)
    search_box.send_keys(Keys.RETURN)
    time.sleep(2)  # Wait for results to load

    image_urls = []
    retries = 0
    max_retries = 5

    while len(image_urls) < num_images and retries < max_retries:
        thumbnails = driver.find_elements(
            By.XPATH, f"//img[@class='{THUMBNAIL_CLASS}']"
        )
        if not thumbnails:
            print("No more images found")
            retries += 1
            continue

        print(f"Found {len(thumbnails)} thumbnails")

        for thumbnail in thumbnails[len(image_urls):]:
            try:
                thumbnail.click()
                time.sleep(0.5)  # Wait for high-res image to load

                high_res_image = driver.find_element(
                    By.CSS_SELECTOR, f"img.{IMAGE_CLASS}"
                )
                img_url = high_res_image.get_attribute("src")
                if img_url and img_url.startswith("http"):
                    image_urls.append(img_url)
                    print(f"Image {len(image_urls)}: {img_url}")

            except (NoSuchElementException, ElementClickInterceptedException):
                continue
            except Exception as e:
                print(f"Error: {e}")

            if len(image_urls) >= num_images:
                break

        # scroll to bottom of window to find more thumbnails
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    return image_urls

def process_and_save_image(response, fruit_name, index):
    """Process downloaded image and save if it meets requirements"""
    try:
        # Open image and convert to RGB
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Check minimum size
        if min(img.size) < 400:
            return False
        
        # Resize to 512x512
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save processed image
        fruit_dir = DOWNLOAD_PATH / fruit_name
        file_path = fruit_dir / f"{fruit_name}_{index}.jpg"
        img_resized.save(file_path, "JPEG", quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def download_image(img_url, fruit_name, index):
    if img_url and img_url.startswith("http"):
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                return process_and_save_image(response, fruit_name, index)
        except Exception as e:
            print(f"Error downloading image: {e}")
    return False

def setup_directories():
    """Create necessary directories"""
    DOWNLOAD_PATH.mkdir(exist_ok=True)
    for fruit in FRUITS:
        (DOWNLOAD_PATH / fruit).mkdir(exist_ok=True)

def download_fruit_images():
    setup_directories()
    driver = setup_driver()
    results = {}

    for fruit in FRUITS:
        print(f"\nProcessing {fruit}...")
        fruit_path = DOWNLOAD_PATH / fruit
        
        image_urls = locate_images(driver, fruit, IMAGES_PER_FRUIT)
        successful_downloads = 0
        
        for i, img_url in enumerate(image_urls):
            if download_image(img_url, fruit, successful_downloads):
                successful_downloads += 1
                print(f"Successfully downloaded and processed image {successful_downloads} for {fruit}")
            
            if successful_downloads >= IMAGES_PER_FRUIT:
                break
                
        results[fruit] = successful_downloads
        print(f"Completed {fruit}: {successful_downloads} images")

    driver.quit()

    # Print summary
    print("\nDownload Summary:")
    for fruit, count in results.items():
        print(f"{fruit}: {count} images")

if __name__ == "__main__":
    download_fruit_images()