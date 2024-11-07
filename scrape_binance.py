import requests
from selenium import webdriver 
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

binance_option_url = r"https://www.binance.com/en-IN/eoptions/BTCUSDT"

driver = webdriver.Firefox()

driver.get(url=binance_option_url)

driver.implicitly_wait(5)

cookies_button = driver.find_element(by=By.CSS_SELECTOR, value="#onetrust-reject-all-handler")
cookies_button.click()

date1 = driver.find_element(by=By.CSS_SELECTOR, value=".market-tables > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)").text
print(date1) 
#.market-tables > div:nth-child(7) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)
maturities = []

for table_child in range(1,20):
    try:
        value = f".market-tables > div:nth-child({table_child}) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)"
        date=driver.find_element(by=By.CSS_SELECTOR, value=value).text
        maturities.append(date)
    except Exception as e:
        print(f"There was no {table_child}th table!\n")
        print(e)
driver.implicitly_wait(5)
print(date)
#driver.close()

