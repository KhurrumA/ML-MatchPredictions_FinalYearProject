from selenium import webdriver #these are just the imports that are needed
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

def get_driver():
    chrome_driver_path = r"C:\Users\Khurrum\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe" #path to my chromedriver
    options = Options() #options object created so that I can change settings
    options.add_argument("--disable-blink-features=AutomationControlled") #Disguises browser as not automated to avoid blocking
    options.add_argument("--window-size=1920,1080")#browser size
    options.add_experimental_option("excludeSwitches", ["enable-automation"])#tells chrome not to show message that says "Chrome is being controlled by automated test software"
    options.add_experimental_option("useAutomationExtension", False)#Disables automation extension that communicates with chrome so that browser appears ike a normal session to avoid errors
    service = Service(chrome_driver_path)#initialising ChromeDriver serves by using path
    driver = webdriver.Chrome(service=service, options=options)# launches Chrome
    return driver

def get_match_urls(season_url):#a new chrome window is opened using the get driver function
    driver = get_driver()
    print(f"Opening URL: {season_url}")#shows the current url thats being opeened
    driver.get(season_url)#opens webpage that contains the match reports that need to be scraped
    time.sleep(5) #i put a 5 second waiting time so that the whole webpage opens fully

    try:
        for i in range(3):#for loop to scroll down the loop 3 times. it waits for 2 seconds between every scroll. This is so that I can mimic a real user
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2)

        tables = driver.find_elements(By.TAG_NAME, "table") #gets all the tables on the page
        match_table = None #match table initialised to none, meaning not found yet
        for table in tables: #starts looping through tables
            table_id = table.get_attribute("id") #get id value from the table
            if table_id and "sched_" in table_id: #checks that the table actually has an id and if the string "sched_ is" is in the id. (the way FBRef name the tables)
                match_table = table #stores table in match_table
                break

        if not match_table: #if no table was found, i just made it return an comment and empty list
            print("Couldn't find table.")
            driver.quit()
            return []

        match_links = match_table.find_elements(By.XPATH, ".//td[@data-stat='match_report']/a") #looks i the table for <a> tags under <td> cells with data_stat="match_report"

        base_url = "https://fbref.com" #sets base url
        match_urls = [
            link.get_attribute("href") if link.get_attribute("href").startswith("https") #full list of URLs for each match report
            else base_url + link.get_attribute("href")
            for link in match_links
        ]

        print(f"Found {len(match_urls)} match reports.") #just shows how many URLs were found
    except Exception as e:
        print(f"Error: {e}") #catches and prints error if anything went wrong
        match_urls = []

    driver.quit()
    return match_urls #URLs returned

def save_match_urls(match_urls, filename="urls_to_be_scraped.csv"): #All of the urls are written to a .csv file that will be used later on by another python script
    if not match_urls: 
        print("No match URLs to save.")#if the list is empty nothing is printed
        return

    with open(filename, mode="w", newline="") as file: #opens CSV so that it can write to it. Writes all of the URLs each on a new row
        writer = csv.writer(file)
        writer.writerow(["Match Report URL"])
        for url in match_urls:
            writer.writerow([url])
    print(f"Match URLs saved to {filename}")

if __name__ == "__main__": #runs this peices of code only when file is run directly
    season_urls = [
        f"https://fbref.com/en/comps/9/{year}-{year+1}/schedule/{year}-{year+1}-Premier-League-Scores-and-Fixtures" 
        for year in range(2014, 2024) #creates a list of URLS for each season so that it can loop through each one. Each page has its own match report links
    ]

    all_match_urls = []
    for url in season_urls:
        match_urls = get_match_urls(url) #scrapes through every seasons URLs and then saves them to list
        all_match_urls.extend(match_urls)

    save_match_urls(all_match_urls) # saves all of the match report links to a CSV file
