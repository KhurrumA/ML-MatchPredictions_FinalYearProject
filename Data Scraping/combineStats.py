import pandas as pd
import time
import random
import sqlite3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

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

#Extract stats from bar graphic on html page
def extract_bar_stats(soup):
    stats = {}
    stat_table = soup.find("table", {"cellspacing": "8"})
    if not stat_table:
        return stats
    rows = stat_table.find_all("tr")
    labels = ["possession", "passing_accuracy", "shots_on_target"]
    index = 0
    for row in rows:
        tds = row.find_all("td")
        if len(tds) == 2 and index < len(labels):
            label = labels[index]
            if label == "shots_on_target":
                stats["home_shots_on_target"] = extract_shots(tds[0])
                stats["away_shots_on_target"] = extract_shots(tds[1])
            else:
                try:
                    stats[f"home_{label}"] = tds[0].find("strong").text.strip().replace("%", "")
                    stats[f"away_{label}"] = tds[1].find("strong").text.strip().replace("%", "")
                except:
                    continue
            index = index + 1
    return stats

def extract_shots(td):
    text = td.get_text(separator=" ").strip()
    match = re.search(r'(\d+)\s+of\s+\d+', text)
    return match.group(1) if match else "0"

#Extract stats that are from grids in html page
def extract_grid_stats(soup):
    stats = {}
    block = soup.find("div", id="team_stats_extra")
    if not block:
        return stats
    divs = [d.get_text(strip=True) for d in block.find_all("div") if d.get_text(strip=True)]
    keys = {"corners", "touches"}
    for i in range(3, len(divs) - 2, 3):
        try:
            stat = divs[i+1].lower().replace(" ", "_")
            if stat in keys:
                stats[f"home_{stat}"] = divs[i]
                stats[f"away_{stat}"] = divs[i+2]
        except IndexError:
            continue
    return stats

# 4. Scraping logic for a single row
def scrape_match_stats(driver, url):
    driver.get(url)
    time.sleep(random.uniform(3, 5))
    soup = BeautifulSoup(driver.page_source, "html.parser")
    bar_stats = extract_bar_stats(soup)
    grid_stats = extract_grid_stats(soup)
    return {**bar_stats, **grid_stats}


if __name__ == "__main__":
    urls_df = pd.read_csv("urls_to_be_scraped.csv")
    conn = sqlite3.connect("matches.db")
    matches_df = pd.read_sql_query("SELECT * FROM matches", conn)

    driver = get_driver()
    missing_matches = []

    for idx, row in urls_df.iterrows():
        url = row['Match Report URL']
        fbref_id = url.split("/")[5]

        print(f"Currently Scraping {url} ({idx+1}/{len(urls_df)})")
        try:
            combined = scrape_match_stats(driver, url)
            match_row = matches_df[matches_df['Match ID'] == fbref_id]
            if not match_row.empty:
                for key, value in combined.items():
                    matches_df.loc[match_row.index[0], key] = value

                #Checking if row didnt scrape properly, might be due to too many requests
                if not combined:
                    missing_matches.append((url, fbref_id))
        except Exception as e:
            print(f" Error: {url}: {e}")
            missing_matches.append((url, fbref_id))
            continue

        if (idx + 1) % 50 == 0:
            matches_df.to_sql("matches", conn, if_exists="replace", index=False)
            print(f"Saved after {idx+1} matches.")

    print("\nInitial scraping complete. Starting retry pass for missing data...")

    #Run through incomplete rows right at the end
    target_columns = ["home_possession", "home_shots_on_target", "home_corners"]
    retry_df = matches_df[matches_df[target_columns].isnull().any(axis=1)]
    retry_ids = retry_df['Match ID'].tolist()

    for url, fbref_id in zip(urls_df['Match Report URL'], urls_df['Match Report URL'].apply(lambda u: u.split("/")[5])):
        if fbref_id in retry_ids:
            print(f"Retrying: {url}")
            try:
                combined = scrape_match_stats(driver, url)
                match_row = matches_df[matches_df['Match ID'] == fbref_id]
                if not match_row.empty:
                    for key, value in combined.items():
                        matches_df.loc[match_row.index[0], key] = value
            except Exception as e:
                print(f" Retry Error: {url}: {e}")

    driver.quit()
    matches_df.to_sql("matches", conn, if_exists="replace", index=False)
    conn.close()
    print("All data processed and saved with retry attempt complete.")
