from selenium import webdriver #imports that are needed
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import sqlite3
import pandas as pd

#Set up chrome driver
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

#scrape every seqason 
def get_match_data(season_url, season):
    driver = get_driver() #launch chrome using parameters set
    print(f"\n Opening {season_url} for {season}...") #shows what season is currently being scraped
    driver.get(season_url) #loads fixture page that needs to be accessed

    time.sleep(5) #5 second timer so I dont get kicked off of FBRef

    matches = [] #list created to store match data for that specific season
    try:
        WebDriverWait(driver, 10).until( #10 second timer just to make sure that everything on th  epage is loaded
            EC.presence_of_element_located((By.CLASS_NAME, "stats_table"))
        )
        fixture_table = driver.find_element(By.CLASS_NAME, "stats_table") #gets the elemnts containing table needed
        rows = fixture_table.find_elements(By.TAG_NAME, "tr") #grabs all of the tr elements

        for row in rows:#loops the rows and grabs all of the td cells
            cols = row.find_elements(By.TAG_NAME, "td")
            if not cols or len(cols) < 6: #rows that are not real matches or are empty are skipped. This was to fix a formatting issue in the table
                continue

            data = {col.get_attribute("data-stat"): col.text.strip() for col in cols} #dictionary created to get "data-stat" attribute

            if not data.get("home_team") or not data.get("away_team") or not data.get("score"): #incomplete rows are skipped
                continue

            try:
                date_element = row.find_element(By.CSS_SELECTOR, 'td[data-stat="date"]') #find the date cell in the row
                raw_date = date_element.get_attribute("csk")#FBRef uses a custom attribute csk to store date
                date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if raw_date else "" #formatting into date format that is needed
            except:
                date = "" #if no date keeps empty

            home_team = data.get("home_team", "") #match score and xG are scraped
            away_team = data.get("away_team", "")
            score = data.get("score", "")
            home_xG = data.get("home_xg", "")
            away_xG = data.get("away_xg", "")
             #regex used to match score string and extract amount of goals scored by either home or away
            home_goals, away_goals = None, None #two variable initialised
            score_match = re.match(r"(\d+)\u2013(\d+)", score) #\u2013 unicode for dash character used by FBRef 
            if score_match:#checks if match was found successfully
                home_goals, away_goals = map(int, score_match.groups())

            try: # converts xG string to floats, if missing it handles the error
                home_xG = float(home_xG.replace(",", "")) if home_xG else None
                away_xG = float(away_xG.replace(",", "")) if away_xG else None
            except ValueError:
                home_xG = away_xG = None

            result = "Draw"#Calculates the outcome of the match
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = "Home Win"
                elif home_goals < away_goals:
                    result = "Away Win"

            xG_winner = "Unknown" #Calculates winner based on xG
            if home_xG is not None and away_xG is not None:
                if home_xG > away_xG:
                    xG_winner = "Home Win"
                elif home_xG < away_xG:
                    xG_winner = "Away Win"
                else:
                    xG_winner = "Draw"

            match_url = "" # initialised match_id and match_url
            match_id = ""
            try:
                match_link = row.find_element(By.CSS_SELECTOR, 'td[data-stat="match_report"] a')
                match_url = match_link.get_attribute("href")
                match_id = str(match_url.split("/")[-2]) if match_url else "" #gets match id that FBRef uses
            except:
                pass

            matches.append([ #adds data to matches
                season, date, home_team, home_goals, away_goals,
                away_team, home_xG, away_xG, result, xG_winner, match_id
            ])

        print(f"Recovered {len(matches)} matches for {season}") #Tells user how many matches have been revoered (should be 380)
        driver.quit()
        return matches #browser closed and matches is returned

    except Exception as e: #shows if there were errors and in which season
        print(f"Error: season {season}: {e}")
        driver.quit()
        return []

#save to sqllite database called matches.db
def save_matches_to_sqlite(matches, db_name="matches.db"):
    df = pd.DataFrame(matches, columns=[ #pandas dataframe created from list of matches
        "Season", "Date", "Home Team", "Home Goals", "Away Goals",
        "Away Team", "Home xG", "Away xG", "Result", "xG Winner", "Match ID"
    ])

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor() #opens connection to database and creates a cursor

    # Drops matches if it already exists and recreates table with primary key
    cursor.execute("DROP TABLE IF EXISTS matches")
    cursor.execute("""
        CREATE TABLE matches (
            "Season" TEXT,
            "Date" TEXT,
            "Home Team" TEXT,
            "Home Goals" INTEGER,
            "Away Goals" INTEGER,
            "Away Team" TEXT,
            "Home xG" REAL,
            "Away xG" REAL,
            "Result" TEXT,
            "xG Winner" TEXT,
            "Match ID" TEXT PRIMARY KEY  
        )
    """)#Match ID is primary key as it is unique 

    df.to_sql("matches", conn, if_exists="append", index=False) #dataframe saved to sql databse
    conn.commit()
    conn.close()

    print(f"\n Saved {len(df)} matches to {db_name} with Match ID as PRIMARY KEY") #prints how many matches were saved

#Main function that loops through season 2004-2024
if __name__ == "__main__":
    seasons = [(year, year + 1) for year in range(2004, 2024)] #creates list like (2004,2005),(2005,2006) etc.
    base_url = "https://fbref.com/en/comps/9/{season}-{next_season}/schedule/{season}-{next_season}-Premier-League-Scores-and-Fixtures"
    #base url for the different seasons pages
    all_matches = [] #list to collect all match data for each season

    for season, next_season in seasons: #loops through all seasons and appends it to all_matches
        season_url = base_url.format(season=season, next_season=next_season)
        season_label = f"{season}-{next_season}"
        season_matches = get_match_data(season_url, season_label)
        all_matches.extend(season_matches)

    save_matches_to_sqlite(all_matches) #saves to database
