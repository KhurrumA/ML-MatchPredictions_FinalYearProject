# Data-Driven Match Predictor (Final Year Project)

A machine learning–based football match prediction system that combines statistical modeling, data engineering, and explainable AI to predict the outcomes of football matches.  
Built as part of a final year university project by **Khurrum Arif**.

---

## Overview

This project uses historical match data and machine learning models to predict outcomes of football matches (Home Win, Draw, Away Win).  
It integrates:
- **FastAPI** backend for serving trained model predictions.
- **Streamlit** web app for an interactive match simulation experience.
- **SHAP** for model interpretability and feature importance visualization.

---

## Features

- **Multiple ML Models** — Random Forest, XGBoost, Logistic Regression  
- **Automated Data Processing** from SQLite database  
- **Model Evaluation** using accuracy, F1-score, confusion matrix  
- **Explainable AI** via SHAP value visualization  
- **Web Interface** built with Streamlit  
- **FastAPI Backend** to serve predictions  
- **Automated Data Scraping** with Selenium and BeautifulSoup  

---

## Project Structure

```
ML-MatchPredictions_FinalYearProject/
│
├── api_backend.py                    # FastAPI backend for prediction
├── streamlit_app.py                  # Streamlit front-end web app
├── ML_models.py                      # Core ML model definitions and feature engineering
├── evaluation of models uploaded.py  # Model evaluation and comparison
├── SHAP_explained.py                 # Model interpretability with SHAP
├── matches.db                        # SQLite database of historical match data
├── pages/                            # Streamlit pages and supporting modules
│   └── matches.py
│
├── Data Scraping/                    # Scripts for fetching and cleaning raw football data
│   ├── combineStats.py
│   ├── matchResults.py
│   ├── URLsToScrape.py
│   └── urls_to_be_scraped.csv
│
│
└── requirements.txt                  # Dependency list
```

---

## Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/<yourusername>/ML-MatchPredictions_FinalYearProject.git
   cd ML-MatchPredictions_FinalYearProject
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate       # For macOS/Linux
   venv\Scripts\activate          # For Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Scraping and Preprocessing

### Overview
The dataset used for training and prediction is collected automatically using Selenium and BeautifulSoup.  
These scripts fetch football match data from external websites, extract key statistics (such as goals, shots, and possession), and save the results to the local SQLite database `matches.db`.

### Steps to Collect and Prepare Data

1. **Set up ChromeDriver for Selenium**
   - Install Google Chrome if not already installed.
   - Download ChromeDriver from [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads).
   - Ensure the ChromeDriver version matches your browser version.
   - Add the driver executable to your system PATH or specify its location in the scripts.

2. **Collect URLs for Matches**
   ```bash
   python "Data Scraping/URLsToScrape.py"
   ```
   This script gathers URLs for matches or seasons to scrape.

3. **Scrape Match Results**
   ```bash
   python "Data Scraping/matchResults.py"
   ```
   Fetches detailed match data (team stats, scores, and performance metrics).

4. **Combine and Clean the Data**
   ```bash
   python "Data Scraping/combineStats.py"
   ```
   This script merges and formats the scraped data, preparing it for machine learning analysis.

5. **Verify the Database**
   Ensure `matches.db` has been populated correctly before proceeding with model training.

---

## How to Run

### 1. Start the FastAPI Backend
```bash
uvicorn api_backend:app --reload
```
This launches the API server locally at `http://127.0.0.1:8000`.

### 2. Start the Streamlit Web App
Open a new terminal and run:
```bash
streamlit run streamlit_app.py
```
This starts the user interface.  
It connects automatically to the backend (`API_URL` set to `http://127.0.0.1:8000`).

### 3. Using the App
- Select two teams from the dropdowns.
- The model predicts **Home Win / Draw / Away Win** probabilities.
- SHAP visualizations show which features influenced the result.

---

## Model Training and Evaluation

To retrain or evaluate models:
```bash
python "evaluation of models uploaded.py"
```

This script:
- Loads match data from `matches.db`
- Trains multiple models (RandomForest, XGBoost, LogisticRegression)
- Evaluates them using cross-validation and performance metrics
- Displays model comparison charts and confusion matrices

---

## Explainability

`SHAP_explained.py` visualizes feature contributions for individual predictions.  
It uses **SHAP TreeExplainer** for tree-based models like RandomForest and XGBoost.

---

## Setup Notes for Selenium

If you are scraping data, make sure:
- ChromeDriver is correctly installed and accessible.
- The driver path is configured in the scraping scripts.
- If running on a server, add:
  ```python
  options.add_argument("--headless")
  ```
  to run Chrome in headless mode.

Example:
```python
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
```

---

## Results

- Models achieve high prediction accuracy across multiple historical seasons.
- SHAP analysis identifies features that most influence predictions, such as team form, goals scored, and defensive performance.

---

## Future Work

- Extend coverage to multiple leagues and seasons  
- Incorporate real-time data APIs  
- Improve visualization of prediction confidence  
- Deploy live via Streamlit Cloud or Render  

---

## Author

**Khurrum Arif**  
Final Year Project — Department of Computer Science, University of Leicester  
(Code received a First Class Mark)

---

## License

This project is for educational purposes only.  
You may modify and extend it for learning and research use.
