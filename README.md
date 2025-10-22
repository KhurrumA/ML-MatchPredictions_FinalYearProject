#  Data-Driven Match Predictor (Final Year Project)

A machine learning–based football match prediction system that combines statistical modeling, data engineering, and explainable AI to predict the outcomes of football matches.  
Built as part of a final year university project by **Khurrum Arif**.

---

##  Overview

This project uses historical match data and machine learning models to predict outcomes of football matches (Home Win, Draw, Away Win).  
It integrates:
- **FastAPI** backend for serving trained model predictions.
- **Streamlit** web app for an interactive match simulation experience.
- **SHAP** for model interpretability and feature importance visualization.

---

##  Features

-  **Multiple ML Models** — Random Forest, XGBoost, Logistic Regression  
-  **Automated Data Processing** from SQLite database  
-  **Model Evaluation** using accuracy, F1-score, confusion matrix  
-  **Explainable AI** via SHAP value visualization  
-  **Web Interface** built with Streamlit  
-  **FastAPI Backend** to serve predictions  

---

##  Project Structure

```
ML-MatchPredictions_FinalYearProject/
│
├── api_backend.py                # FastAPI backend for prediction
├── streamlit_app.py              # Streamlit front-end web app
├── ML_models.py                  # Core ML model definitions and feature engineering
├── evaluation of models uploaded.py # Model evaluation and comparison
├── SHAP_explained.py             # Model interpretability with SHAP
├── matches.db                    # SQLite database of historical match data
└── requirements.txt              # Dependency list
```

---

##  Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/<yourusername>/ML-MatchPredictions_FinalYearProject.git
   cd ML-MatchPredictions_FinalYearProject
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate       # For macOS/Linux
   venv\Scripts\activate        # For Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

##  How to Run

###  1. Start the FastAPI Backend
```bash
uvicorn api_backend:app --reload
```
This launches the API server locally at `http://127.0.0.1:8000`.

###  2. Start the Streamlit Web App
Open a new terminal and run:
```bash
streamlit run streamlit_app.py
```
This starts the user interface.  
It connects automatically to the backend (`API_URL` set to `http://127.0.0.1:8000`).

###  3. Using the App
- Select two teams from the dropdowns.
- The model predicts **Home Win / Draw / Away Win** probabilities.
- Visual explanations (via SHAP) show which features influenced the result.

---

##  Model Training and Evaluation

To retrain or evaluate models:
```bash
python "evaluation of models uploaded.py"
```

This script:
- Loads match data from `matches.db`
- Trains multiple models (RandomForest, XGBoost, LogisticRegression)
- Evaluates them using cross-validation and accuracy metrics
- Displays a confusion matrix and model comparison charts

---

##  Explainability

`SHAP_explained.py` visualizes feature contributions for individual predictions.  
It uses **SHAP TreeExplainer** for tree-based models like RandomForest and XGBoost.

---

##  Results

- Models achieve competitive prediction accuracy across historical data.
- SHAP plots reveal the most impactful features (e.g., recent form, goals scored/conceded).



---

##  Future Work

- Expand to multiple leagues and seasons  
- Integrate live data APIs  
- Enhance prediction confidence visualization  
- Deploy online using Streamlit Cloud or Render  

---

##  Author

**Khurrum Arif**  
Final Year Project — Department of Computer Science University of Leicester (Code recieved a First Class Mark)  
*(University name can be added here)*

---

##  License

This project is for educational purposes only.  
You may modify and extend it for learning and research use.
