import streamlit as st
import requests
import os
import time

API_URL = "http://127.0.0.1:8000"

#  [Heuristic 8: Aesthetic and Minimalist Design]
st.set_page_config(layout="centered", page_title=" Match Predictor", page_icon="⚽")

#  [Heuristic 1: Visibility of System Status]
st.title("A Data Driven Match Simulator")
st.caption("Created by Khurrum Arif")

#  [Heuristic 2: Match Between System and the Real World]
st.markdown("""
Welcome to the **Data-Driven Match Predictor**!  
Use real historical data and advanced models to simulate match outcomes.  
""")

#  [Heuristic 9: Help Users Recognize, Diagnose, and Recover from Errors]
@st.cache_data(ttl=3600)
def get_teams():
    try:
        response = requests.get(f"{API_URL}/teams", timeout=5)
        response.raise_for_status()
        return response.json()["teams"]
    except Exception as e:
        st.error(f" Failed to load teams: {e}")
        return []

@st.cache_data(ttl=3600)
def get_features():
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        response.raise_for_status()
        return response.json()["features"]
    except Exception as e:
        st.error(f"️ Failed to load features: {e}")
        return []

#  [Heuristic 1: Visibility of System Status]
with st.spinner("Loading data from the server..."):
    teams = get_teams()
    all_features = get_features()

#  [Heuristic 9: Error Handling]
if not teams:
    st.error(" No team data available.")
    st.stop()

#  [Heuristic 4: Consistency and Standards]
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(" Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_team = st.selectbox(" Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

#  [Heuristic 5: Error Prevention]
if home_team == away_team:
    st.warning(" Please select two different teams.")
    st.stop()

#  [Heuristic 6: Recognition Rather Than Recall]
st.subheader(" Model Features")
st.markdown("Choose the features the models will use to predict the outcome:")

#  [Heuristic 7: Flexibility and Efficiency of Use]
selected_features = st.multiselect(
    " Select features:",
    options=all_features,
    default=all_features,
    help="Using all features typically yields the best predictions."
)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

#  [Heuristic 3: User Control and Freedom]
if st.button("Predict Match"):
    payload = {
        "home_team": home_team,
        "away_team": away_team,
        "features": selected_features
    }

    with st.spinner(" Making predictions..."):
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response_time = time.time() - start_time

            if response.status_code != 200:
                st.error(f" Prediction failed: {response.text}")
                st.stop()

            result = response.json()
            st.success(f" Prediction complete in {response_time:.2f} seconds.")

            #  [Heuristic 1: Visibility of System Status]
            st.subheader(" Poisson Goal Estimate")
            if result.get("poisson_goals"):
                col1, col2 = st.columns(2)
                col1.metric(f"{result['home_team']} expected goals", result["poisson_goals"][result["home_team"]])
                col2.metric(f"{result['away_team']} expected goals", result["poisson_goals"][result["away_team"]])
            else:
                st.info("️ Poisson model could not provide a goal estimate.")

            #  [Heuristic 2 + 6: Real-World Terms + Recognition]
            st.subheader(" Random Forest Prediction")
            rf = result.get("random_forest")
            if rf:
                st.write(f"**Prediction**: {rf['result']}")
                st.markdown(f" _Reasoning_: {rf['explanation']}")
                if os.path.exists("shap_plot_randomforestclassifier.png"):
                    st.image("shap_plot_randomforestclassifier.png", caption="Feature Impact (RF)")
            else:
                st.info("Random Forest prediction unavailable.")

            st.subheader(" XGBoost Prediction")
            xgb = result.get("xgboost")
            if xgb:
                st.write(f"**Prediction**: {xgb['result']}")
                st.markdown(f" _Reasoning_: {xgb['explanation']}")
                if os.path.exists("shap_plot_xgbclassifier.png"):
                    st.image("shap_plot_xgbclassifier.png", caption="Feature Impact (XGBoost)")
            else:
                st.info("XGBoost prediction unavailable.")

            st.subheader(" Ensemble Prediction (RF + XGB + LR)")
            ensemble = result.get("ensemble")
            if ensemble:
                st.write(f"**Final Prediction**: {ensemble['result']}")
            else:
                st.info("Ensemble model is not available.")

            #  [Heuristic 10: Help & Documentation]
            with st.expander(" Debug Info"):
                st.json(result)

        #  [Heuristic 9: Help Users Recover From Errors]
        except requests.exceptions.RequestException as e:
            st.error(f" API connection failed: {e}")
        except Exception as e:
            st.error(f" Unexpected error: {e}")
