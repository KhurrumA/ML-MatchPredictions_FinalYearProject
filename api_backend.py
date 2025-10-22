from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import logging
from datetime import datetime
from ML_models import (
    load_and_prepare_data,
    engineer_features,
    train_on_selected_features,
    build_poisson_model,
    predict_poisson_goals
)
from SHAP_explained import explain_single_prediction
import sqlite3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


logger.info("Loading and preparing data...")
df = load_and_prepare_data()
X, y, df_model = engineer_features(df)
mu, mu_home, att_strength, def_strength = build_poisson_model(df)

valid_teams = sorted(df["Home Team"].unique())
all_features = list(X.columns)
logger.info(f"Loaded data with {len(df)} matches and {len(all_features)} features")

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    features: List[str]

@app.get("/teams")
def get_team_names():
    return {"teams": valid_teams}

@app.get("/features")
def get_feature_names():
    return {"features": all_features}

@app.post("/predict")
def predict_match(req: MatchRequest):
    logger.info(f"Prediction request for {req.home_team} vs {req.away_team}")

    if req.home_team not in valid_teams or req.away_team not in valid_teams:
        msg = f"Invalid team name. Valid teams: {valid_teams}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    feature_aliases = {
        "Home xG": "home_xg",
        "Away xG": "away_xg",
        "Home Shots on Target": "home_shots_on_target",
        "Away Shots on Target": "away_shots_on_target",
        "Home Possession": "home_possession",
        "Away Possession": "away_possession",
        "Home Passing Accuracy": "home_passing_accuracy",
        "Away Passing Accuracy": "away_passing_accuracy",
        "Home Corners": "home_corners",
        "Away Corners": "away_corners",
        "Home Touches": "home_touches",
        "Away Touches": "away_touches",
        "xG Diff": "xg_diff",
        "Form Diff": "form_diff",
        "Possession Diff": "possession_diff",
        "Home Form Score": "home_form_score",
        "Away Form Score": "away_form_score",
        "h2h_home_wins": "h2h_home_wins",
        "h2h_away_wins": "h2h_away_wins",
        "h2h_draws": "h2h_draws"
    }

    model_features = [feature_aliases.get(f, f) for f in req.features]

    missing_features = set(model_features) - set(all_features)
    if missing_features:
        msg = f"Invalid features selected: {missing_features}. Valid features: {all_features}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    h2h_matches = df_model[
        (df_model["Home Team"] == req.home_team) &
        (df_model["Away Team"] == req.away_team)
    ]

    if not h2h_matches.empty:
        latest_row = pd.DataFrame(columns=df_model.columns)
        latest_row.loc[0, "Home Team"] = req.home_team
        latest_row.loc[0, "Away Team"] = req.away_team
        logger.info("Averaging over all head-to-head match data")
        for col in df_model.columns:
            if col.startswith("home_") or col.startswith("h2h_home"):
                latest_row.at[0, col] = h2h_matches[col].mean()
            elif col.startswith("away_") or col.startswith("h2h_away"):
                latest_row.at[0, col] = h2h_matches[col].mean()
            elif col in ["xg_diff", "form_diff", "possession_diff"]:
                latest_row.at[0, col] = h2h_matches[col].mean()
            elif col == "h2h_draws":
                latest_row.at[0, col] = h2h_matches[col].mean()
    else:
        logger.info("No direct head-to-head, using recent form data")

        recent_home = df_model[
            (df_model["Home Team"] == req.home_team)
        ].sort_values("Date", ascending=False).head(5)

        recent_away = df_model[
            (df_model["Away Team"] == req.away_team)
        ].sort_values("Date", ascending=False).head(5)

        if recent_home.empty or recent_away.empty:
            missing_teams = []
            if recent_home.empty:
                missing_teams.append(req.home_team)
            if recent_away.empty:
                missing_teams.append(req.away_team)

            msg = f"Prediction unavailable: Not enough historical data for {', '.join(missing_teams)}."
            logger.warning(msg)
            return {
                "home_team": req.home_team,
                "away_team": req.away_team,
                "poisson_goals": None,
                "random_forest": {
                    "result": "Unavailable",
                    "explanation": msg
                },
                "xgboost": {
                    "result": "Unavailable",
                    "explanation": msg
                }
            }

        latest_row = pd.DataFrame(columns=df_model.columns)
        latest_row.loc[0, "Home Team"] = req.home_team
        latest_row.loc[0, "Away Team"] = req.away_team

        for col in df_model.columns:
            if col.startswith("home_") or col.startswith("h2h_home"):
                latest_row.at[0, col] = recent_home[col].mean()
            elif col.startswith("away_") or col.startswith("h2h_away"):
                latest_row.at[0, col] = recent_away[col].mean()
            elif col in ["xg_diff", "form_diff", "possession_diff"]:
                latest_row.at[0, col] = recent_home[col].mean()
            elif col == "h2h_draws":
                latest_row.at[0, col] = 0

    required_cols = ["home_form_score", "away_form_score", "h2h_home_wins", "h2h_away_wins", "h2h_draws"]
    if latest_row[required_cols].isnull().any(axis=None):
        msg = "Missing form/head-to-head data to simulate this match"
        logger.warning(msg)
        return {
            "home_team": req.home_team,
            "away_team": req.away_team,
            "poisson_goals": None,
            "random_forest": {
                "result": "Unavailable",
                "explanation": msg
            },
            "xgboost": {
                "result": "Unavailable",
                "explanation": msg
            }
        }

    try:
        match_features = latest_row[model_features].fillna(0).astype(float)
        logger.info(f"Prepared features with shape {match_features.shape}")
    except Exception as e:
        msg = f"Feature preparation failed: {str(e)}"
        logger.error(msg, exc_info=True)
        raise HTTPException(status_code=400, detail=msg)

    try:
        rf_model, xgb_model, lr_model, ensemble_model, le = train_on_selected_features(X, y, model_features)

        rf_pred_class = rf_model.predict(match_features)[0]
        rf_pred_label = le.inverse_transform([rf_pred_class])[0]
        rf_reason = explain_single_prediction(rf_model, match_features.iloc[[0]], model_features)

        xgb_pred_class = xgb_model.predict(match_features)[0]
        xgb_pred_label = le.inverse_transform([xgb_pred_class])[0]
        xgb_reason = explain_single_prediction(xgb_model, match_features.iloc[[0]], model_features)

        lr_pred_class = lr_model.predict(match_features)[0]
        lr_pred_label = le.inverse_transform([lr_pred_class])[0]
        lr_reason = explain_single_prediction(lr_model, match_features.iloc[[0]], model_features)

        ensemble_pred_class = ensemble_model.predict(match_features)[0]
        ensemble_pred_label = le.inverse_transform([ensemble_pred_class])[0]

        lambda_home, lambda_away = predict_poisson_goals(req.home_team, req.away_team, mu, mu_home, att_strength, def_strength)

        logger.info("Predictions completed successfully")
        return {
            "home_team": req.home_team,
            "away_team": req.away_team,
            "poisson_goals": {
                req.home_team: round(lambda_home, 2),
                req.away_team: round(lambda_away, 2)
            },
            "random_forest": {
                "result": rf_pred_label,
                "explanation": rf_reason,
                "debug_shape": str(match_features.shape),
                "debug_features": model_features
            },
            "xgboost": {
                "result": xgb_pred_label,
                "explanation": xgb_reason,
                "debug_shape": str(match_features.shape),
                "debug_features": model_features
            },
            "ensemble": {
                "result": ensemble_pred_label,
            }
        }
    except Exception as e:
        msg = f"Prediction failed: {str(e)}"
        logger.error(msg, exc_info=True)
        raise HTTPException(status_code=500, detail=msg)

@app.get("/seasons")
def get_seasons():
    conn = sqlite3.connect("matches.db")
    df = pd.read_sql_query("SELECT DISTINCT Season FROM matches", conn)
    conn.close()
    return {"seasons": sorted(df["Season"].dropna().unique().tolist())}

@app.get("/match-stats")
def get_match_stats(season: str, team: str):
    conn = sqlite3.connect("matches.db")
    df = pd.read_sql_query(
        "SELECT * FROM matches WHERE Season = ? AND (`Home Team` = ? OR `Away Team` = ?)",
        conn,
        params=(season, team, team)
    )
    conn.close()

    if df.empty:
        return {"matches": []}

    match_list = []
    for _, row in df.iterrows():
        match = {}
        for col in df.columns:
            val = row[col]
            match[col] = val if pd.notnull(val) else "No data"
        match_list.append(match)

    return {"matches": match_list}
