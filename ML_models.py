import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

def load_and_prepare_data(db_path="matches.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] >= "2016-08-01"].sort_values("Date").reset_index(drop=True)
    return df

def engineer_features(df):
    def calculate_form_points(team, matches):
        points = 0
        for _, row in matches.iterrows():
            if team == row["Home Team"]:
                if row["Home Goals"] > row["Away Goals"]: points += 3
                elif row["Home Goals"] == row["Away Goals"]: points += 1
            elif team == row["Away Team"]:
                if row["Away Goals"] > row["Home Goals"]: points += 3
                elif row["Away Goals"] == row["Home Goals"]: points += 1
        return points / len(matches) if len(matches) > 0 else 0

    home_form_scores, away_form_scores = [], []
    h2h_home_wins, h2h_away_wins, h2h_draws = [], [], []

    for idx, row in df.iterrows():
        match_date = row["Date"]
        home_team = row["Home Team"]
        away_team = row["Away Team"]
        past_matches = df[df["Date"] < match_date]
        last_home = past_matches[(past_matches["Home Team"] == home_team) | (past_matches["Away Team"] == home_team)].tail(5)
        last_away = past_matches[(past_matches["Home Team"] == away_team) | (past_matches["Away Team"] == away_team)].tail(5)
        home_form_scores.append(calculate_form_points(home_team, last_home))
        away_form_scores.append(calculate_form_points(away_team, last_away))
        h2h = past_matches[((past_matches["Home Team"] == home_team) & (past_matches["Away Team"] == away_team)) |
                           ((past_matches["Home Team"] == away_team) & (past_matches["Away Team"] == home_team))]
        hw = sum((h2h["Home Team"] == home_team) & (h2h["Home Goals"] > h2h["Away Goals"]))
        aw = sum((h2h["Away Team"] == away_team) & (h2h["Away Goals"] > h2h["Home Goals"]))
        dr = sum(h2h["Home Goals"] == h2h["Away Goals"])
        h2h_home_wins.append(hw)
        h2h_away_wins.append(aw)
        h2h_draws.append(dr)

    df["home_form_score"] = home_form_scores
    df["away_form_score"] = away_form_scores
    df["h2h_home_wins"] = h2h_home_wins
    df["h2h_away_wins"] = h2h_away_wins
    df["h2h_draws"] = h2h_draws
    rename_map = {
        "Home xG": "home_xg",
        "Away xG": "away_xg"
    }
    df.rename(columns=rename_map, inplace=True)

    numeric_cols = [
        "home_xg", "away_xg", "home_shots_on_target", "away_shots_on_target",
        "home_possession", "away_possession", "home_passing_accuracy", "away_passing_accuracy",
        "home_corners", "away_corners", "home_touches", "away_touches"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    df["form_diff"] = df["home_form_score"] - df["away_form_score"]
    df["possession_diff"] = df["home_possession"] - df["away_possession"]

    features = numeric_cols + [
        "home_form_score", "away_form_score",
        "xg_diff", "form_diff", "possession_diff",
        "h2h_home_wins", "h2h_away_wins", "h2h_draws"
    ]

    df_model = df.dropna(subset=features + ["Result"])
    X = df_model[features]
    y = df_model.loc[X.index, "Result"]

    return X, y, df_model


def build_poisson_model(df):
    """Constructs attack/defense strength dictionaries based on historical goal stats."""
    df_poisson = df[["Date", "Home Team", "Away Team", "Home Goals", "Away Goals"]].dropna()
    mu = df_poisson["Away Goals"].mean()
    mu_home = np.log(df_poisson["Home Goals"].mean()) - np.log(mu)

    teams = pd.concat([df_poisson["Home Team"], df_poisson["Away Team"]]).unique()
    attack_strength = {}
    defense_strength = {}

    for team in teams:
        att_goals = df_poisson[df_poisson["Home Team"] == team]["Home Goals"].sum() + \
                    df_poisson[df_poisson["Away Team"] == team]["Away Goals"].sum()
        def_goals = df_poisson[df_poisson["Home Team"] == team]["Away Goals"].sum() + \
                    df_poisson[df_poisson["Away Team"] == team]["Home Goals"].sum()
        matches = len(df_poisson[(df_poisson["Home Team"] == team) | (df_poisson["Away Team"] == team)])

        attack_strength[team] = np.log(att_goals / matches + 1e-5) - np.log(mu)
        defense_strength[team] = np.log(def_goals / matches + 1e-5) - np.log(mu)

    return mu, mu_home, attack_strength, defense_strength

def predict_poisson_goals(home_team, away_team, mu, mu_home, attack_strength, defense_strength):
    """Returns expected goals (λ) for home and away team."""
    att_home = attack_strength.get(home_team, 0)
    def_away = defense_strength.get(away_team, 0)
    att_away = attack_strength.get(away_team, 0)
    def_home = defense_strength.get(home_team, 0)

    lambda_home = np.exp(np.log(mu) + mu_home + att_home + def_away)
    lambda_away = np.exp(np.log(mu) + att_away + def_home)

    return lambda_home, lambda_away

def train_on_selected_features(X, y, model_features):
    print(f"Training model on features: {model_features}")
    

    X = X[model_features].copy()    # Overwrite X directly so the model learns only selected features
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y_encoded)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_model.fit(X, y_encoded)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty='l1',
            C=10,
            solver='liblinear',
            max_iter=1000,
            random_state=42
        ))
    ])
    lr_model.fit(X, y_encoded)
    

    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('lr', lr_model)],
        voting='soft'
    )
    ensemble_model.fit(X, y_encoded)

    return rf_model, xgb_model, lr_model, ensemble_model, le

