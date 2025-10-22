import requests

API_URL = "http://127.0.0.1:8000"

def test_predict_success():
    payload = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "features": [
            "home_xg", "away_xg", "home_shots_on_target", "away_shots_on_target",
            "home_possession", "away_possession", "home_passing_accuracy", "away_passing_accuracy",
            "home_corners", "away_corners", "home_touches", "away_touches",
            "home_form_score", "away_form_score", "xg_diff", "form_diff",
            "possession_diff", "h2h_home_wins", "h2h_away_wins", "h2h_draws"
        ]
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "random_forest" in result
    assert "xgboost" in result
    assert "ensemble" in result
    print("test_predict_success passed")

test_predict_success()
