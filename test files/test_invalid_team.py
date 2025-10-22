import requests

API_URL = "http://127.0.0.1:8000"

def test_predict_invalid_team():
    payload = {
        "home_team": "Fake United",
        "away_team": "Chelsea",
        "features": ["home_xg", "away_xg"]
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 400
    print("test_predict_invalid_team passed")

test_predict_invalid_team()