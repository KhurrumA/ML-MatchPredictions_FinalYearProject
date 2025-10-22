import requests

API_URL = "http://127.0.0.1:8000"

def test_predict_invalid_feature():
    payload = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "features": ["home_xg", "non_existent_feature"]
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 400
    print("test_predict_invalid_feature passed")

test_predict_invalid_feature()