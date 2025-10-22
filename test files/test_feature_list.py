import requests

API_URL = "http://127.0.0.1:8000"

def test_get_features():
    response = requests.get(f"{API_URL}/features")
    assert response.status_code == 200
    features = response.json()["features"]
    assert isinstance(features, list)
    assert "home_xg" in features
    print("test_get_features passed")

test_get_features()