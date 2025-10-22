import requests

API_URL = "http://127.0.0.1:8000"

def test_get_teams():
    response = requests.get(f"{API_URL}/teams")
    assert response.status_code == 200
    teams = response.json()["teams"]
    assert isinstance(teams, list)
    assert "Arsenal" in teams
    print("test_get_teams passed")

test_get_teams()
