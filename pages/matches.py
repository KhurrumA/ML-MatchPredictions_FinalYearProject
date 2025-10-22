import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Match Explorer", layout="wide")
st.title(" Match Data Explorer")
st.markdown("Select a season and team to view detailed match statistics and visualizations.")

@st.cache_data(ttl=3600)
def load_seasons():
    response = requests.get(f"{API_URL}/seasons")
    return response.json()["seasons"]

@st.cache_data(ttl=3600)
def load_teams():
    response = requests.get(f"{API_URL}/teams")
    return response.json()["teams"]

def plot_pie_chart(home_val, away_val, home_team, away_team, title, metric):
    return px.pie(
        names=[home_team, away_team],
        values=[home_val, away_val],
        title=f"{title} ({metric})"
    )

def plot_bar_chart(home_val, away_val, home_team, away_team, title, ylabel):
    return px.bar(
        x=[home_team, away_team],
        y=[home_val, away_val],
        labels={"x": "Team", "y": ylabel},
        title=title
    )

# Load dropdown data
seasons = load_seasons()
teams = load_teams()

# Sidebar filters
col1, col2 = st.columns(2)
with col1:
    selected_season = st.selectbox("Season", seasons)
with col2:
    selected_team = st.selectbox("Team", teams)

# Initialize session state
if "match_df" not in st.session_state:
    st.session_state.match_df = None

if st.button("🔍 Show Matches"):
    with st.spinner("Fetching match data..."):
        params = {"season": selected_season, "team": selected_team}
        response = requests.get(f"{API_URL}/match-stats", params=params)
        data = response.json()

        if not data["matches"]:
            st.warning("⚠️ No matches found for this selection.")
        else:
            match_df = pd.DataFrame(data["matches"])
            match_df["Match Label"] = match_df["Date"] + " - " + match_df["Home Team"] + " vs " + match_df["Away Team"]
            st.session_state.match_df = match_df
            st.success(f" Found {len(match_df)} matches for {selected_team} in {selected_season}")

# If match data is present
if st.session_state.match_df is not None:
    match_df = st.session_state.match_df
    st.dataframe(match_df, use_container_width=True)

    st.subheader(" Select a Match to View Details")
    dropdown_key = f"match_selector_{selected_season}_{selected_team}"
    selected_match_label = st.selectbox("Choose a Match", match_df["Match Label"], key=dropdown_key)
    selected_row = match_df[match_df["Match Label"] == selected_match_label].iloc[0]

    st.subheader(" Match Details")
    st.json(selected_row.to_dict())

    st.subheader(" Match Visualizations")
    home = selected_row["Home Team"]
    away = selected_row["Away Team"]

    def safe_float(val):
        try:
            return float(val)
        except:
            return None

    charts_shown = False

    # Expected Goals (xG)
    home_xg = safe_float(selected_row.get("home_xg", "No data"))
    away_xg = safe_float(selected_row.get("away_xg", "No data"))
    if home_xg is not None and away_xg is not None:
        st.plotly_chart(plot_bar_chart(home_xg, away_xg, home, away, "Expected Goals (xG)", "xG"))
        charts_shown = True

    # Possession %
    home_poss = safe_float(selected_row.get("home_possession", "No data"))
    away_poss = safe_float(selected_row.get("away_possession", "No data"))
    if home_poss is not None and away_poss is not None:
        st.plotly_chart(plot_pie_chart(home_poss, away_poss, home, away, " Possession Share", "%"))
        charts_shown = True

    # Shots on Target
    home_shots = safe_float(selected_row.get("home_shots_on_target", "No data"))
    away_shots = safe_float(selected_row.get("away_shots_on_target", "No data"))
    if home_shots is not None and away_shots is not None:
        st.plotly_chart(plot_bar_chart(home_shots, away_shots, home, away, " Shots on Target", "Shots"))
        charts_shown = True

    # Corners
    home_corners = safe_float(selected_row.get("home_corners", "No data"))
    away_corners = safe_float(selected_row.get("away_corners", "No data"))
    if home_corners is not None and away_corners is not None:
        st.plotly_chart(plot_bar_chart(home_corners, away_corners, home, away, " Corners", "Corners"))
        charts_shown = True

    # Passing Accuracy
    home_acc = safe_float(selected_row.get("home_passing_accuracy", "No data"))
    away_acc = safe_float(selected_row.get("away_passing_accuracy", "No data"))
    if home_acc is not None and away_acc is not None:
        st.plotly_chart(plot_bar_chart(home_acc, away_acc, home, away, " Passing Accuracy", "%"))
        charts_shown = True

    if not charts_shown:
        st.warning("No detailed match statistics are available for this match")
