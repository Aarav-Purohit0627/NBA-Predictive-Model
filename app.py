from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load trained model and encoder
model = pickle.load(open("game_winner_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load player stats & team advanced stats
players_df = pd.read_csv("player_stats.csv")
team_stats_df = pd.read_csv("team_advanced_stats.csv")

# Define NBA team colors (hex values)
team_colors = {  # Define team colors
    "Atlanta Hawks": "#E03A3E", "Boston Celtics": "#007A33", "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160", "Chicago Bulls": "#CE1141", "Cleveland Cavaliers": "#6F263D",
    "Dallas Mavericks": "#00538C", "Denver Nuggets": "#0E2240", "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A", "Houston Rockets": "#CE1141", "Indiana Pacers": "#002D62",
    "Los Angeles Clippers": "#C8102E", "Los Angeles Lakers": "#552583", "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E", "Milwaukee Bucks": "#00471B", "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#85714D", "New York Knicks": "#006BB6", "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0", "Philadelphia 76ers": "#006BB6", "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E", "Sacramento Kings": "#5A2D81", "San Antonio Spurs": "#C4CED4",
    "Toronto Raptors": "#CE1141", "Utah Jazz": "#002B5C", "Washington Wizards": "#002B5C"
}

def get_team_players(team):
    """Retrieve top 5 players & their PPG for a team."""
    team_players = players_df[players_df["Team"] == team][["Player", "Avg_Points"]]
    return list(team_players.itertuples(index=False, name=None))[:5]  # Top 5 players

def get_team_stats(team):
    """Retrieve advanced stats for a team."""
    stats = team_stats_df[team_stats_df["Team"] == team]
    if not stats.empty:
        return stats.iloc[0].to_dict()  # Convert row to dictionary
    return {"Off_Rating": "N/A", "Def_Rating": "N/A", "Win_Perc": "N/A", "Home_Adv": "N/A"}  # Default if missing

def get_team_PER(team):
    """Retrieve the team's average Player Efficiency Rating (PER)."""
    team_per = players_df[players_df["Team"] == team]["Avg_Points"].mean() * 0.4  # Simplified PER formula
    return round(team_per, 2) if not pd.isna(team_per) else 0.0  # Default to 0 if missing

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    teams = list(label_encoder.classes_)

    team_A, team_B = None, None
    team_A_color, team_B_color = "#cccccc", "#cccccc"
    team_A_players, team_B_players = [], []

    # Default advanced stats
    off_rating_A = def_rating_A = win_perc_A = home_adv_A = "N/A"
    off_rating_B = def_rating_B = win_perc_B = home_adv_B = "N/A"
    avg_per_A = avg_per_B = 0.0  # Ensure PER is included

    if request.method == "POST":
        team_A = request.form["team_A"]
        team_B = request.form["team_B"]

        if team_A not in teams or team_B not in teams:
            prediction = "Invalid team name(s). Please choose valid teams."
        else:
            # Encode teams
            team_A_encoded = label_encoder.transform([team_A])[0]
            team_B_encoded = label_encoder.transform([team_B])[0]

            # Get team colors
            team_A_color = team_colors.get(team_A, "#cccccc")
            team_B_color = team_colors.get(team_B, "#cccccc")

            # Get top 5 players & PPG for each team
            team_A_players = get_team_players(team_A)
            team_B_players = get_team_players(team_B)

            # Get advanced stats
            team_A_stats = get_team_stats(team_A)
            team_B_stats = get_team_stats(team_B)

            # Assign advanced stats
            off_rating_A, def_rating_A, win_perc_A, home_adv_A = team_A_stats["Off_Rating"], team_A_stats["Def_Rating"], team_A_stats["Win_Perc"], team_A_stats["Home_Adv"]
            off_rating_B, def_rating_B, win_perc_B, home_adv_B = team_B_stats["Off_Rating"], team_B_stats["Def_Rating"], team_B_stats["Win_Perc"], team_B_stats["Home_Adv"]

            # Get PER for both teams
            avg_per_A = get_team_PER(team_A)
            avg_per_B = get_team_PER(team_B)

            # Prepare input data (now includes `Avg_PER_A` & `Avg_PER_B`)
            input_data = pd.DataFrame([[team_A_encoded, team_B_encoded, off_rating_A, def_rating_A, win_perc_A, home_adv_A,
                                        off_rating_B, def_rating_B, win_perc_B, home_adv_B, avg_per_A, avg_per_B]])

            # Make prediction
            predicted_winner_encoded = model.predict(input_data)[0]
            predicted_winner = label_encoder.inverse_transform([predicted_winner_encoded])[0]

            prediction = f"Predicted Winner: {predicted_winner}"

    return render_template(
        "index.html",
        prediction=prediction,
        teams=teams,
        team_A=team_A,
        team_B=team_B,
        team_A_color=team_A_color,
        team_B_color=team_B_color,
        team_A_players=team_A_players,
        team_B_players=team_B_players,
        off_rating_A=off_rating_A, def_rating_A=def_rating_A, win_perc_A=win_perc_A, home_adv_A=home_adv_A,
        off_rating_B=off_rating_B, def_rating_B=def_rating_B, win_perc_B=win_perc_B, home_adv_B=home_adv_B,
        avg_per_A=avg_per_A, avg_per_B=avg_per_B
    )

if __name__ == "__main__":
    app.run(debug=True)
