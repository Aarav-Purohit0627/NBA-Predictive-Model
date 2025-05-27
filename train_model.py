from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# Load datasets
games_df = pd.read_csv("full_game_results.csv")
players_df = pd.read_csv("player_stats.csv")
team_stats_df = pd.read_csv("team_advanced_stats.csv")  # Contains Off/Def Ratings, Win-Loss %, etc.

# Initialize LabelEncoder
le = LabelEncoder()
all_teams = pd.concat([games_df["Team_A"], games_df["Team_B"]]).unique()
le.fit(all_teams)

# Encode teams
games_df["team_A_encoded"] = le.transform(games_df["Team_A"])
games_df["team_B_encoded"] = le.transform(games_df["Team_B"])
games_df["winner_encoded"] = le.transform(games_df["Winner"])

# Compute team-level player averages (Player Efficiency Rating - PER)
players_df["PER"] = (players_df["Avg_Points"] * 0.4)  # Simplified PER formula (can be improved)
team_avg_per = players_df.groupby("Team")["PER"].mean().reset_index()
team_avg_per.rename(columns={"PER": "Avg_PER"}, inplace=True)

# Merge advanced stats (Off/Def Ratings, Win %, Home Advantage)
games_df = games_df.merge(team_stats_df, left_on="Team_A", right_on="Team", how="left").drop(columns=["Team"])
games_df = games_df.merge(team_stats_df, left_on="Team_B", right_on="Team", how="left", suffixes=("_A", "_B")).drop(columns=["Team"])

# Merge PER
games_df = games_df.merge(team_avg_per, left_on="Team_A", right_on="Team", how="left").drop(columns=["Team"])
games_df = games_df.merge(team_avg_per, left_on="Team_B", right_on="Team", how="left", suffixes=("_A", "_B")).drop(columns=["Team"])

# Define features and target
X = games_df[["team_A_encoded", "team_B_encoded", "Off_Rating_A", "Def_Rating_A", "Win_Perc_A", "Home_Adv_A",
              "Off_Rating_B", "Def_Rating_B", "Win_Perc_B", "Home_Adv_B", "Avg_PER_A", "Avg_PER_B"]]
y = games_df["winner_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoder
with open("game_winner_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(le, encoder_file)

print("Model and label encoder saved successfully with advanced features!")
