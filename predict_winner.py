import joblib
import pandas as pd
import shap
import numpy as np

# Load trained model and encoder
model = joblib.load("game_winner_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load team advanced stats
team_stats_df = pd.read_csv("team_advanced_stats.csv")
players_df = pd.read_csv("player_stats.csv")

def get_team_features(team):
    """Retrieve team stats: Off/Def Ratings, Win %, Home Advantage, and Avg PER."""
    team_stats = team_stats_df[team_stats_df["Team"] == team]
    if team_stats.empty:
        return [0] * 4  # If no data, return zeroed features
    
    off_rating = team_stats["Off_Rating"].values[0]
    def_rating = team_stats["Def_Rating"].values[0]
    win_perc = team_stats["Win_Perc"].values[0]
    home_adv = team_stats["Home_Adv"].values[0]

    return [off_rating, def_rating, win_perc, home_adv]

def get_team_PER(team):
    """Retrieve team average Player Efficiency Rating (PER)."""
    team_per = players_df[players_df["Team"] == team]["Avg_Points"].mean()
    return team_per if not np.isnan(team_per) else 0  # Default to 0 if no data

def predict_winner(team_a, team_b):
    try:
        if team_a not in encoder.classes_ or team_b not in encoder.classes_:
            return "Error: One or both teams are unknown to the model."

        team_a_encoded = encoder.transform([team_a])[0]
        team_b_encoded = encoder.transform([team_b])[0]

        # Get team advanced stats
        team_a_features = get_team_features(team_a)
        team_b_features = get_team_features(team_b)

        # Get Player Efficiency Ratings
        team_a_PER = get_team_PER(team_a)
        team_b_PER = get_team_PER(team_b)

        # Create feature input for model
        input_data = pd.DataFrame([[team_a_encoded, team_b_encoded] + team_a_features + team_b_features + [team_a_PER, team_b_PER]], 
                                  columns=["team_A_encoded", "team_B_encoded", "Off_Rating_A", "Def_Rating_A", "Win_Perc_A", "Home_Adv_A",
                                           "Off_Rating_B", "Def_Rating_B", "Win_Perc_B", "Home_Adv_B", "Avg_PER_A", "Avg_PER_B"])
        
        # Make prediction
        winner_encoded = model.predict(input_data)[0]
        winner = encoder.inverse_transform([winner_encoded])[0]

        # SHAP explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Display feature importance
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1], input_data.iloc[0], matplotlib=True).show()

        return f"Predicted Winner: {winner}"

    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    print(predict_winner("Chicago Bulls", "Los Angeles Lakers"))
