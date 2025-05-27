import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("full_game_data.csv")

# Encode team names and winner as numerical values
encoder = LabelEncoder()
df['Team_A'] = encoder.fit_transform(df['Team_A'])
df['Team_B'] = encoder.transform(df['Team_B'])
df['Winner'] = encoder.transform(df['Winner'])

# Define features (X) and target variable (y)
X = df[['Team_A', 'Team_B', 'Score_A', 'Score_B']]
y = df['Winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and encoder
import joblib
joblib.dump(model, "game_winner_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
