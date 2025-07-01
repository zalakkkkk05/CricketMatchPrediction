import os
<<<<<<< HEAD
import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError

# Paths: build absolute paths to ensure they match your project structure
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PICKLES_DIR = os.path.join(ROOT_DIR, 'src', 'models', 'pickles')
FEATURE_COLUMNS_PATH = os.path.join(PICKLES_DIR, 'feature_columns.pkl')
MODEL_PATH = os.path.join(PICKLES_DIR, 'stacked_model.pkl')
LABEL_ENCODER_PATH = os.path.join(PICKLES_DIR, 'label_encoder.pkl')

# Load feature columns, trained model, and label encoder
with open(FEATURE_COLUMNS_PATH, 'rb') as f:
    feature_columns = pickle.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Dummy value for numeric features
DUMMY_NUMERIC = 0.5

# Example input: change these for your match
team1 = "Mumbai Indians"
team2 = "Royal Challengers Bangalore"
city = "Mumbai"
venue = "Wankhede Stadium"
toss_winner = "Mumbai Indians"
toss_decision = "field"  # or "bat"
win_type = "chase_win"   # new: must match one-hot encoded win_type columns

# Create a DataFrame for one sample with all features initialized to 0
input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

# Set numeric features with dummy values
input_df.loc[0, [
    "team1_form_last5", "team2_form_last5", "team1_head_to_head_win_ratio",
    "team1_venue_win_ratio", "team1_home_win_ratio", "team1_toss_win_match_win_ratio",
    "team1_total_impact_score", "team2_total_impact_score"
]] = DUMMY_NUMERIC

# One-hot features to set, matching the column names in your dataset
features_to_set = [
    f"team1_{team1}", f"team2_{team2}", f"city_{city}", f"venue_{venue}",
    f"toss_winner_{toss_winner}", f"toss_decision_{toss_decision}", f"win_type_{win_type}"
]

print("\n🔎 Attempting to set these features:")
for feat in features_to_set:
    if feat in input_df.columns:
        input_df.at[0, feat] = 1
        print(f" ✅ Set {feat}")
    else:
        print(f" ⚠️ WARNING: Feature {feat} not found in feature columns!")

# Show non-zero features for verification
non_zero = input_df.loc[:, input_df.iloc[0] != 0]
print("\n🔍 Non-zero input features:")
print(non_zero)

# Make prediction
try:
    pred_encoded = model.predict(input_df)[0]
    predicted_team = label_encoder.inverse_transform([pred_encoded])[0]
    print(f"\n✅ Predicted Match Winner: {predicted_team}")
except NotFittedError:
    print("\n❌ Model is not fitted. Please train the model first.")
except Exception as e:
    print(f"\n❌ Prediction failed with error: {e}")
=======
import pandas as pd
import joblib

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(CURRENT_DIR, "pickles")
MODEL_PATH = os.path.join(PICKLE_DIR, "stacked_ensemble.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PICKLE_DIR, "feature_columns.pkl")

# --- Load model and feature columns ---
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

# --- Input Preparer ---
def prepare_input(team1, team2, venue, city):
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  # Fill with zeros

    def encode_and_flag(prefix, value):
        col = f"{prefix}_{value}"
        if col in input_df.columns:
            input_df.at[0, col] = 1
        else:
            print(f"⚠️ MISSING COLUMN in prediction: {col}")

    # One-hot encode applicable fields
    encode_and_flag('team1', team1)
    encode_and_flag('team2', team2)
    encode_and_flag('venue', venue)
    encode_and_flag('toss_winner', team1)  # Dummy toss winner
    encode_and_flag('city', city)
    input_df['toss_decision_field'] = 1  # Assume fielding

    # Fill numeric features with dummy values
    for feature in [
        'team1_form_last5', 'team2_form_last5',
        'team1_head_to_head_win_ratio', 'team1_venue_win_ratio',
        'team1_home_win_ratio', 'team1_toss_win_match_win_ratio',
        'team1_total_impact_score', 'team2_total_impact_score'
    ]:
        if feature in input_df.columns:
            input_df.at[0, feature] = float(0.5)
        else:
            print(f"⚠️ MISSING NUMERIC FEATURE: {feature}")

    print("\n🔍 Non-zero input values:")
    print(input_df.loc[:, (input_df != 0).any(axis=0)])
    return input_df

# --- Main prediction function ---
def predict_match_winner(team1, team2, venue, city):
    input_df = prepare_input(team1, team2, venue, city)
    prediction = model.predict(input_df)[0]
    print(f"\n🏏 Predicted Winner: {prediction}")
    return prediction

# --- Example Run ---
if __name__ == "__main__":
    team1 = "Mumbai_Indians"
    team2 = "Lucknow_Super_Giants"
    venue = "Wankhede_Stadium"
    city = "Mumbai"

    predict_match_winner(team1, team2, venue, city)
>>>>>>> 354d432415c61d7dc5d79c61012f8e81f231c9eb
