import os
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
