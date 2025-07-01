import os
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
