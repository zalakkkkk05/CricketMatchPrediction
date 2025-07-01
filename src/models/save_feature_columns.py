import pandas as pd
import joblib
import os

# Path to your encoded dataset
DATASET_PATH = os.path.join('..', '..', 'Dataset', 'matches_encoded.csv')

# Load the encoded dataset
df = pd.read_csv(DATASET_PATH)

# Drop the target column (adjust if needed)
target_column = 'winner'
if target_column in df.columns:
    feature_columns = df.drop(columns=[target_column]).columns.tolist()
else:
    feature_columns = df.columns.tolist()

# Save the feature columns
FEATURE_COLUMNS_PATH = os.path.join('pickles', 'feature_columns.pkl')
joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")
