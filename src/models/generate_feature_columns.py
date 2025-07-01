# generate_feature_columns.py

import pandas as pd
import joblib
import os

# Load the final encoded dataset used for training
df = pd.read_csv('../../Dataset/matches_encoded.csv')

# Drop target column
X = df.drop(columns=['winner'])

# Define output path
FEATURE_COLUMNS_PATH = os.path.join('pickles', 'feature_columns.pkl')

# Save correct feature columns
joblib.dump(X.columns.tolist(), FEATURE_COLUMNS_PATH)
print(f"✅ Feature columns saved to {FEATURE_COLUMNS_PATH}")
