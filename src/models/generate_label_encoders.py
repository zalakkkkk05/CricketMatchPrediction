import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Absolute paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Dataset/matches.csv"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pickles/label_encoder.pkl"))

# Load dataset
df = pd.read_csv(DATA_PATH)

# Columns to encode
categorical_columns = ['team1', 'team2', 'venue', 'city', 'toss_winner', 'winner']
label_encoders = {}

# Clean and encode each column
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(' ', '_')  # Replace spaces with underscores
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✅ Encoded {col}: {list(le.classes_)}")

# Save label encoders
joblib.dump(label_encoders, ENCODER_PATH)
print(f"\n✅ All label encoders saved to: {ENCODER_PATH}")
