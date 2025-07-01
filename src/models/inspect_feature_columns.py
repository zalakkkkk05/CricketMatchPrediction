import pickle
import os

FEATURE_COLUMNS_PATH = os.path.join(
    os.path.dirname(__file__), "pickles", "feature_columns.pkl"
)

with open(FEATURE_COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)

print(f"✅ Loaded {len(feature_columns)} feature columns:\n")
for col in feature_columns:
    print(col)
