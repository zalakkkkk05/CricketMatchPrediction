import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 📂 Paths
# ─────────────────────────────────────────────
DATASET_PATH = os.path.join("Dataset", "matches_encoded.csv")
PICKLES_DIR = os.path.join("src", "models", "pickles")
FEATURE_COLUMNS_PATH = os.path.join(PICKLES_DIR, "feature_columns.pkl")
LABEL_ENCODER_PATH = os.path.join(PICKLES_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(PICKLES_DIR, "stacked_ensemble.pkl")  # ✅ matches app.py

# ─────────────────────────────────────────────
# 📥 Load & preprocess data
# ─────────────────────────────────────────────
print("\n🔎 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded with shape: {df.shape}")

df.dropna(inplace=True)
print(f"\n⚠️ Dropped rows with NaNs. New shape: {df.shape}")

# Drop low-sample classes (<2 samples)
target_counts = df['winner'].value_counts()
to_drop = target_counts[target_counts < 2].index.tolist()
if to_drop:
    print(f"⚠️ Dropping classes with <2 samples: {to_drop}")
    df = df[~df['winner'].isin(to_drop)]

# One-hot encode 'win_type' if present
if 'win_type' in df.columns:
    print("\nℹ️ One-hot encoding 'win_type'...")
    win_type_dummies = pd.get_dummies(df['win_type'], prefix='win_type')
    df = pd.concat([df.drop(columns=['win_type']), win_type_dummies], axis=1)
    print(f"✅ Encoded columns: {list(win_type_dummies.columns)}")

# Convert boolean columns to integers
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
print(f"\n🔧 Converted {len(bool_cols)} boolean columns to integers")

# ─────────────────────────────────────────────
# 🧠 Feature & label separation
# ─────────────────────────────────────────────
X = df.drop(columns=[
    'id', 'season', 'date', 'result', 'dl_applied',
    'winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3'
])
y = df['winner']

# Check for non-numeric features
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"\n❌ Found non-numeric features: {non_numeric_cols}")
    print("Exiting. Please fix your feature engineering.")
    exit(1)

# Save feature columns
os.makedirs(PICKLES_DIR, exist_ok=True)
with open(FEATURE_COLUMNS_PATH, 'wb') as f:
    pickle.dump(list(X.columns), f)
print(f"\n✅ Saved {len(X.columns)} feature columns → {FEATURE_COLUMNS_PATH}")

# Label encode the target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✅ Saved label encoder → {LABEL_ENCODER_PATH}")

# ─────────────────────────────────────────────
# 🧪 Train-test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ─────────────────────────────────────────────
# 🏗️ Model setup
# ─────────────────────────────────────────────
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
]
meta_model = LogisticRegression(max_iter=1000, random_state=42)

stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1,
)

# ─────────────────────────────────────────────
# 🚀 Train & evaluate
# ─────────────────────────────────────────────
print("\n🚀 Training stacked ensemble model...")
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained! Test accuracy: {acc:.4f}")

# ─────────────────────────────────────────────
# 💾 Save model
# ─────────────────────────────────────────────
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(stacked_model, f)
print(f"✅ Saved trained stacked model → {MODEL_PATH}")
