import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')
df = df.fillna(0)

# Drop irrelevant/non-numeric columns
drop_cols = ['winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'result', 'win_type']
X = df.drop(columns=drop_cols)
y = df['winner']

# ✅ Clean feature names for LightGBM and XGBoost
X.columns = X.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X = X.loc[:, ~X.columns.duplicated()]  # remove duplicated columns if any

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Ensure X_train and X_test are proper DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Load pre-trained models
rf = joblib.load('D:/CricketMatchPrediction/src/models/pickles/random_forest.pkl')
xgb = joblib.load('D:/CricketMatchPrediction/src/models/pickles/xgboost.pkl')
lgb = joblib.load('D:/CricketMatchPrediction/src/models/pickles/lightgbm.pkl')
cat = joblib.load('D:/CricketMatchPrediction/src/models/pickles/catboost.pkl')

# Define base learners
base_learners = [
    ('rf', rf),
    ('xgb', xgb),
    ('lgb', lgb),
    ('cat', cat)
]

# Create stacking ensemble
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

# Fit stacked model
stacked_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Stacked Ensemble Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the stacked model
joblib.dump(stacked_model, 'D:/CricketMatchPrediction/src/models/pickles/stacked_ensemble.pkl')
