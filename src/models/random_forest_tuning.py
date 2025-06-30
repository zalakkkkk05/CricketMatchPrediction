import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')
df = df.fillna(0)

# Prepare features and target
drop_cols = ['winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'result', 'win_type']
X = df.drop(columns=drop_cols)
y = df['winner']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Set up parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize RandomForest
rf = RandomForestClassifier(random_state=42)

# Set up GridSearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

# Run grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("✅ Best Parameters:", grid_search.best_params_)

# Best estimator predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Tuned Random Forest Accuracy: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Create pickles directory if it doesn't exist
os.makedirs('src/models/pickles', exist_ok=True)

joblib.dump(best_rf, '../models/pickles/random_forest.pkl')
