import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')
df = df.fillna(0)

# Prepare features and target
drop_cols = ['winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'result', 'win_type']
X = df.drop(columns=drop_cols)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(df['winner'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Accuracy: {accuracy:.4f}\n")

# Classification report (decoded team names)
print("📊 Classification Report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

# Confusion matrix
conf_matrix = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=le.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.show()

# Feature importance
xgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Feature Importances - XGBoost")
plt.tight_layout()
plt.show()

joblib.dump(model, '../models/pickles/xgboost.pkl')
