import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and clean data
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')
df.fillna(0, inplace=True)

df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df = df.loc[:, ~df.columns.duplicated()]

# Prepare X and y
drop_cols = ['winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'result', 'win_type']
X = df.drop(columns=drop_cols)

le = LabelEncoder()
y = le.fit_transform(df['winner'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train LightGBM
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ LightGBM Accuracy: {acc:.4f}")
print("📊 Classification Report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

# Confusion Matrix
cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=le.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance
lgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Feature Importances - LightGBM")
plt.tight_layout()
plt.show()

joblib.dump(model, '../models/pickles/lightgbm.pkl')
