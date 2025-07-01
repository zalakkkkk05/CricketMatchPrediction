import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ✅ Load data (updated path)
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')

# ✅ Drop unnecessary columns
drop_cols = ['id', 'season', 'date', 'result', 'dl_applied', 'winner',
             'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'win_type']
X = df.drop(columns=drop_cols)
y = df['winner']

# ✅ Handle missing values
X = X.fillna(0)

# ✅ Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ✅ Train CatBoost model
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train, y_train)

# ✅ Predict
y_pred = model.predict(X_test)

# ✅ Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ CatBoost Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ✅ Confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("CatBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# ✅ Save confusion matrix image
plt.savefig('D:/CricketMatchPrediction/pictures/Figure_5_catboost_confusion_matrix.png')
plt.show()

joblib.dump(model, '../models/pickles/catboost.pkl')
