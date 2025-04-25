import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load encoded dataset
df = pd.read_csv('D:/CricketMatchPrediction/Dataset/matches_encoded.csv')

# Handle missing values (choose one)
df = df.fillna(0)  # Recommended
# df = df.dropna()  # Alternative


# 2. Prepare X (features) and y (target)
drop_cols = ['winner', 'player_of_match', 'umpire1', 'umpire2', 'umpire3', 'date', 'result', 'win_type']
X = df.drop(columns=drop_cols)
y = df['winner']


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Optional: Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
