from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import load_data

# Charger MNIST (digits)
X, y = load_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle A : Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
acc = accuracy_score(y_test, y_pred)

print("Accuracy Model A (Random Forest MNIST):", acc)