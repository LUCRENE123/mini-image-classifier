from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

# Charger dataset MNIST (digits)
X, y = load_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle final (choix du groupe)
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("===================================")
print("FINAL MODEL - Random Forest")
print("Accuracy:", acc)
print("===================================")