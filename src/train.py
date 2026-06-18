from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import normalize_data

# 1. Charger dataset MNIST (digits)
digits = load_digits()

X = digits.data
y = digits.target

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Normalisation (DIFFÉRENCE AVEC DATA SCIENTIST A)
X_train, X_test = normalize_data(X_train, X_test)

# 4. Modèle (DATA SCIENTIST B)
model = LogisticRegression(max_iter=2000)

# 5. Training
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Accuracy
acc = accuracy_score(y_test, y_pred)

print("===================================")
print("DATA SCIENTIST B MODEL")
print("Logistic Regression + Normalization")
print("Accuracy:", acc)
print("===================================")