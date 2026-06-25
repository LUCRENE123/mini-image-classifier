import mlflow
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import normalize_data

# Charger dataset MNIST (digits)
digits = load_digits()
X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation (si utils fonctionne)
X_train, X_test = normalize_data(X_train, X_test)

# Paramètres du modèle
n_estimators = 250
max_depth = 30
random_state = 60

mlflow.set_experiment("MiniProjet2")

with mlflow.start_run():

    # Log des paramètres
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Modèle
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Log métrique
    mlflow.log_metric("accuracy", acc)

    print("===================================")
    print("FINAL MODEL - Random Forest")
    print("Accuracy:", acc)
    print("===================================")