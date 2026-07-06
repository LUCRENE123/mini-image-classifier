import mlflow
import mlflow.sklearn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import normalize_data

# Configuration MLflow
mlflow.set_experiment("Mini-Image-Classifier")

with mlflow.start_run():

    # Chargement dataset
    digits = load_digits()

    X = digits.data
    y = digits.target

    # ❌ DATA SCIENTIST B - modification volontaire (BUG)
    # Suppression d'une colonne importante du dataset
    X = X[:, 1:]   # on supprime la première colonne

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Normalisation
    X_train, X_test = normalize_data(X_train, X_test)

    # Modèle
    model = LogisticRegression(max_iter=2000)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    # Logs MLflow
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 2000)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")