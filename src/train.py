import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mini-projet-mlflow")

# Chargement des données
data = load_iris()
X = data.data
y = data.target

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Exécution MLflow
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)