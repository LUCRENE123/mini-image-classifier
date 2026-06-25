import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Charger les données (exemple simple)
data = load_iris()
X = data.data
y = data.target

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Définir l'expérience MLflow
mlflow.set_experiment("mini-projet-mlflow")

# 4. RUN MLflow
with mlflow.start_run():

    # 🔥 Modèle avec paramètres modifiés (Étudiant A)
    model = RandomForestClassifier(
        n_estimators=300,      # beaucoup d’arbres
        max_depth=20,          # profondeur élevée
        min_samples_split=2,
        random_state=42
    )

    # 5. Entraînement
    model.fit(X_train, y_train)

    # 6. Prédictions
    y_pred = model.predict(X_test)

    # 7. Accuracy
    acc = accuracy_score(y_test, y_pred)

    # 8. Logs MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("accuracy", acc)

    # 9. Sauvegarde du modèle
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)