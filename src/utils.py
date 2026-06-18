<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler

def normalize_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
=======
# Fonctions utilitaires
from sklearn.datasets import load_digits

def load_data():
    digits = load_digits()

    X = digits.data      # images aplaties (8x8 → 64 features)
    y = digits.target    # labels 0-9

    return X, y
>>>>>>> feature-model-alpha
