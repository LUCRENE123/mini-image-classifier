# Fonctions utilitaires
from sklearn.datasets import load_digits

def load_data():
    digits = load_digits()

    X = digits.data      # images aplaties (8x8 → 64 features)
    y = digits.target    # labels 0-9

    return X, y