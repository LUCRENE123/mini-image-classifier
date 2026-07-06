from sklearn.datasets import load_iris

def test_dataset_integrity():
    data = load_iris()
    X = data.data
    y = data.target

    # Vérifie nombre de colonnes
    assert X.shape[1] == 4

    # Vérifie qu'il n'y a pas de valeurs vides
    assert not (X == None).any()

    # Vérifie que la cible n'est pas vide
    assert len(y) > 0