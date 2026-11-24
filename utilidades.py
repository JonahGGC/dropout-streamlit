"""
utilidades.py — piezas compartidas por el entrenamiento y la app.

La clase GradientBoostingBalanceado vive aquí (y no en la app) para que joblib pueda reconstruir el modelo
guardado: el pickle guarda la clase por referencia (utilidades.GradientBoostingBalanceado).
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

CLASES = ["Dropout", "Enrolled", "Graduate"]      # códigos 0, 1, 2 (igual que en los notebooks)
RANDOM_STATE = 42


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """GradientBoostingClassifier con pesos de clase 'balanced' calculados dentro de fit() (igual que en config.py)."""

    def fit(self, X, y, sample_weight=None):
        w = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            w = w * sample_weight
        return super().fit(X, y, sample_weight=w)
