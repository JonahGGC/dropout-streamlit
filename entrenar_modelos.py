"""
entrenar_modelos.py — reentrena los 3 modelos de la app con las mismas configuraciones del proyecto final
y los guarda como Pipeline completo (preprocesamiento + modelo) en modelos/*.joblib.

Uso:  python entrenar_modelos.py [carpeta_con_train.csv_y_columnas.json]
Verifica al final que las métricas en Test coincidan con resultados/04_test_todos_los_modelos.csv.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from utilidades import CLASES, RANDOM_STATE, GradientBoostingBalanceado

RAIZ = Path(__file__).resolve().parent
ORIGEN = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "datos"

cols = json.load(open(ORIGEN / "columnas.json", encoding="utf-8"))
tr = pd.read_csv(ORIGEN / "train.csv", index_col="idx_original")
te = pd.read_csv(ORIGEN / "test.csv", index_col="idx_original")
X_tr, y_tr = tr.drop(columns="y")[cols["orden"]], tr["y"].to_numpy()
X_te, y_te = te.drop(columns="y")[cols["orden"]], te["y"].to_numpy()
hp = json.load(open(RAIZ / "resultados" / "03_mejores_hiperparametros.json", encoding="utf-8"))


def preprocesador():
    return ColumnTransformer([
        ("num", StandardScaler(), cols["numericas"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), cols["nominales"]),
        ("bin", "passthrough", cols["binarias"]),
    ])


def pipe(modelo):
    return Pipeline([("preprocessor", preprocesador()), ("model", modelo)])


BEST_LR = dict(solver="saga", l1_ratio=1.0, C=0.3, class_weight="balanced")
rf_p = hp["Random Forest"]
gb_p, hgb_p = hp["Gradient Boosting"], hp["HistGradientBoosting"]


def rf_opt():
    return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RANDOM_STATE, **rf_p)


def miembros():
    lr = LogisticRegression(**BEST_LR, max_iter=5000, random_state=RANDOM_STATE)
    svm = CalibratedClassifierCV(SVC(kernel="rbf", C=100, gamma=0.001, class_weight=None, cache_size=1000),
                                 method="sigmoid", cv=5, ensemble=False)
    gb = GradientBoostingBalanceado(random_state=RANDOM_STATE, **gb_p)
    hgb = HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE, **hgb_p)
    return [("lr", lr), ("svm", svm), ("rf", rf_opt()), ("gb", gb), ("hgb", hgb)]


MODELOS = {
    "rl": ("Regresión logística (L1, ajustada)", pipe(LogisticRegression(**BEST_LR, max_iter=5000, random_state=RANDOM_STATE))),
    "rf": ("Random Forest (optimizado)", pipe(rf_opt())),
    "voting": ("Voting soft (optimizado)", pipe(VotingClassifier(miembros(), voting="soft", weights=hp["Voting (soft)"]["weights"]))),
}

# Valores de referencia del proyecto (Test) para comprobar que el reentrenamiento coincide
ref = pd.read_csv(RAIZ / "resultados" / "04_test_todos_los_modelos.csv", index_col=0)
claves_ref = {"rl": "Regresión Logística (F_1)", "rf": "Random Forest optimizado", "voting": "Voting (soft) optimizado"}

(RAIZ / "modelos").mkdir(exist_ok=True)
for clave, (nombre, p) in MODELOS.items():
    t0 = time.perf_counter()
    p.fit(X_tr, y_tr)
    proba = p.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    f1 = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)
    auc = np.mean([roc_auc_score(y_te == k, proba[:, k]) for k in range(3)])
    r = ref.loc[claves_ref[clave]]
    print(f"{nombre:38s} F1 macro {f1:.4f} (proyecto {r['F1']:.4f}) | Acc {acc:.4f} ({r['Accuracy']:.4f}) | "
          f"AUC {auc:.4f} ({r['ROC-AUC']:.4f}) | {time.perf_counter() - t0:.0f} s", flush=True)
    joblib.dump(p, RAIZ / "modelos" / f"{clave}.joblib", compress=3)
