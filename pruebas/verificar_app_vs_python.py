"""
Verificación «app vs Python directo» (requisito de pruebas de la tarea).

Mismos datos de entrada (los 885 estudiantes del conjunto de prueba), dos caminos:
  (a) Python directo: joblib.load(...).predict_proba(X) en este script, sin tocar el código de la app.
  (b) Camino de la app: cada estudiante se convierte en un diccionario (como lo hace el formulario), pasa por
      nucleo.predecir() y se resume con nucleo.tabla_predicciones() / nucleo.consenso().
Se comparan probabilidades y clases, y las métricas resultantes contra las del proyecto (resultados/04_test_todos_los_modelos.csv).

Uso:   python pruebas/verificar_app_vs_python.py            (todas las filas, ~2-3 min)
       python pruebas/verificar_app_vs_python.py --rapido    (solo 100 filas, para pruebas automáticas)
Escribe resultados/verificacion_app_vs_python.csv, que el Modo developer muestra en su última sección.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import nucleo  # noqa: E402  (después de ajustar sys.path)
from catalogos import cargar_columnas, cargar_particion  # noqa: E402
from nucleo import MODELOS_INFO  # noqa: E402


def verificar(max_filas: int | None = None) -> pd.DataFrame:
    orden = cargar_columnas()["orden"]
    _, _, X_test, y_test = cargar_particion()
    if max_filas:
        X_test, y_test = X_test.iloc[:max_filas], y_test[:max_filas]

    # (a) Python directo: sin pasar por nucleo.py
    directo = {k: joblib.load(RAIZ / "modelos" / v["archivo"]).predict_proba(X_test[orden]) for k, v in MODELOS_INFO.items()}

    # (b) Camino de la app, estudiante por estudiante (como el formulario: dict -> DataFrame de 1 fila)
    modelos = nucleo.cargar_modelos()
    app = {k: np.zeros_like(directo[k]) for k in MODELOS_INFO}
    for i in range(len(X_test)):
        registro = {c: float(X_test.iloc[i][c]) for c in orden}
        probas = nucleo.predecir(modelos, pd.DataFrame([registro]))
        nucleo.tabla_predicciones(probas)      # el mismo resumen que ve el consejero
        for k in MODELOS_INFO:
            app[k][i] = probas[k][0]

    ref = pd.read_csv(RAIZ / "resultados" / "04_test_todos_los_modelos.csv", index_col=0)
    filas = []
    for k, info in MODELOS_INFO.items():
        d, a = directo[k], app[k]
        cd, ca = d.argmax(1), a.argmax(1)
        f1_app = f1_score(y_test, ca, average="macro")
        fila_ref = ref.loc[info["fila"]] if not max_filas else None
        filas.append({
            "Modelo": info["nombre"],
            "Estudiantes": len(X_test),
            "Máx. diferencia de probabilidad (app − Python)": float(np.abs(d - a).max()),
            "Clases distintas": int((cd != ca).sum()),
            "Accuracy (app)": accuracy_score(y_test, ca),
            "F1 macro (app)": f1_app,
            "F1 macro (proyecto)": float(fila_ref["F1"]) if fila_ref is not None else np.nan,
            "Resultado": "Coincide" if (np.abs(d - a).max() < 1e-9 and (cd == ca).all()) else "DIFERENTE",
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="solo 100 filas; no escribe el CSV")
    args = ap.parse_args()
    res = verificar(100 if args.rapido else None)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    if not args.rapido:
        res.to_csv(RAIZ / "resultados" / "verificacion_app_vs_python.csv", index=False)
        print("\nEscrito: resultados/verificacion_app_vs_python.csv")
    sys.exit(0 if (res["Resultado"] == "Coincide").all() else 1)
