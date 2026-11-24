"""
nucleo.py — lógica de la app sin interfaz: carga de modelos, predicción, consenso, explicación local y lectura de CSV.

Se mantiene separada de app.py para poder probarla (pruebas/) con los mismos datos que usa el modelo en Python.
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import utilidades  # noqa: F401  (el pickle del Voting referencia utilidades.GradientBoostingBalanceado)
from catalogos import (CATALOGOS, ETIQUETAS, cargar_columnas, etiqueta_valor, validar_registro)
from utilidades import CLASES

RAIZ = Path(__file__).resolve().parent

# Cada modelo: archivo, nombre en las tablas del proyecto (resultados/*.csv) y color (paleta categórica validada)
MODELOS_INFO = {
    "rl": dict(nombre="Regresión logística", corto="RL", archivo="rl.joblib", color="#6250d6",
               fila="Regresión Logística (F_1)",
               resumen="Modelo lineal con penalización L1 (C = 0.3) y pesos de clase balanceados. "
                        "Es la línea base interpretable del proyecto."),
    "rf": dict(nombre="Random Forest", corto="RF", archivo="rf.joblib", color="#00907f",
               fila="Random Forest optimizado",
               resumen="300 árboles de profundidad 16 con criterio de entropía, max_features = log2 y pesos "
                        "balanceados; hiperparámetros elegidos con búsqueda aleatoria en validación cruzada."),
    "voting": dict(nombre="Voting soft", corto="Voting", archivo="voting.joblib", color="#b87400",
                   fila="Voting (soft) optimizado",
                   resumen="Promedio ponderado (1, 1, 2, 2, 2) de las probabilidades de cinco modelos: regresión "
                           "logística, SVM RBF calibrado, Random Forest, Gradient Boosting e HistGradientBoosting."),
}
COLOR_MODELOS = {v["nombre"]: v["color"] for v in MODELOS_INFO.values()}
COLOR_CLASES = {"Dropout": "#df5d28", "Enrolled": "#1baf7a", "Graduate": "#2a78d6"}


def cargar_modelos() -> dict:
    """Pipelines completos (preprocesamiento + modelo) guardados con joblib."""
    return {k: joblib.load(RAIZ / "modelos" / v["archivo"]) for k, v in MODELOS_INFO.items()}


def predecir(modelos: dict, X: pd.DataFrame) -> dict:
    """Probabilidades (n, 3) de cada modelo. Las columnas se reordenan como en el entrenamiento."""
    X = X[cargar_columnas()["orden"]]
    return {k: modelos[k].predict_proba(X) for k in MODELOS_INFO}


def consenso(pred_clases: list[int]) -> dict:
    """Resume qué tanto coinciden los tres modelos y qué nivel de atención sugiere."""
    n_drop = sum(c == 0 for c in pred_clases)
    cuenta = pd.Series(pred_clases).value_counts()
    mayoritaria = int(cuenta.idxmax()) if cuenta.iloc[0] >= 2 else None
    if n_drop >= 2:
        nivel, clave = "Alerta de posible abandono", "alerta"
    elif n_drop == 1:
        nivel, clave = "Señal mixta", "mixta"
    else:
        nivel, clave = "Sin alerta de abandono", "ninguna"
    return dict(n_dropout=n_drop, nivel=nivel, clave=clave, mayoritaria=mayoritaria,
                unanime=(cuenta.iloc[0] == 3))


def tabla_predicciones(probas: dict, i: int = 0) -> pd.DataFrame:
    """Una fila por modelo para el registro i: predicción, confianza y probabilidades."""
    filas = []
    for k, info in MODELOS_INFO.items():
        p = probas[k][i]
        c = int(p.argmax())
        filas.append({"Modelo": info["nombre"], "Predicción": CLASES[c], "Confianza": float(p[c]),
                      **{f"P({n})": float(p[j]) for j, n in enumerate(CLASES)}})
    return pd.DataFrame(filas)


def frase_diferencias(tabla: pd.DataFrame) -> str:
    """Describe en una frase si los modelos coinciden o en qué difieren."""
    clases = tabla["Predicción"].tolist()
    if len(set(clases)) == 1:
        return f"Los tres modelos coinciden: **{clases[0]}**."
    grupos = tabla.groupby("Predicción")["Modelo"].apply(list)
    partes = [f"{' y '.join(m)} → **{c}**" for c, m in grupos.items()]
    return "Los modelos no coinciden: " + "; ".join(partes) + "."


# ----------------------------------------------------------------------------------------------------------------
# Explicación local con el modelo logístico (contribución = coeficiente × valor transformado)
# ----------------------------------------------------------------------------------------------------------------
def contribuciones_lr(pipe_lr, fila: pd.DataFrame, clase: int, top: int = 6) -> pd.DataFrame:
    """Qué variables empujan más a favor o en contra de `clase` en el modelo logístico, para una fila.
    Las columnas de one-hot de una misma variable se suman. No incluye el intercepto."""
    cols = cargar_columnas()
    pre, lr = pipe_lr.named_steps["preprocessor"], pipe_lr.named_steps["model"]
    z = pre.transform(fila[cols["orden"]])[0]
    nombres = pre.get_feature_names_out()
    aporte = lr.coef_[clase] * z
    por_var: dict[str, float] = {}
    for n, a in zip(nombres, aporte):
        crudo = n.split("__", 1)[1]
        var = next((v for v in cols["nominales"] if crudo.startswith(v + "_")), crudo)
        por_var[var] = por_var.get(var, 0.0) + float(a)
    filas = [{"variable": v, "etiqueta": ETIQUETAS[v], "valor": etiqueta_valor(v, fila.iloc[0][v]), "aporte": a}
             for v, a in por_var.items() if abs(a) > 1e-9]
    d = pd.DataFrame(filas)
    if d.empty:
        return d
    d["abs"] = d["aporte"].abs()
    return d.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# Archivos CSV
# ----------------------------------------------------------------------------------------------------------------
def plantilla_csv(tipicos: dict) -> bytes:
    cols = cargar_columnas()["orden"]
    return pd.DataFrame([{c: tipicos[c] for c in cols}]).to_csv(index=False).encode("utf-8")


def leer_csv(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV con separador coma o punto y coma (como data.csv del UCI) y limpia los encabezados."""
    texto = contenido.decode("utf-8-sig", errors="replace")
    sep = ";" if texto.splitlines()[0].count(";") > texto.splitlines()[0].count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=sep)
    df.columns = df.columns.str.strip().str.replace("\t", "", regex=False)
    return df.rename(columns={"Nacionality": "Nationality"})


def procesar_csv(df: pd.DataFrame, observados: dict, vistas: dict, max_filas: int = 2000):
    """Valida cada fila. Devuelve (X_validas, tabla_de_problemas, notas_generales)."""
    cols = cargar_columnas()["orden"]
    notas = []
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas en el CSV: " + ", ".join(faltan[:8]) + ("…" if len(faltan) > 8 else ""))
    extras = [c for c in df.columns if c not in cols]
    if extras:
        notas.append("Se ignoraron columnas que el modelo no usa: " + ", ".join(map(str, extras[:6])) + ("…" if len(extras) > 6 else ""))
    if len(df) == 0:
        raise ValueError("El CSV no tiene filas.")
    if len(df) > max_filas:
        notas.append(f"Solo se procesan las primeras {max_filas:,} filas (el archivo trae {len(df):,}).")
        df = df.head(max_filas)
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    problemas, validas = [], []
    for i, fila in num.iterrows():
        d = fila.to_dict()
        errores, avisos = validar_registro(d, observados, vistas)
        for e in errores:
            problemas.append({"Fila": int(i) + 2, "Tipo": "Error", "Detalle": e})
        for a in avisos:
            problemas.append({"Fila": int(i) + 2, "Tipo": "Aviso", "Detalle": a})
        if not errores:
            validas.append(i)
    return num.loc[validas], pd.DataFrame(problemas, columns=["Fila", "Tipo", "Detalle"]), notas
