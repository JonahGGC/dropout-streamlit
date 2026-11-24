"""
Pruebas automáticas de la app (pytest). Se corren desde la raíz del repositorio:

    python -m pytest pruebas -q

La comparación completa «app vs Python» sobre los 885 casos de Test está en verificar_app_vs_python.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import nucleo  # noqa: E402
from catalogos import (cargar_columnas, cargar_particion, categorias_vistas, rangos_observados,  # noqa: E402
                       valores_tipicos, validar_registro)
from utilidades import CLASES  # noqa: E402


@pytest.fixture(scope="module")
def datos():
    X_tr, y_tr, X_te, y_te = cargar_particion()
    return dict(X_tr=X_tr, X_te=X_te, y_te=y_te, tip=valores_tipicos(X_tr),
                obs=rangos_observados(X_tr), vistas=categorias_vistas(X_tr))


@pytest.fixture(scope="module")
def modelos():
    return nucleo.cargar_modelos()


# ---- modelos y predicción --------------------------------------------------------------------------------------
def test_los_tres_modelos_son_pipelines_completos(modelos):
    assert set(modelos) == {"rl", "rf", "voting"}
    for m in modelos.values():
        assert list(m.named_steps) == ["preprocessor", "model"]


def test_particion_y_columnas(datos):
    assert datos["X_te"].shape == (885, 36)
    assert list(datos["X_te"].columns) == cargar_columnas()["orden"]


def test_probabilidades_validas_y_clase_es_argmax(datos, modelos):
    X = datos["X_te"].iloc[:60]
    for k, p in nucleo.predecir(modelos, X).items():
        assert p.shape == (60, 3)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert (p >= 0).all() and (p <= 1).all()
        assert (modelos[k].predict(X) == p.argmax(axis=1)).all(), f"{k}: predict() no coincide con argmax(predict_proba)"


def test_el_orden_de_columnas_no_importa(datos, modelos):
    X = datos["X_te"].iloc[:20]
    a = nucleo.predecir(modelos, X)
    b = nucleo.predecir(modelos, X[X.columns[::-1]])
    for k in a:
        assert np.allclose(a[k], b[k])


def test_app_igual_a_python_directo(datos, modelos):
    """Misma entrada, dos caminos: Pipeline.predict_proba directo vs funciones de la app (fila por fila)."""
    orden = cargar_columnas()["orden"]
    X = datos["X_te"].iloc[:40]
    directo = {k: m.predict_proba(X[orden]) for k, m in modelos.items()}
    for i in range(len(X)):
        p = nucleo.predecir(modelos, pd.DataFrame([{c: float(X.iloc[i][c]) for c in orden}]))
        for k in directo:
            assert np.allclose(p[k][0], directo[k][i], atol=1e-9)


def test_metricas_de_test_reproducen_las_del_proyecto(datos, modelos):
    from sklearn.metrics import f1_score
    ref = pd.read_csv(RAIZ / "resultados" / "04_test_todos_los_modelos.csv", index_col=0)
    for k, info in nucleo.MODELOS_INFO.items():
        pred = modelos[k].predict(datos["X_te"])
        assert f1_score(datos["y_te"], pred, average="macro") == pytest.approx(ref.loc[info["fila"], "F1"], abs=1e-6)


# ---- consenso y textos -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("clases,clave", [([0, 0, 2], "alerta"), ([0, 0, 0], "alerta"), ([0, 1, 2], "mixta"),
                                           ([1, 2, 2], "ninguna"), ([2, 2, 2], "ninguna")])
def test_semaforo_de_consenso(clases, clave):
    assert nucleo.consenso(clases)["clave"] == clave


def test_tabla_y_frase_de_diferencias(datos, modelos):
    p = nucleo.predecir(modelos, datos["X_te"].iloc[:1])
    t = nucleo.tabla_predicciones(p)
    assert list(t["Modelo"]) == [v["nombre"] for v in nucleo.MODELOS_INFO.values()]
    assert set(t["Predicción"]) <= set(CLASES)
    assert np.allclose(t[["P(Dropout)", "P(Enrolled)", "P(Graduate)"]].sum(axis=1), 1.0)
    assert t["Confianza"].between(1 / 3, 1).all()
    assert "coinciden" in nucleo.frase_diferencias(t) or "no coinciden" in nucleo.frase_diferencias(t)


def test_explicacion_local_del_modelo_logistico(datos, modelos):
    c = nucleo.contribuciones_lr(modelos["rl"], datos["X_te"].iloc[[0]], clase=0)
    assert 0 < len(c) <= 6 and {"etiqueta", "valor", "aporte"} <= set(c.columns)


# ---- validación de entradas ------------------------------------------------------------------------------------
def _reg(datos, **cambios):
    d = dict(datos["tip"])
    d.update(cambios)
    return validar_registro(d, datos["obs"], datos["vistas"])


def test_estudiante_tipico_es_valido(datos):
    errores, _ = _reg(datos)
    assert errores == []


@pytest.mark.parametrize("cambios,fragmento", [
    ({"Age at enrollment": 5}, "entre"),
    ({"Admission grade": 250}, "entre"),
    ({"Curricular units 1st sem (grade)": 25}, "entre"),
    ({"Gender": 3}, "0 o 1"),
    ({"Curricular units 1st sem (approved)": 9, "Curricular units 1st sem (enrolled)": 6, "Curricular units 1st sem (evaluations)": 10}, "no pueden superar"),
    ({"Curricular units 2nd sem (approved)": 0, "Curricular units 2nd sem (grade)": 12.0}, "debe ser 0"),
    ({"Curricular units 2nd sem (approved)": 3, "Curricular units 2nd sem (grade)": 0.0}, "no puede ser 0"),
    ({"Course": 9003.5}, "entero"),
])
def test_entradas_invalidas_se_rechazan(datos, cambios, fragmento):
    errores, _ = _reg(datos, **cambios)
    assert any(fragmento in e for e in errores), errores


def test_campo_vacio_se_rechaza(datos):
    errores, _ = _reg(datos, **{"Age at enrollment": float("nan")})
    assert errores and "Falta" in errores[0]


def test_categoria_nunca_vista_solo_avisa(datos):
    errores, avisos = _reg(datos, **{"Course": 9999})
    assert errores == [] and any("no aparece" in a for a in avisos)


# ---- CSV -------------------------------------------------------------------------------------------------------
def test_csv_plantilla_ida_y_vuelta(datos):
    df = nucleo.leer_csv(nucleo.plantilla_csv(datos["tip"]))
    X, problemas, _ = nucleo.procesar_csv(df, datos["obs"], datos["vistas"])
    assert len(X) == 1 and problemas[problemas["Tipo"] == "Error"].empty


def test_csv_con_punto_y_coma_y_encabezado_del_uci(datos):
    orden = cargar_columnas()["orden"]
    fila = pd.DataFrame([datos["tip"]])[orden].rename(columns={"Nationality": "Nacionality"})
    texto = fila.to_csv(index=False, sep=";")
    X, _, _ = nucleo.procesar_csv(nucleo.leer_csv(texto.encode("utf-8")), datos["obs"], datos["vistas"])
    assert len(X) == 1


def test_csv_con_filas_malas_las_separa(datos):
    orden = cargar_columnas()["orden"]
    buena, mala = dict(datos["tip"]), dict(datos["tip"])
    mala["Age at enrollment"] = 3
    df = pd.DataFrame([buena, mala, buena])[orden]
    X, problemas, _ = nucleo.procesar_csv(df, datos["obs"], datos["vistas"])
    assert len(X) == 2
    assert list(problemas.loc[problemas["Tipo"] == "Error", "Fila"]) == [3]   # fila 3 del archivo = 2.ª de datos


def test_csv_sin_columnas_requeridas(datos):
    with pytest.raises(ValueError, match="Faltan columnas"):
        nucleo.procesar_csv(pd.DataFrame({"a": [1]}), datos["obs"], datos["vistas"])


# ---- la interfaz completa arranca sin errores --------------------------------------------------------------------
def test_la_app_arranca_sin_excepciones():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
