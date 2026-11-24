"""
catalogos.py — etiquetas en español, catálogos de códigos, valores típicos y validación de entradas.

Los códigos de las variables nominales (carrera, vía de ingreso, escolaridad, ocupación…) son los del diccionario
del dataset UCI 697 (Realinho et al., 2022). Las opciones que se ofrecen en los formularios son únicamente los
códigos que aparecen en el conjunto de entrenamiento: así el modelo recibe categorías que ya conoce.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"

# ----------------------------------------------------------------------------------------------------------------
# Etiquetas de las 36 variables (nombre original del dataset → español)
# ----------------------------------------------------------------------------------------------------------------
ETIQUETAS = {
    "Marital status": "Estado civil",
    "Application mode": "Vía de ingreso",
    "Application order": "Orden de preferencia de la carrera",
    "Course": "Carrera",
    "Daytime/evening attendance": "Turno",
    "Previous qualification": "Estudios previos",
    "Previous qualification (grade)": "Nota de estudios previos (0–200)",
    "Nationality": "Nacionalidad",
    "Mother's qualification": "Escolaridad de la madre",
    "Father's qualification": "Escolaridad del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Nota de admisión (0–200)",
    "Displaced": "Desplazado de su lugar de origen",
    "Educational special needs": "Necesidades educativas especiales",
    "Debtor": "Deudor",
    "Tuition fees up to date": "Colegiatura al corriente",
    "Gender": "Género",
    "Scholarship holder": "Becario",
    "Age at enrollment": "Edad al inscribirse",
    "International": "Estudiante internacional",
    "Curricular units 1st sem (credited)": "1.er sem. · UC acreditadas",
    "Curricular units 1st sem (enrolled)": "1.er sem. · UC inscritas",
    "Curricular units 1st sem (evaluations)": "1.er sem. · evaluaciones presentadas",
    "Curricular units 1st sem (approved)": "1.er sem. · UC aprobadas",
    "Curricular units 1st sem (grade)": "1.er sem. · promedio (0–20)",
    "Curricular units 1st sem (without evaluations)": "1.er sem. · UC sin evaluaciones",
    "Curricular units 2nd sem (credited)": "2.º sem. · UC acreditadas",
    "Curricular units 2nd sem (enrolled)": "2.º sem. · UC inscritas",
    "Curricular units 2nd sem (evaluations)": "2.º sem. · evaluaciones presentadas",
    "Curricular units 2nd sem (approved)": "2.º sem. · UC aprobadas",
    "Curricular units 2nd sem (grade)": "2.º sem. · promedio (0–20)",
    "Curricular units 2nd sem (without evaluations)": "2.º sem. · UC sin evaluaciones",
    "Unemployment rate": "Tasa de desempleo (%)",
    "Inflation rate": "Inflación (%)",
    "GDP": "PIB, variación (%)",
}

AYUDAS = {
    "Application order": "0 = primera opción, 9 = última opción.",
    "Previous qualification (grade)": "Nota con la que concluyó sus estudios previos (escala 0–200).",
    "Admission grade": "Nota de admisión a la carrera (escala 0–200).",
    "Curricular units 1st sem (credited)": "Unidades curriculares (UC = materias) reconocidas por estudios previos.",
    "Curricular units 1st sem (without evaluations)": "UC inscritas a las que no se presentó a ninguna evaluación.",
    "Curricular units 2nd sem (credited)": "Unidades curriculares (UC = materias) reconocidas por estudios previos.",
    "Curricular units 2nd sem (without evaluations)": "UC inscritas a las que no se presentó a ninguna evaluación.",
    "Unemployment rate": "Contexto macroeconómico del año de ingreso.",
    "Inflation rate": "Contexto macroeconómico del año de ingreso.",
    "GDP": "Contexto macroeconómico del año de ingreso.",
}

# ----------------------------------------------------------------------------------------------------------------
# Catálogos de códigos (diccionario UCI 697, traducido)
# ----------------------------------------------------------------------------------------------------------------
_ESCOLARIDAD = {
    1: "Secundaria – 12.º año o equivalente", 2: "Superior – licenciatura (bachelor)", 3: "Superior – grado",
    4: "Superior – maestría", 5: "Superior – doctorado", 6: "Frecuentó educación superior",
    9: "12.º año – incompleto", 10: "11.º año – incompleto", 11: "7.º año (antiguo)", 12: "Otro – 11.º año",
    13: "2.º año del curso complementario de secundaria", 14: "10.º año", 18: "Curso de comercio general",
    19: "Educación básica 3.er ciclo (9.º–11.º) o equivalente", 20: "Curso complementario de secundaria",
    22: "Curso técnico-profesional", 25: "Curso complementario de secundaria – no concluido", 26: "7.º año",
    27: "2.º ciclo del curso general de secundaria", 29: "9.º año – incompleto", 30: "8.º año",
    31: "Curso general de administración y comercio", 33: "Contabilidad y administración suplementaria",
    34: "Desconocida", 35: "No sabe leer ni escribir", 36: "Sabe leer sin haber cursado el 4.º año",
    37: "Educación básica 1.er ciclo (4.º/5.º) o equivalente", 38: "Educación básica 2.º ciclo (6.º–8.º) o equivalente",
    39: "Curso de especialización tecnológica", 40: "Superior – grado (1.er ciclo)",
    41: "Estudios superiores especializados", 42: "Curso técnico superior profesional",
    43: "Superior – maestría (2.º ciclo)", 44: "Superior – doctorado (3.er ciclo)",
}

_OCUPACION = {
    0: "Estudiante", 1: "Poder legislativo y ejecutivo, directores y gerentes",
    2: "Especialistas en actividades intelectuales y científicas", 3: "Técnicos y profesiones de nivel intermedio",
    4: "Personal administrativo", 5: "Servicios personales, seguridad y vendedores",
    6: "Agricultura, pesca y silvicultura (calificados)", 7: "Industria, construcción y oficios (calificados)",
    8: "Operadores de instalaciones y máquinas, ensambladores", 9: "Trabajadores no calificados",
    10: "Fuerzas armadas", 90: "Otra situación", 99: "(en blanco)",
    101: "Oficiales de las fuerzas armadas", 102: "Sargentos de las fuerzas armadas",
    103: "Otro personal de las fuerzas armadas", 112: "Directores de servicios administrativos y comerciales",
    122: "Profesionales de la salud", 123: "Docentes", 125: "Especialistas en TIC",
    131: "Técnicos de ciencias e ingeniería (nivel intermedio)", 132: "Técnicos y profesionales de la salud (nivel intermedio)",
    134: "Técnicos en servicios jurídicos, sociales, deportivos y culturales", 135: "Técnicos de TIC",
    141: "Oficinistas, secretarios y operadores de datos", 143: "Operadores de datos, contabilidad, estadística y registros",
    144: "Otro personal de apoyo administrativo", 151: "Trabajadores de servicios personales", 152: "Vendedores",
    153: "Trabajadores de cuidado personal", 154: "Personal de protección y seguridad",
    161: "Agricultores y agropecuarios calificados (orientados al mercado)",
    163: "Agricultores, ganaderos y pescadores de subsistencia",
    171: "Construcción calificada (excepto electricistas)", 172: "Metalurgia y metalmecánica (calificados)",
    173: "Imprenta, instrumentos de precisión, joyería y artesanía", 175: "Alimentos, carpintería, confección y otros oficios",
    181: "Operadores de plantas fijas y máquinas", 182: "Ensambladores", 183: "Conductores de vehículos y equipo móvil",
    191: "Personal de limpieza", 192: "No calificados en agricultura, ganadería, pesca y silvicultura",
    193: "No calificados en minería, construcción, manufactura y transporte", 194: "Auxiliares de preparación de alimentos",
    195: "Vendedores ambulantes y servicios en la calle",
}

CATALOGOS: dict[str, dict[int, str]] = {
    "Marital status": {1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a", 4: "Divorciado/a", 5: "Unión libre",
                       6: "Separado/a legalmente"},
    "Application mode": {
        1: "1.ª fase – contingente general", 2: "Ordenanza 612/93", 5: "1.ª fase – contingente especial (Azores)",
        7: "Titulares de otros cursos superiores", 10: "Ordenanza 854-B/99", 15: "Estudiante internacional (licenciatura)",
        16: "1.ª fase – contingente especial (Madeira)", 17: "2.ª fase – contingente general",
        18: "3.ª fase – contingente general", 26: "Ordenanza 533-A/99, b2 (otro plan)",
        27: "Ordenanza 533-A/99, b3 (otra institución)", 39: "Mayores de 23 años", 42: "Transferencia",
        43: "Cambio de carrera", 44: "Diplomados en especialización tecnológica", 51: "Cambio de institución/carrera",
        53: "Diplomados de ciclo corto", 57: "Cambio de institución/carrera (internacional)"},
    "Course": {
        33: "Tecnologías de producción de biocombustibles", 171: "Diseño de animación y multimedia",
        8014: "Servicio social (nocturno)", 9003: "Agronomía", 9070: "Diseño de comunicación",
        9085: "Enfermería veterinaria", 9119: "Ingeniería informática", 9130: "Equinocultura", 9147: "Administración",
        9238: "Servicio social", 9254: "Turismo", 9500: "Enfermería", 9556: "Higiene oral",
        9670: "Gestión de publicidad y marketing", 9773: "Periodismo y comunicación", 9853: "Educación básica",
        9991: "Administración (nocturno)"},
    "Previous qualification": {
        1: "Educación secundaria", 2: "Superior – licenciatura (bachelor)", 3: "Superior – grado",
        4: "Superior – maestría", 5: "Superior – doctorado", 6: "Frecuentó educación superior",
        9: "12.º año – incompleto", 10: "11.º año – incompleto", 12: "Otro – 11.º año", 14: "10.º año",
        15: "10.º año – incompleto", 19: "Educación básica 3.er ciclo (9.º–11.º) o equivalente",
        38: "Educación básica 2.º ciclo (6.º–8.º) o equivalente", 39: "Curso de especialización tecnológica",
        40: "Superior – grado (1.er ciclo)", 42: "Curso técnico superior profesional",
        43: "Superior – maestría (2.º ciclo)"},
    "Nationality": {
        1: "Portuguesa", 2: "Alemana", 6: "Española", 11: "Italiana", 13: "Neerlandesa", 14: "Inglesa", 17: "Lituana",
        21: "Angoleña", 22: "Caboverdiana", 24: "Guineana", 25: "Mozambiqueña", 26: "Santotomense", 32: "Turca",
        41: "Brasileña", 62: "Rumana", 100: "Moldava", 101: "Mexicana", 103: "Ucraniana", 105: "Rusa", 108: "Cubana",
        109: "Colombiana"},
    "Mother's qualification": _ESCOLARIDAD,
    "Father's qualification": _ESCOLARIDAD,
    "Mother's occupation": _OCUPACION,
    "Father's occupation": _OCUPACION,
}

# Variables binarias: (etiqueta del 0, etiqueta del 1)
BINARIAS = {
    "Daytime/evening attendance": ("Nocturno", "Diurno"),
    "Displaced": ("No", "Sí"),
    "Educational special needs": ("No", "Sí"),
    "Debtor": ("No", "Sí"),
    "Tuition fees up to date": ("No", "Sí"),
    "Gender": ("Mujer", "Hombre"),
    "Scholarship holder": ("No", "Sí"),
    "International": ("No", "Sí"),
}

# Límites duros de captura (más amplios que lo observado: lo observado solo genera advertencia)
LIMITES = {
    "Application order": (0, 9), "Previous qualification (grade)": (0.0, 200.0), "Admission grade": (0.0, 200.0),
    "Age at enrollment": (16, 80),
    "Unemployment rate": (0.0, 40.0), "Inflation rate": (-10.0, 30.0), "GDP": (-20.0, 20.0),
}
for _s in ("1st", "2nd"):
    for _k in ("credited", "enrolled", "evaluations", "approved", "without evaluations"):
        LIMITES[f"Curricular units {_s} sem ({_k})"] = (0, 60)
    LIMITES[f"Curricular units {_s} sem (grade)"] = (0.0, 20.0)

# Campos del modo simplificado (los otros 22 se llenan con valores típicos del Train)
CAMPOS_SIMPLE = [
    "Course", "Age at enrollment", "Scholarship holder", "Tuition fees up to date", "Debtor", "Admission grade",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
]

# Secciones del modo avanzado
SECCIONES_AVANZADO = {
    "Datos personales": ["Marital status", "Gender", "Age at enrollment", "Nationality", "International", "Displaced",
                         "Educational special needs"],
    "Familia": ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"],
    "Ingreso": ["Course", "Daytime/evening attendance", "Application mode", "Application order",
                "Previous qualification", "Previous qualification (grade)", "Admission grade"],
    "Situación económica": ["Tuition fees up to date", "Debtor", "Scholarship holder"],
    "1.er semestre": ["Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
                      "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
                      "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"],
    "2.º semestre": ["Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
                     "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
                     "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"],
    "Contexto macroeconómico": ["Unemployment rate", "Inflation rate", "GDP"],
}


# ----------------------------------------------------------------------------------------------------------------
# Datos de referencia (Train/Test, columnas)
# ----------------------------------------------------------------------------------------------------------------
def cargar_columnas() -> dict:
    with open(DATOS / "columnas.json", encoding="utf-8") as f:
        return json.load(f)


def cargar_particion():
    """X_train, y_train, X_test, y_test con las columnas en el orden del entrenamiento."""
    orden = cargar_columnas()["orden"]
    tr = pd.read_csv(DATOS / "train.csv", index_col="idx_original")
    te = pd.read_csv(DATOS / "test.csv", index_col="idx_original")
    return tr.drop(columns="y")[orden], tr["y"].to_numpy(), te.drop(columns="y")[orden], te["y"].to_numpy()


def valores_tipicos(X_train: pd.DataFrame) -> dict:
    """Mediana (numéricas) o moda (nominales y binarias) del Train: el «estudiante típico»."""
    cols = cargar_columnas()
    tip = {}
    for c in X_train.columns:
        tip[c] = X_train[c].mode().iloc[0] if c in cols["nominales"] + cols["binarias"] else X_train[c].median()
        tip[c] = tip[c].item() if hasattr(tip[c], "item") else tip[c]
    return tip


def rangos_observados(X_train: pd.DataFrame) -> dict:
    return {c: (float(X_train[c].min()), float(X_train[c].max())) for c in cargar_columnas()["numericas"]}


def categorias_vistas(X_train: pd.DataFrame) -> dict:
    return {c: sorted(int(v) for v in X_train[c].unique()) for c in cargar_columnas()["nominales"]}


def etiqueta_valor(col: str, valor) -> str:
    """Texto legible de un valor (para tablas y explicaciones)."""
    if col in CATALOGOS:
        return CATALOGOS[col].get(int(valor), f"código {int(valor)}")
    if col in BINARIAS:
        return BINARIAS[col][int(valor)]
    if isinstance(valor, (float, np.floating)) and not float(valor).is_integer():
        return f"{valor:.2f}"
    return str(int(valor)) if float(valor).is_integer() else str(valor)


# ----------------------------------------------------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------------------------------------------------
def validar_registro(d: dict, observados: dict, vistas: dict) -> tuple[list[str], list[str]]:
    """Revisa un registro (dict con las 36 variables). Devuelve (errores, avisos).
    Errores: impiden predecir. Avisos: se predice, pero el dato queda fuera de lo que vio el modelo."""
    errores, avisos = [], []
    cols = cargar_columnas()
    for c in cols["orden"]:
        v = d.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errores.append(f"Falta el valor de «{ETIQUETAS[c]}».")
    if errores:
        return errores, avisos

    for c, (lo, hi) in LIMITES.items():
        if not (lo <= float(d[c]) <= hi):
            errores.append(f"«{ETIQUETAS[c]}» debe estar entre {lo:g} y {hi:g} (se recibió {d[c]}).")
    for c in cols["binarias"]:
        if int(d[c]) not in (0, 1):
            errores.append(f"«{ETIQUETAS[c]}» debe ser 0 o 1 (se recibió {d[c]}).")
    for c in cols["nominales"]:
        if float(d[c]) != int(d[c]):
            errores.append(f"«{ETIQUETAS[c]}» debe ser un código entero (se recibió {d[c]}).")
        elif int(d[c]) not in vistas[c]:
            avisos.append(f"«{ETIQUETAS[c]}»: el código {int(d[c])} no aparece en el entrenamiento; "
                          "el modelo lo trata como categoría poco frecuente.")
    for c in cols["numericas"]:
        if c.startswith("Curricular units") and "grade" not in c and float(d[c]) != int(d[c]):
            errores.append(f"«{ETIQUETAS[c]}» debe ser un número entero de unidades curriculares.")

    # coherencia entre las unidades curriculares de cada semestre (se cumple en el 100 % del Train)
    for s, nombre in (("1st", "1.er"), ("2nd", "2.º")):
        env, ev = d[f"Curricular units {s} sem (enrolled)"], d[f"Curricular units {s} sem (evaluations)"]
        ap, nota = d[f"Curricular units {s} sem (approved)"], d[f"Curricular units {s} sem (grade)"]
        if ap > env:
            errores.append(f"{nombre} semestre: las UC aprobadas ({ap:g}) no pueden superar las inscritas ({env:g}).")
        if ap > ev:
            errores.append(f"{nombre} semestre: las UC aprobadas ({ap:g}) no pueden superar las evaluaciones presentadas ({ev:g}).")
        if ap == 0 and nota > 0:
            errores.append(f"{nombre} semestre: sin UC aprobadas el promedio debe ser 0 (se capturó {nota:g}).")
        if ap > 0 and nota == 0:
            errores.append(f"{nombre} semestre: con UC aprobadas el promedio no puede ser 0.")

    for c, (lo, hi) in observados.items():
        if lo <= float(d[c]) <= hi:
            continue
        if c in LIMITES:
            avisos.append(f"«{ETIQUETAS[c]}» = {d[c]:g} está fuera del rango visto en el entrenamiento "
                          f"({lo:g}–{hi:g}); la predicción es una extrapolación.")
    return errores, avisos
