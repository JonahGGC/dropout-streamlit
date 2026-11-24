"""
app.py — Predicción de deserción estudiantil con ensambles (Proyecto final · Machine Learning, MIC1 · UACh).

Tres secciones:
  · Proyecto         información general, base de datos y descripción de los tres modelos.
  · Modo consejero   captura (1 simplificada / 2 avanzada / CSV) y predicción comparada de los tres modelos.
  · Modo developer   estadísticas de rendimiento de los modelos (Test, CV, inferencia) y pruebas con casos reales.

Los modelos (modelos/*.joblib) son Pipelines completos reentrenados con la configuración del proyecto;
las métricas salen de las tablas del experimento (resultados/*.csv) o de evaluar esos modelos sobre el Test.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

import catalogos as cat
import nucleo
from nucleo import COLOR_CLASES, COLOR_MODELOS, MODELOS_INFO
from utilidades import CLASES

RAIZ = Path(__file__).resolve().parent
REPO_URL = ""   # ← pega aquí la URL del repositorio de GitHub (se muestra en la pestaña Proyecto si no está vacía)

st.set_page_config(page_title="Deserción estudiantil · Machine Learning", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.4rem; max-width: 1180px; }
    /* Sin colores fijos de texto ni de fondo: hereda del tema (claro u oscuro) y solo tiñe con el color del estado */
    .caja { background: color-mix(in srgb, var(--c) 9%, transparent); border:1px solid rgba(128,128,128,.30);
            border-left:7px solid var(--c); border-radius:12px; padding:16px 22px; margin:4px 0 14px 0; }
    .caja .t { font-size:1.45rem; font-weight:700; line-height:1.25; }
    .caja .s { font-size:1.02rem; opacity:.85; margin-top:4px; }
    .modelo { border-top:5px solid var(--c); }
    .peq { font-size:0.9rem; opacity:.72; }
    </style>
    """,
    unsafe_allow_html=True,
)

def oscuro() -> bool:
    """True si el tema activo es el oscuro (Streamlit lo infiere del color de fondo de la app)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def tema() -> dict:
    """Colores de las gráficas y tablas que no pueden heredarse del tema. Las series (modelos y clases) usan los mismos
    tonos en ambos temas: validados contra fondo claro (#F6F5F1) y oscuro (#171614)."""
    if oscuro():
        return dict(texto="#EDEBE8", texto_sobre_marca="#0F1B2D", rampa=["#1b2432", "#5aa0f0"],
                    umbral_celda=0.68, texto_celda_alta="#0F1B2D", texto_celda_baja="#EDEBE8", resalte="#23406b")
    # umbral_celda: fracción a partir de la cual cambia el color del texto de la celda (cruce de contraste ≈ 0.75 / 0.68)
    return dict(texto="#0F1B2D", texto_sobre_marca="#0F1B2D", rampa=["#eef3fb", "#1c5db3"],
                umbral_celda=0.75, texto_celda_alta="#ffffff", texto_celda_baja="#0F1B2D", resalte="#d9e8fb")


ORDEN_MODELOS = [v["nombre"] for v in MODELOS_INFO.values()]
RANGO_MODELOS = [v["color"] for v in MODELOS_INFO.values()]
RANGO_CLASES = [COLOR_CLASES[c] for c in CLASES]
ESCALA_MODELOS = alt.Scale(domain=ORDEN_MODELOS, range=RANGO_MODELOS)
ESCALA_CLASES = alt.Scale(domain=CLASES, range=RANGO_CLASES)


# ================================================================================================================
# Recursos en caché
# ================================================================================================================
@st.cache_resource(show_spinner="Cargando los tres modelos…")
def modelos():
    return nucleo.cargar_modelos()


@st.cache_data
def referencias():
    X_tr, _, _, _ = cat.cargar_particion()
    return dict(tipicos=cat.valores_tipicos(X_tr), observados=cat.rangos_observados(X_tr),
                vistas=cat.categorias_vistas(X_tr), n_train=len(X_tr))


@st.cache_data(show_spinner="Evaluando los modelos en el conjunto de prueba…")
def evaluacion_test():
    """Probabilidades de los tres modelos reales sobre el Test (885 estudiantes) y la clase real."""
    _, _, X_te, y_te = cat.cargar_particion()
    return nucleo.predecir(modelos(), X_te), y_te, X_te


@st.cache_data
def tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(RAIZ / "resultados" / f"{nombre}.csv", index_col=0, encoding="utf-8-sig")


def fila_desde_dict(d: dict) -> pd.DataFrame:
    return pd.DataFrame([d])[cat.cargar_columnas()["orden"]]


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


# ================================================================================================================
# Gráficas
# ================================================================================================================
def grafica_probabilidades(t: pd.DataFrame) -> alt.Chart:
    filas = []
    for _, r in t.iterrows():
        acum = 0.0
        for c in CLASES:
            p = r[f"P({c})"]
            hueco = min(0.003, p / 4)   # medio hueco por lado: separa los segmentos sin depender del color de fondo
            filas.append({"Modelo": r["Modelo"], "Clase": c, "Prob": p, "ini": acum + hueco, "fin": acum + p - hueco, "mid": acum + p / 2})
            acum += p
    d = pd.DataFrame(filas)
    base = alt.Chart(d).encode(y=alt.Y("Modelo:N", sort=ORDEN_MODELOS, title=None, axis=alt.Axis(labelFontSize=14, labelLimit=170)))
    T = tema()
    barras = base.mark_bar(size=36).encode(
        x=alt.X("ini:Q", title="Probabilidad estimada", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        x2="fin:Q", color=alt.Color("Clase:N", scale=ESCALA_CLASES, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Modelo", "Clase", alt.Tooltip("Prob:Q", format=".1%")])
    texto = base.transform_filter("datum.Prob >= 0.07").mark_text(fontSize=13, fontWeight="bold").encode(
        x="mid:Q", text=alt.Text("Prob:Q", format=".0%"),
        color=alt.condition("datum.Clase == 'Graduate'", alt.value("white"), alt.value(T["texto_sobre_marca"])))
    return (barras + texto).properties(height=alt.Step(46))


def grafica_contribuciones(c: pd.DataFrame, clase: str) -> alt.Chart:
    d = c.copy()
    d["texto"] = d["etiqueta"] + " = " + d["valor"]
    d["sentido"] = np.where(d["aporte"] >= 0, f"A favor de {clase}", f"En contra de {clase}")
    return alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("texto:N", sort=d["texto"].tolist(), title=None, axis=alt.Axis(labelLimit=360, labelFontSize=13)),
        x=alt.X("aporte:Q", title="Aporte al puntaje de la clase (coeficiente × valor estandarizado)"),
        color=alt.Color("sentido:N", scale=alt.Scale(domain=[f"A favor de {clase}", f"En contra de {clase}"],
                                                      range=[COLOR_CLASES[clase], "#898781"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["etiqueta", "valor", alt.Tooltip("aporte:Q", format="+.2f")]).properties(height=34 * len(d) + 50)


def grafica_matrices(probas: dict, y: np.ndarray) -> alt.Chart:
    filas = []
    for k, info in MODELOS_INFO.items():
        pred = probas[k].argmax(axis=1)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True)
        for i, real in enumerate(CLASES):
            for j, pr in enumerate(CLASES):
                filas.append({"Modelo": info["nombre"], "Real": real, "Predicha": pr, "n": int(cm[i, j]),
                              "frac": float(norm[i, j]), "texto": f"{cm[i, j]}\n{100 * norm[i, j]:.0f} %"})
    d = pd.DataFrame(filas)
    base = alt.Chart(d).encode(
        x=alt.X("Predicha:N", sort=CLASES, title="Clase predicha", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Real:N", sort=CLASES, title="Clase real"))
    T = tema()
    celdas = base.mark_bar(width={"band": 0.96}, height={"band": 0.96}).encode(
        color=alt.Color("frac:Q", scale=alt.Scale(domain=[0, 1], range=T["rampa"]), legend=None),
        tooltip=["Modelo", "Real", "Predicha", "n", alt.Tooltip("frac:Q", format=".1%", title="% de la clase real")])
    texto = base.mark_text(fontSize=14, lineBreak="\n").encode(
        text="texto:N", color=alt.condition(f"datum.frac > {T['umbral_celda']}", alt.value(T["texto_celda_alta"]),
                                            alt.value(T["texto_celda_baja"])))
    return (celdas + texto).properties(width=230, height=230).facet(
        column=alt.Column("Modelo:N", sort=ORDEN_MODELOS, title=None, header=alt.Header(labelFontSize=15)))


def grafica_roc(probas: dict, y: np.ndarray) -> alt.Chart:
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#898781", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    paneles = []
    for j, c in enumerate(CLASES):
        filas = []
        for k, info in MODELOS_INFO.items():
            fpr, tpr, _ = roc_curve(y == j, probas[k][:, j])
            paso = max(1, len(fpr) // 120)
            for a, b in list(zip(fpr, tpr))[::paso] + [(1.0, 1.0)]:
                filas.append({"Modelo": info["nombre"], "FPR": a, "TPR": b})
        curvas = alt.Chart(pd.DataFrame(filas)).mark_line(strokeWidth=2.2).encode(
            x=alt.X("FPR:Q", title="Falsos positivos (1 − especificidad)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="Verdaderos positivos (recall)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Modelo:N", scale=ESCALA_MODELOS, legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("Modelo:N", sort=ORDEN_MODELOS, legend=None),
            tooltip=["Modelo", alt.Tooltip("FPR:Q", format=".2f"), alt.Tooltip("TPR:Q", format=".2f")])
        paneles.append((diag + curvas).properties(width=250, height=250, title=c))
    return alt.hconcat(*paneles)


# ================================================================================================================
# Entradas
# ================================================================================================================
def campo(col: str, prefijo: str, tip: dict, vistas: dict):
    """Crea el widget adecuado para la variable `col` y devuelve el valor capturado."""
    cols = cat.cargar_columnas()
    etiqueta, ayuda, key = cat.ETIQUETAS[col], cat.AYUDAS.get(col), f"{prefijo}_{col}"
    if col in cat.CATALOGOS:
        ops = vistas[col]
        ops = sorted(ops, key=lambda c: cat.CATALOGOS[col].get(c, "")) if col == "Course" else sorted(ops)
        return st.selectbox(etiqueta, ops, index=ops.index(int(tip[col])), key=key, help=ayuda,
                            format_func=lambda c, col=col: cat.CATALOGOS[col].get(c, f"código {c}"))
    if col in cat.BINARIAS:
        a, b = cat.BINARIAS[col]
        return st.radio(etiqueta, [0, 1], index=int(tip[col]), key=key, help=ayuda, horizontal=True,
                        format_func=lambda v, a=a, b=b: b if v else a)
    lo, hi = cat.LIMITES.get(col, (0, 60))
    if "(grade)" in col and ("Curricular" not in col):          # notas 0–200
        return st.number_input(etiqueta, float(lo), float(hi), float(round(tip[col], 1)), 0.5, format="%.1f", key=key, help=ayuda)
    if "(grade)" in col:                                          # promedios 0–20
        return st.number_input(etiqueta, float(lo), float(hi), float(round(tip[col], 2)), 0.1, format="%.2f", key=key, help=ayuda)
    if col == "Admission grade":
        return st.number_input(etiqueta, float(lo), float(hi), float(round(tip[col], 1)), 0.5, format="%.1f", key=key, help=ayuda)
    if col in ("Unemployment rate", "Inflation rate", "GDP"):
        return st.number_input(etiqueta, float(lo), float(hi), float(round(tip[col], 2)), 0.1, format="%.2f", key=key, help=ayuda)
    return st.number_input(etiqueta, int(lo), int(hi), int(round(tip[col])), 1, key=key, help=ayuda)


def mostrar_problemas(errores: list[str], avisos: list[str]) -> bool:
    """Muestra errores y avisos. Devuelve True si se puede predecir."""
    for e in errores:
        st.error(e, icon="⛔")
    for a in avisos:
        st.warning(a, icon="⚠️")
    return not errores


# ================================================================================================================
# Resultado de una predicción
# ================================================================================================================
def mostrar_resultado(d: dict, tecnico: bool = False):
    fila = fila_desde_dict(d)
    tiempos = {}
    probas = {}
    for k in MODELOS_INFO:
        t0 = time.perf_counter()
        probas[k] = modelos()[k].predict_proba(fila[cat.cargar_columnas()["orden"]])
        tiempos[k] = (time.perf_counter() - t0) * 1000
    t = nucleo.tabla_predicciones(probas)
    cons = nucleo.consenso([CLASES.index(p) for p in t["Predicción"]])

    color = {"alerta": "#c4491a", "mixta": "#6b6a64", "ninguna": "#0f7a53"}[cons["clave"]]
    if cons["clave"] == "alerta":
        sub = f"{cons['n_dropout']} de 3 modelos predicen **Dropout** (abandono)."
    elif cons["clave"] == "mixta":
        sub = "1 de 3 modelos predice **Dropout**; los demás no. Conviene revisar el caso."
    else:
        mayor = CLASES[cons["mayoritaria"]] if cons["mayoritaria"] is not None else None
        sub = ("Ningún modelo predice Dropout. " + (f"La predicción mayoritaria es **{mayor}**." if mayor else
               "Los modelos se reparten entre Enrolled y Graduate."))
    sub_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", sub)
    st.markdown(f'<div class="caja" style="--c:{color}"><div class="t">{cons["nivel"]}</div>'
                f'<div class="s">{sub_html}</div></div>', unsafe_allow_html=True)

    vista = t.copy()
    for c in ["Confianza"] + [f"P({n})" for n in CLASES]:
        vista[c] = vista[c] * 100
    if tecnico:
        vista["Tiempo (ms)"] = [tiempos[k] for k in MODELOS_INFO]
    cfg = {"Confianza": st.column_config.ProgressColumn("Confianza", min_value=0, max_value=100, format="%.0f%%",
                                                          help="Probabilidad que el modelo asigna a la clase que predice."),
           **{f"P({n})": st.column_config.NumberColumn(f"P({n})", format="%.1f%%") for n in CLASES}}
    if tecnico:
        cfg["Tiempo (ms)"] = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(vista, hide_index=True, width="stretch", column_config=cfg)
    izq, der = st.columns([1, 1.25])
    with izq:
        st.markdown(nucleo.frase_diferencias(t))
        spread = (t["P(Dropout)"].max() - t["P(Dropout)"].min()) * 100
        st.caption(f"Diferencia máxima entre modelos en P(Dropout): {spread:.1f} puntos porcentuales. "
                   f"Promedio de los tres: {100 * t['P(Dropout)'].mean():.1f} %.")
    with der:
        st.altair_chart(grafica_probabilidades(t), width="stretch")

    clase_lr = int(probas["rl"][0].argmax())
    contrib = nucleo.contribuciones_lr(modelos()["rl"], fila, clase_lr)
    with st.expander(f"¿Qué pesó más en esta predicción? (modelo logístico, clase «{CLASES[clase_lr]}»)"):
        if contrib.empty:
            st.info("No hay contribuciones distintas de cero para este caso.")
        else:
            st.altair_chart(grafica_contribuciones(contrib, CLASES[clase_lr]), width="stretch")
            st.caption("El modelo logístico es el único de los tres que se puede leer de forma directa: cada barra es lo que "
                       "una variable suma o resta al puntaje de la clase. Es una explicación del modelo, no una causa: "
                       "los otros dos modelos pueden pesar las variables de otra forma.")
    explicacion_resultados()
    return probas


def explicacion_resultados():
    with st.expander("¿Cómo interpretar estos resultados? (y qué límites tienen)"):
        st.markdown(
            """
**¿Qué representa la clase predicha?**
- **Dropout**: el estudiante abandona la carrera antes de terminarla.
- **Enrolled**: sigue inscrito al terminar la duración normal de la carrera (va con rezago, pero no abandonó).
- **Graduate**: se gradúa dentro de la duración normal.

**¿Cómo leer las probabilidades?** Cada modelo reparte 100 % entre las tres clases y predice la más alta. La *confianza*
es la probabilidad de la clase predicha. Los modelos se entrenaron con pesos de clase balanceados, así que las
probabilidades **sirven para comparar casos y modelos, pero no son frecuencias reales** de abandono (no están calibradas).

**¿Cómo usar el semáforo?** Es una regla simple sobre los votos: *alerta* si 2 o 3 modelos predicen Dropout, *señal mixta* si
solo 1, *sin alerta* si ninguno. Sirve para priorizar a quién dar seguimiento, no para decidir por una persona.

**Limitaciones**
- La clase **Enrolled** es la más difícil: se confunde con las otras dos (F1 de 0.49 a 0.53 en el conjunto de prueba).
- Los modelos aprendieron de estudiantes de **una sola institución** (Instituto Politécnico de Portalegre, Portugal); en otro
  contexto pueden fallar más.
- La predicción se hace **después del 2.º semestre**; no es una alerta temprana de ingreso.
- Entre los modelos hay diferencias de F1 de pocas centésimas que **no son estadísticamente significativas** tras corregir por
  comparaciones múltiples (ver Modo developer): ningún modelo es «el correcto»; por eso se muestran los tres.
- Las variables no capturadas en el modo simplificado se completan con el valor típico del entrenamiento.
            """)


# ================================================================================================================
# Modo consejero
# ================================================================================================================
def vista_simple(ref: dict):
    tip, vistas, obs = ref["tipicos"], ref["vistas"], ref["observados"]
    st.caption("Valores iniciales = estudiante típico del conjunto de entrenamiento. Cámbialos por los datos reales del "
               "estudiante al cierre del 2.º semestre. Las otras 22 variables del modelo se completan con valores típicos.")
    with st.form("form_simple"):
        v = {}
        st.markdown("**Perfil y situación**")
        c1, c2, c3 = st.columns(3)
        with c1:
            v["Course"] = campo("Course", "s", tip, vistas)
            v["Age at enrollment"] = campo("Age at enrollment", "s", tip, vistas)
        with c2:
            v["Admission grade"] = campo("Admission grade", "s", tip, vistas)
            v["Scholarship holder"] = campo("Scholarship holder", "s", tip, vistas)
        with c3:
            v["Tuition fees up to date"] = campo("Tuition fees up to date", "s", tip, vistas)
            v["Debtor"] = campo("Debtor", "s", tip, vistas)
        st.markdown("**Rendimiento académico** (UC = unidades curriculares, es decir, materias)")
        s1, s2 = st.columns(2)
        for col_ui, sem, nombre in ((s1, "1st", "1.er semestre"), (s2, "2nd", "2.º semestre")):
            with col_ui:
                with st.container(border=True):
                    st.markdown(f"_{nombre}_")
                    a, b = st.columns(2)
                    with a:
                        v[f"Curricular units {sem} sem (enrolled)"] = campo(f"Curricular units {sem} sem (enrolled)", "s", tip, vistas)
                        v[f"Curricular units {sem} sem (approved)"] = campo(f"Curricular units {sem} sem (approved)", "s", tip, vistas)
                    with b:
                        v[f"Curricular units {sem} sem (evaluations)"] = campo(f"Curricular units {sem} sem (evaluations)", "s", tip, vistas)
                        v[f"Curricular units {sem} sem (grade)"] = campo(f"Curricular units {sem} sem (grade)", "s", tip, vistas)
        enviar = st.form_submit_button("Estimar riesgo de abandono", type="primary")
    if enviar:
        d = {**tip, **v}
        errores, avisos = cat.validar_registro(d, obs, vistas)
        st.session_state["simple"] = None if errores else d
        st.session_state["simple_msgs"] = (errores, avisos)
    if "simple_msgs" in st.session_state:
        if mostrar_problemas(*st.session_state["simple_msgs"]) and st.session_state.get("simple"):
            d = st.session_state["simple"]
            st.divider()
            mostrar_resultado(d)
            supuestos = [c for c in cat.cargar_columnas()["orden"] if c not in cat.CAMPOS_SIMPLE]
            with st.expander(f"Valores supuestos para las otras {len(supuestos)} variables"):
                st.dataframe(pd.DataFrame({"Variable": [cat.ETIQUETAS[c] for c in supuestos],
                                           "Valor supuesto": [cat.etiqueta_valor(c, tip[c]) for c in supuestos]}),
                             hide_index=True, width="stretch")
                st.caption("Para capturarlas tú mismo usa el modo 2 · Avanzado.")


def vista_avanzada(ref: dict):
    tip, vistas, obs = ref["tipicos"], ref["vistas"], ref["observados"]
    t_uno, t_csv = st.tabs(["Un estudiante (36 variables)", "Archivo CSV (varios estudiantes)"])
    with t_uno:
        st.caption("Las 36 variables del modelo, agrupadas. Valores iniciales = estudiante típico del Train.")
        with st.form("form_avanzado"):
            v = {}
            for seccion, campos in cat.SECCIONES_AVANZADO.items():
                with st.expander(seccion, expanded=seccion in ("Ingreso", "1.er semestre", "2.º semestre")):
                    cols_ui = st.columns(3)
                    for i, c in enumerate(campos):
                        with cols_ui[i % 3]:
                            v[c] = campo(c, "a", tip, vistas)
            enviar = st.form_submit_button("Predecir con los tres modelos", type="primary")
        if enviar:
            errores, avisos = cat.validar_registro(v, obs, vistas)
            st.session_state["avanzado"] = None if errores else v
            st.session_state["avanzado_msgs"] = (errores, avisos)
        if "avanzado_msgs" in st.session_state:
            if mostrar_problemas(*st.session_state["avanzado_msgs"]) and st.session_state.get("avanzado"):
                st.divider()
                mostrar_resultado(st.session_state["avanzado"])
    with t_csv:
        vista_csv(ref)


def vista_csv(ref: dict):
    st.markdown("Sube un CSV con **una fila por estudiante** y las 36 columnas del dataset (mismos nombres y códigos que "
                "`data.csv` de UCI; separador coma o punto y coma). Cada fila se valida antes de predecir.")
    st.download_button("Descargar plantilla (1 fila de ejemplo)", nucleo.plantilla_csv(ref["tipicos"]),
                       file_name="plantilla_estudiantes.csv", mime="text/csv")
    archivo = st.file_uploader("Archivo CSV", type=["csv"], accept_multiple_files=False)
    if archivo is None:
        return
    try:
        df = nucleo.leer_csv(archivo.getvalue())
        X, problemas, notas = nucleo.procesar_csv(df, ref["observados"], ref["vistas"])
    except Exception as e:  # noqa: BLE001
        st.error(f"No pude leer el archivo: {e}", icon="⛔")
        return
    for n in notas:
        st.info(n)
    n_err = problemas[problemas["Tipo"] == "Error"]["Fila"].nunique()
    st.write(f"**{len(X):,} filas válidas** de {len(X) + n_err:,}" + (f" · {n_err} con errores (no se predicen)." if n_err else "."))
    if len(problemas):
        with st.expander(f"Errores y avisos ({len(problemas)})", expanded=bool(n_err)):
            st.dataframe(problemas, hide_index=True, width="stretch")
    if X.empty:
        return
    probas = nucleo.predecir(modelos(), X)
    clases = {k: probas[k].argmax(axis=1) for k in MODELOS_INFO}
    res = pd.DataFrame({"Fila": X.index + 2})
    for k, info in MODELOS_INFO.items():
        res[info["corto"]] = [CLASES[c] for c in clases[k]]
    res["Modelos que predicen Dropout"] = sum((clases[k] == 0).astype(int) for k in MODELOS_INFO)
    res["Nivel"] = res["Modelos que predicen Dropout"].map(lambda n: "Alerta" if n >= 2 else ("Señal mixta" if n == 1 else "Sin alerta"))
    res["P(Dropout) promedio"] = np.mean([probas[k][:, 0] for k in MODELOS_INFO], axis=0) * 100
    res = res.sort_values(["Modelos que predicen Dropout", "P(Dropout) promedio"], ascending=False)
    a, b, c = st.columns(3)
    a.metric("Alerta de abandono", int((res["Nivel"] == "Alerta").sum()))
    b.metric("Señal mixta", int((res["Nivel"] == "Señal mixta").sum()))
    c.metric("Sin alerta", int((res["Nivel"] == "Sin alerta").sum()))
    st.dataframe(res, hide_index=True, width="stretch",
                 column_config={"P(Dropout) promedio": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
    st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")
    st.caption("Ordenado por prioridad de seguimiento: primero los casos con más modelos prediciendo Dropout.")


def pestana_consejero():
    ref = referencias()
    st.subheader("Modo consejero")
    st.markdown("Estima, con **tres modelos a la vez**, si un estudiante terminará la carrera (**Graduate**), seguirá "
                "inscrito con rezago (**Enrolled**) o la abandonará (**Dropout**). Está pensado para tutores y consejeros: "
                "da un semáforo de atención y deja ver en qué coinciden y en qué no los modelos.")
    modo = st.radio("Modo de captura", ["1 · Simplificado", "2 · Avanzado"], horizontal=True,
                    help="El simplificado pide 14 datos y completa el resto con valores típicos; el avanzado captura las 36 "
                         "variables del modelo o carga un CSV con varios estudiantes.")
    if modo.startswith("1"):
        vista_simple(ref)
    else:
        vista_avanzada(ref)


# ================================================================================================================
# Modo developer
# ================================================================================================================
def parsear_ic(celda) -> tuple[float, float, float] | None:
    m = re.match(r"\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", str(celda))
    return tuple(map(float, m.groups())) if m else None


def dev_rendimiento():
    met, ic = tabla("04_test_todos_los_modelos"), tabla("04_test_intervalos_bootstrap")
    filas = {v["fila"]: v["nombre"] for v in MODELOS_INFO.values()}
    m = met.loc[list(filas)].rename(index=filas)
    cols = st.columns(3)
    for col_ui, (k, info) in zip(cols, MODELOS_INFO.items()):
        f1, lo, hi = parsear_ic(ic.loc[info["fila"], "F1"])
        with col_ui:
            with st.container(border=True):
                st.markdown(f'<div class="modelo" style="--c:{info["color"]}; padding-top:6px"><b>{info["nombre"]}</b></div>', unsafe_allow_html=True)
                st.metric("F1 macro (Test)", f"{f1:.3f}", help="Promedio simple del F1 de las tres clases.")
                st.caption(f"IC 95 % bootstrap: [{lo:.3f}, {hi:.3f}] · Accuracy {m.loc[info['nombre'], 'Accuracy']:.3f} · "
                           f"ROC-AUC {m.loc[info['nombre'], 'ROC-AUC']:.3f}")
    st.markdown("##### Métricas en el conjunto de prueba (885 estudiantes que no se usaron para entrenar ni elegir hiperparámetros)")
    nombres = {"Precision": "Precisión macro", "Recall": "Recall macro", "F1": "F1 macro", "Kappa": "κ de Cohen",
               "F1 (CV)": "F1 macro (CV)", "Predicción (ms)": "Predicción, 885 casos (ms)"}
    mostrar = m[["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC", "F1 (CV)", "Test − CV (F1)"]].rename(columns=nombres)
    cols_mejor = [c for c in mostrar.columns if c != "Test − CV (F1)"]
    st.dataframe(mostrar.style.format("{:.3f}").highlight_max(subset=cols_mejor, axis=0, color=tema()["resalte"]), width="stretch")
    with st.expander("Todas las métricas (F1 ponderado, κ de Cohen, tiempo de predicción)"):
        todas = m[["Accuracy", "Precision", "Recall", "F1", "F1 ponderado", "MCC", "Kappa", "ROC-AUC", "Predicción (ms)"]].rename(columns=nombres)
        st.dataframe(todas.style.format({**{c: "{:.3f}" for c in todas.columns}, "Predicción, 885 casos (ms)": "{:.0f}"}), width="stretch")
    dummy = met.loc["Dummy (referencia)"]
    st.caption(f"Azul = mejor de los tres en esa métrica. Referencia sin aprender nada (Dummy, siempre Graduate): accuracy "
               f"{dummy['Accuracy']:.3f} y F1 macro {dummy['F1']:.3f}. «Test − CV» negativo = el modelo rinde menos en Test que en validación.")

    st.markdown("##### Comparación visual")
    sel = st.radio("Métrica", ["F1", "Accuracy", "MCC", "ROC-AUC", "Recall", "Precision"], horizontal=True,
                   format_func=lambda s: {"F1": "F1 macro", "Precision": "Precisión macro", "Recall": "Recall macro"}.get(s, s), key="dev_metrica")
    d = pd.DataFrame([{"Modelo": info["nombre"], **dict(zip(["valor", "lo", "hi"], parsear_ic(ic.loc[info["fila"], sel])))}
                      for info in MODELOS_INFO.values()])
    base = alt.Chart(d).encode(y=alt.Y("Modelo:N", sort=ORDEN_MODELOS, title=None, axis=alt.Axis(labelFontSize=14, labelLimit=170)))
    puntos = base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title="Valor en Test (punto) e IC 95 % bootstrap (barra)",
                                                         scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=7, format=".2f")), x2="hi:Q",
                                                  color=alt.Color("Modelo:N", scale=ESCALA_MODELOS, legend=None))
    punto = base.mark_point(filled=True, size=140, opacity=1).encode(x="valor:Q", color=alt.Color("Modelo:N", scale=ESCALA_MODELOS, legend=None),
                                                                    tooltip=["Modelo", alt.Tooltip("valor:Q", format=".3f"),
                                                                             alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")])
    txt = base.mark_text(dy=-16, fontSize=13, fontWeight="bold", color=tema()["texto"]).encode(x="valor:Q", text=alt.Text("valor:Q", format=".3f"))
    st.altair_chart((puntos + punto + txt).properties(height=alt.Step(52)), width="stretch")
    st.caption("Los intervalos se traslapan casi por completo: con 885 estudiantes de prueba la incertidumbre es de ±0.03, "
               "mayor que las diferencias entre modelos.")

    with st.expander("¿Por qué estos tres modelos? Todos los modelos del proyecto en Test"):
        todos = met.drop(index=["Dummy (referencia)"])[["Accuracy", "F1", "ROC-AUC", "F1 (CV)"]].rename(
            columns={"F1": "F1 macro (Test)", "F1 (CV)": "F1 macro (CV)"}).sort_values("F1 macro (Test)", ascending=False)
        elegidos = set(filas)
        st.dataframe(todos.style.format("{:.3f}").apply(lambda r: [f"background-color:{tema()['resalte']}" if r.name in elegidos else "" for _ in r], axis=1),
                     width="stretch")
        st.markdown(
            """
- **Random Forest optimizado**: mejor F1 macro en Test entre todos los modelos del proyecto.
- **Voting soft optimizado**: mejor F1 macro en validación cruzada y mejor ROC-AUC entre los modelos optimizados; combina cinco modelos.
- **Regresión logística ajustada**: línea base interpretable; ninguno de los ensambles la supera de forma estadísticamente
  significativa tras la corrección de Holm.

Se eligieron tres modelos de familias distintas (lineal, bosque, ensamble heterogéneo) para que la comparación diga algo: los
mejores por Test (Random Forest, Bagging, Pasting) son casi el mismo modelo. El orden en validación cruzada y en Test no coincide
(Spearman −0.32), así que ninguna selección por un solo criterio sería más defendible.
            """)


def dev_clases():
    probas, y, _ = evaluacion_test()
    st.caption("Esta sección se calcula aquí mismo: los tres modelos guardados predicen los 885 estudiantes del conjunto de prueba.")
    f1c, rec = tabla("04_test_f1_por_clase"), tabla("04_test_recall_por_clase")
    filas = {v["fila"]: v["nombre"] for v in MODELOS_INFO.values()}
    a_ = f1c.loc[list(filas)].rename(index=filas)
    b_ = rec.loc[list(filas)].rename(index=filas)
    largo = pd.concat([a_.stack().rename("F1"), b_.stack().rename("Recall")], axis=1).reset_index()
    largo.columns = ["Modelo", "Clase", "F1", "Recall"]
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.markdown("##### F1 y recall por clase")
        sel = st.radio("Métrica por clase", ["F1", "Recall"], horizontal=True, key="dev_porclase")
        g = alt.Chart(largo).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Clase:N", sort=CLASES, title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("Modelo:N", sort=ORDEN_MODELOS),
            y=alt.Y(f"{sel}:Q", scale=alt.Scale(domain=[0, 1]), title=sel),
            color=alt.Color("Modelo:N", scale=ESCALA_MODELOS, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Modelo", "Clase", alt.Tooltip(f"{sel}:Q", format=".3f")])
        t = alt.Chart(largo).mark_text(dy=-8, fontSize=11, color=tema()["texto"]).encode(
            x=alt.X("Clase:N", sort=CLASES), xOffset=alt.XOffset("Modelo:N", sort=ORDEN_MODELOS), y=f"{sel}:Q", text=alt.Text(f"{sel}:Q", format=".2f"))
        st.altair_chart((g + t).properties(height=300), width="stretch")
    with c2:
        st.markdown("##### Lectura")
        enr, otras = largo[largo["Clase"] == "Enrolled"], largo[largo["Clase"] != "Enrolled"]
        rec_enr = enr.set_index("Modelo")["Recall"]
        st.markdown(
            f"""
- **Graduate** y **Dropout** se predicen bien (F1 de {otras['F1'].min():.2f} a {otras['F1'].max():.2f}).
- **Enrolled** es la clase difícil: el F1 queda entre {enr['F1'].min():.2f} y {enr['F1'].max():.2f} en los tres modelos. Es una
  categoría intermedia y los datos no alcanzan a separarla; no es cuestión de elegir otro algoritmo.
- El recall de Enrolled es de {rec_enr['Regresión logística']:.2f} en la regresión logística, {rec_enr['Random Forest']:.2f} en el
  Random Forest y {rec_enr['Voting soft']:.2f} en el Voting: los modelos que aciertan más en Graduate dejan escapar más Enrolled.
            """)
    st.markdown("##### Matrices de confusión (Test)")
    st.altair_chart(grafica_matrices(probas, y), width="content")
    st.caption("Cada celda muestra cuántos estudiantes y qué porcentaje de su clase real (filas suman 100 %).")
    st.markdown("##### Curvas ROC")
    st.altair_chart(grafica_roc(probas, y), width="content")


def dev_validacion():
    st.markdown("##### Validación cruzada (5 folds sobre Train)")
    cv = tabla("03_modelos_optimizados_cv")
    filas = {v["fila"]: v["nombre"] for v in MODELOS_INFO.values()}
    st.dataframe(cv.loc[list(filas), ["Accuracy", "F1", "MCC", "ROC-AUC", "Gap F1 (train−CV)"]].rename(index=filas).rename(columns={"F1": "F1 macro"}), width="stretch")
    st.caption("Media ± desviación estándar entre folds. El «gap» es F1 en los folds de entrenamiento menos F1 en validación: "
               "Random Forest y Voting memorizan bastante más el Train (0.18–0.20) que la regresión logística (0.03).")

    st.markdown("##### De entrenamiento a validación y a prueba")
    gen = tabla("04_generalizacion").loc[list(filas)].rename(index=filas)
    largo = gen[["F1 train", "F1 CV (5 folds)", "F1 Test"]].stack().reset_index()
    largo.columns = ["Modelo", "Etapa", "F1 macro"]
    pts = alt.Chart(largo).mark_point(filled=True, size=130, opacity=1).encode(
        y=alt.Y("Modelo:N", sort=ORDEN_MODELOS, title=None, axis=alt.Axis(labelFontSize=14, labelLimit=170)),
        x=alt.X("F1 macro:Q", scale=alt.Scale(domain=[0.65, 1.0]), axis=alt.Axis(tickCount=8, format=".2f")),
        shape=alt.Shape("Etapa:N", sort=["F1 train", "F1 CV (5 folds)", "F1 Test"], legend=alt.Legend(orient="bottom", title=None, symbolFillColor=tema()["texto"], symbolStrokeColor=tema()["texto"])),
        color=alt.Color("Modelo:N", scale=ESCALA_MODELOS, legend=None),
        tooltip=["Modelo", "Etapa", alt.Tooltip("F1 macro:Q", format=".3f")])
    st.altair_chart(pts.properties(height=alt.Step(52)), width="stretch")
    st.caption("Forma = etapa (círculo = entrenamiento, cuadrado = validación cruzada, triángulo = prueba). Todos pierden entre 0.01 y 0.03 de F1 de la validación a la prueba.")

    st.markdown("##### ¿Alguna diferencia es real? Inferencia estadística")
    rep, par = tabla("04_cv_repetida_vs_lr_f1"), tabla("04_pareadas_vs_lr_regla")
    cand = {"Random Forest optimizado": "Random Forest", "Voting (soft) optimizado": "Voting soft"}
    r = rep.loc[list(cand)].rename(index=cand)[["Diferencia F1 vs LR", "IC 95% inferior", "IC 95% superior", "Gana (de 25)", "p-valor", "p Holm"]]
    st.markdown("**Validación cruzada repetida 5×5 frente a la regresión logística** (prueba t corregida de Nadeau–Bengio, ajuste de Holm por 11 comparaciones)")
    st.dataframe(r.style.format({"Diferencia F1 vs LR": "{:+.4f}", "IC 95% inferior": "{:+.4f}", "IC 95% superior": "{:+.4f}", "Gana (de 25)": "{:.0f}", "p-valor": "{:.3f}", "p Holm": "{:.3f}"}), width="stretch")
    p = par.loc[list(cand)].rename(index=cand)[["F1", "Accuracy", "ROC-AUC", "McNemar (solo él / solo LR)"]]
    st.markdown("**Conjunto de prueba, diferencias pareadas** (frente a la regresión logística con la regla P(Dropout) ≥ 0.40 del proyecto; entre corchetes el IC 95 % bootstrap)")
    st.dataframe(p, width="stretch")
    st.info(f"Tras la corrección de Holm, **ningún ensamble supera a la regresión logística** en la validación cruzada repetida "
            f"(p Holm = {r.loc['Voting soft', 'p Holm']:.2f} para el Voting y {r.loc['Random Forest', 'p Holm']:.2f} para el Random Forest). "
            "En Test, McNemar sí detecta que algunos aciertan más casos que la regresión logística, pero la diferencia de ROC-AUC "
            "incluye el cero: cambian el **punto de operación** (más exactitud, menos recall de Enrolled), no la capacidad de discriminar.")


def dev_configuracion():
    import json
    hp = json.load(open(RAIZ / "resultados" / "03_mejores_hiperparametros.json", encoding="utf-8"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Pipeline (idéntico al del entrenamiento)")
        pre = modelos()["rl"].named_steps["preprocessor"]
        n_cols = len(pre.get_feature_names_out())
        cols = cat.cargar_columnas()
        st.markdown(f"""
1. **StandardScaler** en {len(cols['numericas'])} variables numéricas.
2. **One-hot** en {len(cols['nominales'])} variables nominales (`min_frequency = 0.01`; categorías raras → «infrecuente»).
3. **Passthrough** en {len(cols['binarias'])} variables binarias.
4. Resultado: **{n_cols} columnas** → modelo.

El preprocesamiento se ajustó solo con Train y viaja dentro de cada `.joblib`; la app no transforma nada por su cuenta.
Partición 80/20 estratificada, `random_state = 42` (Train 3,539 · Test 885).
""")
        pesos = hp["Voting (soft)"]["weights"]
        st.markdown(f"**Voting soft**: miembros (LR, SVM RBF calibrado con Platt, RF, Gradient Boosting, HistGradientBoosting), pesos {tuple(pesos)}.")
    with c2:
        st.markdown("##### Hiperparámetros")
        st.json({"Regresión logística": {"penalty": "L1 (saga)", "C": 0.3, "class_weight": "balanced", "max_iter": 5000},
                 "Random Forest": {"n_estimators": 300, **hp["Random Forest"]},
                 "SVM RBF (miembro del Voting)": {"C": 100, "gamma": 0.001, "calibración": "sigmoide, cv=5"},
                 "Gradient Boosting (miembro)": hp["Gradient Boosting"], "HistGradientBoosting (miembro)": hp["HistGradientBoosting"]},
                expanded=True)
        tam = {v["nombre"]: (RAIZ / "modelos" / v["archivo"]).stat().st_size / 1e6 for v in MODELOS_INFO.values()}
        st.caption("Archivos guardados: " + " · ".join(f"{k} {v:.1f} MB" for k, v in tam.items()))

    st.markdown("##### Importancia de variables")
    imp = tabla("04_importancia_variables")
    metodo = st.radio("Medida", list(imp.columns), horizontal=True, key="dev_imp",
                      format_func=lambda s: {"MDI (RF)": "MDI del Random Forest", "Permutación RF (ΔF1)": "Permutación · Random Forest",
                                              "Permutación Voting (soft) (ΔF1)": "Permutación · Voting soft"}.get(s, s))
    top = imp[metodo].sort_values(ascending=False).head(10).rename_axis("var").reset_index()
    top["Variable"] = top["var"].map(cat.ETIQUETAS)
    g = alt.Chart(top).mark_bar(cornerRadiusEnd=4, color="#00907f" if "RF" in metodo else "#b87400").encode(
        y=alt.Y("Variable:N", sort=top["Variable"].tolist(), title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13)),
        x=alt.X(f"{metodo}:Q", title="Importancia" if "MDI" in metodo else "Caída del F1 macro en Test al permutar la variable"),
        tooltip=["Variable", alt.Tooltip(f"{metodo}:Q", format=".3f")])
    st.altair_chart(g.properties(height=300), width="stretch")
    st.caption("La importancia por permutación se calculó a posteriori sobre Test y solo describe el modelo; no se usó para seleccionar nada. "
               "Domina el rendimiento académico (UC aprobadas) y la situación de pago.")


def dev_probar():
    probas, y, X_te = evaluacion_test()
    st.markdown("Elige un estudiante real del **conjunto de prueba**, ejecuta los tres modelos y compara con lo que realmente pasó.")
    st.session_state.setdefault("dev_pos", 0)
    a, b = st.columns([2, 1])
    with b:
        st.write("")
        st.button("Elegir uno al azar", on_click=lambda: st.session_state.update(dev_pos=int(np.random.default_rng().integers(len(X_te)))))
    with a:
        pos = st.number_input("Posición en el conjunto de prueba (0–884)", 0, len(X_te) - 1, step=1, key="dev_pos")
    fila = X_te.iloc[[pos]]
    real = CLASES[int(y[pos])]
    st.markdown(f"Clase real: **{real}** · id original en el dataset: `{fila.index[0]}`")
    d = fila.iloc[0].to_dict()
    with st.expander("Datos de este estudiante"):
        orden = cat.cargar_columnas()["orden"]
        st.dataframe(pd.DataFrame({"Variable": [cat.ETIQUETAS[c] for c in orden], "Valor": [cat.etiqueta_valor(c, d[c]) for c in orden]}), hide_index=True, width="stretch", height=300)
    p = mostrar_resultado(d, tecnico=True)
    aciertos = {info["nombre"]: CLASES[int(p[k][0].argmax())] == real for k, info in MODELOS_INFO.items()}
    st.write("**Contra la realidad:** " + " · ".join(f"{n}: {'✅ acierta' if ok else '❌ falla'}" for n, ok in aciertos.items()))
    ya = {k: (probas[k][pos] - p[k][0]) for k in MODELOS_INFO}
    st.caption("Control: la diferencia máxima entre estas probabilidades y las del lote de prueba calculado antes es "
               f"{max(abs(v).max() for v in ya.values()):.2e}.")


def dev_verificacion():
    ruta = RAIZ / "resultados" / "verificacion_app_vs_python.csv"
    st.markdown("Prueba de que la app hace lo mismo que Python directo con los mismos datos: se predijeron los 885 estudiantes del "
                "Test (a) con `Pipeline.predict_proba` leyendo los `.joblib` en un script y (b) pasando cada fila por las funciones que "
                "usa la app (`nucleo.predecir`), y se compararon las probabilidades y las clases.")
    if ruta.exists():
        v = pd.read_csv(ruta)
        cfg = {"Máx. diferencia de probabilidad (app − Python)": st.column_config.NumberColumn(format="%.1e"),
               **{c: st.column_config.NumberColumn(format="%.4f") for c in ("Accuracy (app)", "F1 macro (app)", "F1 macro (proyecto)") if c in v.columns}}
        st.dataframe(v, hide_index=True, width="stretch", column_config=cfg)
        st.caption("La diferencia de ~1e-16 es el error de redondeo de punto flotante: en la práctica las probabilidades son idénticas y las clases coinciden en los 885 casos.")
    else:
        st.info("Aún no hay archivo de verificación. Se genera con `python pruebas/verificar_app_vs_python.py`.")
    st.code("python pruebas/verificar_app_vs_python.py", language="bash")


def pestana_developer():
    st.subheader("Modo developer")
    st.markdown("Estadísticas de rendimiento de los tres modelos. Las cifras vienen de los experimentos del proyecto final "
                "(`resultados/*.csv`) o se calculan aquí con los modelos reales sobre el conjunto de prueba; no hay valores inventados.")
    nombres = ["Rendimiento en Test", "Por clase y errores", "Validación e inferencia", "Configuración", "Probar con un caso real", "Verificación app vs Python"]
    funciones = [dev_rendimiento, dev_clases, dev_validacion, dev_configuracion, dev_probar, dev_verificacion]
    for tab, fn in zip(st.tabs(nombres, on_change="rerun", key="nav_dev"), funciones):
        if tab.open:
            with tab:
                fn()


# ================================================================================================================
# Pestaña Proyecto
# ================================================================================================================
def pestana_proyecto():
    st.title("Predicción de deserción estudiantil con métodos de ensamble")
    st.markdown("Aplicación web del proyecto final de **Machine Learning**: tres modelos reales, entrenados y evaluados "
                "en el proyecto, comparados lado a lado.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Alumno:** Jonahtan Gabriel González Cardoza  \n**Matrícula:** 345491")
    with c2:
        with st.container(border=True):
            st.markdown("**Institución:** Universidad Autónoma de Chihuahua (UACh)  \n"
                        "**Programa y materia:** Maestría en Ingeniería en Computación · Machine Learning (MIC1)  \n"
                        "**Asesora / docente:** Dra. Olanda Prieto Ordaz")
    st.markdown("#### El problema")
    st.markdown("En el conjunto de datos, **3 de cada 10 estudiantes (32.1 %) abandonó** la carrera antes de terminar. Detectar a tiempo "
                "quién está en riesgo permite intervenir (tutorías, apoyo económico, seguimiento). Se plantea una **clasificación en tres "
                "clases**: *Dropout* (abandona), *Enrolled* (sigue inscrito con rezago al terminar la duración normal) y *Graduate* (se gradúa).")
    st.markdown("#### Objetivo de la aplicación")
    st.markdown("- **Modo consejero:** que un tutor o consejero capture los datos de un estudiante (o cargue un CSV) y vea la predicción "
                "de los tres modelos, en qué coinciden y qué tan seguros están.\n"
                "- **Modo developer:** que un revisor técnico vea el rendimiento real de los modelos (métricas en el conjunto de prueba, "
                "validación cruzada, pruebas estadísticas, errores por clase) y pruebe casos reales.")
    st.markdown("#### Base de datos")
    st.markdown("**Predict Students' Dropout and Academic Success** (UCI, id 697; Realinho et al., 2022, *Data* 7(11), 146; CC BY 4.0). "
                "**4,424 estudiantes** de 17 licenciaturas del Instituto Politécnico de Portalegre (Portugal), **36 variables** "
                "(demográficas, socioeconómicas, trayectoria al ingreso, rendimiento de los dos primeros semestres y contexto macroeconómico) "
                "y la clase final. Partición **80/20 estratificada** (semilla 42): 3,539 para entrenar y validar, 885 de prueba que se usan una sola vez por modelo.")
    st.markdown("#### Los tres modelos")
    cols = st.columns(3)
    met, ic = tabla("04_test_todos_los_modelos"), tabla("04_test_intervalos_bootstrap")
    for col_ui, (k, info) in zip(cols, MODELOS_INFO.items()):
        with col_ui:
            with st.container(border=True):
                st.markdown(f'<div class="modelo" style="--c:{info["color"]}; padding-top:6px"><b style="font-size:1.15rem">{info["nombre"]}</b></div>', unsafe_allow_html=True)
                st.write(info["resumen"])
                r = met.loc[info["fila"]]
                st.caption(f"Test: F1 macro **{r['F1']:.3f}** · accuracy {r['Accuracy']:.3f} · ROC-AUC {r['ROC-AUC']:.3f}")
    st.markdown("**Por qué estos tres.** El Random Forest tuvo el mejor F1 macro en Test (0.716); el Voting soft, el mejor en validación cruzada "
                "(0.737) y en ROC-AUC (0.890) entre los modelos optimizados; la regresión logística es la línea base interpretable. Son tres familias distintas. "
                "Ojo: las diferencias entre ellos son de pocas centésimas y, tras la corrección por comparaciones múltiples, "
                "**no son estadísticamente significativas**; la app los muestra juntos precisamente por eso.")
    st.markdown("#### Cómo usarla")
    st.markdown("1. Ve a **Modo consejero**, elige el modo **1 · Simplificado** (14 datos) o **2 · Avanzado** (36 variables o CSV).\n"
                "2. Pulsa el botón para predecir: verás la predicción de cada modelo, su confianza y las probabilidades por clase.\n"
                "3. Abre «¿Cómo interpretar estos resultados?» para entender la salida y sus límites.\n"
                "4. En **Modo developer** consulta las estadísticas de rendimiento y prueba con estudiantes reales del conjunto de prueba.\n"
                "5. Tema claro u oscuro: menú **⋮** (arriba a la derecha) y elige *System*, *Light* o *Dark*. Por defecto sigue el de tu dispositivo.")
    if REPO_URL:
        st.markdown(f"[Código fuente en GitHub]({REPO_URL})")
    st.caption("Herramienta de apoyo a la decisión: no sustituye el criterio de tutores y consejeros.")


# ================================================================================================================
# Navegación
# ================================================================================================================
def main():
    t_proy, t_cons, t_dev = st.tabs(["🏠 Proyecto", "🎓 Modo consejero", "🛠️ Modo developer"], on_change="rerun", key="nav")
    if t_proy.open:
        with t_proy:
            pestana_proyecto()
    if t_cons.open:
        with t_cons:
            pestana_consejero()
    if t_dev.open:
        with t_dev:
            pestana_developer()


main()
