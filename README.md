# Predicción de deserción estudiantil con métodos de ensamble — App web

Aplicación web en **Streamlit** del proyecto final de **Machine Learning** (Maestría en Ingeniería en Computación, Universidad Autónoma de Chihuahua).
Integra tres modelos reales entrenados en el proyecto y los compara lado a lado para clasificar a un estudiante como
**Dropout** (abandona), **Enrolled** (sigue inscrito con rezago) o **Graduate** (se gradúa).

| | |
|---|---|
| **Alumno** | Jonahtan Gabriel González Cardoza · matrícula 345491 |
| **Institución / programa** | Universidad Autónoma de Chihuahua (UACh) · Maestría en Ingeniería en Computación |
| **Materia** | Machine Learning (MIC1) |
| **Docente / asesora** | Dra. Olanda Prieto Ordaz |
| **App en línea** | _pega aquí la URL pública, p. ej. `https://dropout-ml.streamlit.app`_ |

## Qué hace la app

La app tiene tres pestañas:

- **Proyecto**: datos del alumno, problema, objetivo, base de datos y descripción de los tres modelos con sus métricas de Test.
- **Modo consejero** (usuario final: tutor o consejero universitario). Captura un estudiante y muestra la predicción de los tres modelos con un semáforo de atención, la probabilidad de cada clase, en qué coinciden y en qué difieren los modelos y qué variables pesaron más.
  - **Modo 1 · Simplificado**: 14 datos que un consejero conoce (carrera, edad, beca, colegiatura, nota de admisión y el desempeño de los dos semestres). Las otras 22 variables se rellenan con el estudiante típico del conjunto de entrenamiento.
  - **Modo 2 · Avanzado**: las 36 variables del modelo, agrupadas por tema, o un **archivo CSV** con varios estudiantes (con plantilla descargable).
  - Todas las entradas se validan antes de predecir (rangos, enteros, categorías y coherencia entre unidades curriculares aprobadas, inscritas y evaluadas).
- **Modo developer** (revisor técnico). Métricas reales en el conjunto de prueba con intervalos de confianza bootstrap, comparación visual, matrices de confusión, curvas ROC, desempeño por clase, validación cruzada, pruebas estadísticas, hiperparámetros, importancia de variables, un caso real del Test paso a paso y la verificación «app vs Python».

**Tema claro y oscuro**: en el menú ⋮ (arriba a la derecha) se elige *System*, *Light* o *Dark*; por defecto sigue el modo de tu dispositivo. Las gráficas y tablas se adaptan a los dos y los colores de modelos y clases se validaron contra fondo claro y oscuro. Si cambias de tema con una pantalla abierta, las etiquetas de las gráficas se ajustan en cuanto interactúas con la app.

Las cifras del Modo developer salen de los experimentos del proyecto (`resultados/*.csv`) o se calculan en vivo con los modelos reales sobre el Test. No hay valores artificiales.

## Los tres modelos

Los tres son *pipelines* de scikit-learn completos (`ColumnTransformer` + modelo) guardados con `joblib`, así que la app no transforma nada por su cuenta: usa exactamente el preprocesamiento del entrenamiento (escalado de 19 numéricas, one-hot de 9 nominales, 8 binarias; 106 columnas).

| Modelo | Descripción | F1 macro (Test) | Accuracy (Test) | ROC-AUC (Test) |
|---|---|---|---|---|
| Regresión logística | L1 (`saga`), C = 0.3, pesos de clase balanceados. Línea base interpretable | 0.688 | 0.727 | 0.884 |
| Random Forest optimizado | 300 árboles, profundidad 16, entropía, `max_features = log2`, pesos balanceados | 0.716 | 0.759 | 0.887 |
| Voting soft optimizado | Promedio ponderado (1, 1, 2, 2, 2) de regresión logística, SVM RBF calibrado, Random Forest, Gradient Boosting e HistGradientBoosting | 0.703 | 0.756 | 0.890 |

Conjunto de prueba: 885 estudiantes (partición 80/20 estratificada, `random_state = 42`) que no se usaron para entrenar ni para elegir hiperparámetros.
Con 885 estudiantes la incertidumbre es de aproximadamente ±0.03 de F1: los intervalos de los tres modelos se traslapan casi por completo, y tras la corrección de Holm en la validación cruzada repetida ningún ensamble supera de forma significativa a la regresión logística.

## Estructura del repositorio

```
app.py                       Interfaz de Streamlit (pestañas Proyecto, Modo consejero y Modo developer)
nucleo.py                    Lógica sin interfaz: carga de modelos, predicción, consenso, explicación local, lectura de CSV
catalogos.py                 Etiquetas en español, catálogos de códigos, límites y reglas de validación
utilidades.py                Clase GradientBoostingBalanceado (necesaria para cargar el Voting) y constantes
entrenar_modelos.py          Reentrena los tres pipelines y regenera modelos/*.joblib
modelos/                     rl.joblib, rf.joblib, voting.joblib
datos/                       train.csv, test.csv (partición del proyecto) y columnas.json
resultados/                  Tablas del proyecto que alimentan el Modo developer
pruebas/                     test_app.py (pytest) y verificar_app_vs_python.py
.streamlit/config.toml       Temas claro y oscuro y configuración del servidor
requirements.txt             Dependencias con versiones fijas
requirements-dev.txt         Lo anterior más pytest
```

## Ejecutar en local

Se desarrolló y probó con **Python 3.13**.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

> Los archivos `.joblib` solo son compatibles con la versión de scikit-learn con la que se guardaron (1.9.1, fijada en `requirements.txt`).
> Si cambias de versión, corre `python entrenar_modelos.py` para regenerarlos.

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest pruebas -q                       # 29 pruebas: predicción, validaciones, CSV y arranque de la interfaz
python pruebas/verificar_app_vs_python.py         # compara los 885 casos de Test, ~1 min; escribe resultados/verificacion_app_vs_python.csv
```

`verificar_app_vs_python.py` predice los 885 estudiantes de Test de dos maneras: (a) `joblib.load(...).predict_proba(X)` directo en Python y
(b) fila por fila a través de las funciones que usa la app. Resultado: la diferencia máxima de probabilidad es del orden de 1e-16, las clases
predichas son idénticas y el F1 macro coincide con el del proyecto (0.688, 0.716 y 0.703).

## Despliegue en Streamlit Community Cloud

1. Crea un repositorio **público** en GitHub y sube el contenido de esta carpeta (`app.py` debe quedar en la raíz del repositorio). Los modelos pesan ~11 MB en total, muy por debajo del límite de GitHub.
2. Entra a <https://share.streamlit.io> con tu cuenta de GitHub y elige **Create app** → **Deploy a public app from GitHub**.
3. Selecciona el repositorio, la rama (`main`) y el archivo principal `app.py`.
4. Abre **Advanced settings** y elige **Python 3.13**, la misma versión con la que se probó. La versión de Python solo se elige al desplegar (después hay que borrar y volver a crear la app para cambiarla) y `runtime.txt` no se usa.
5. No hace falta ningún *secret*: la app no usa contraseñas, tokens ni APIs externas.
6. Pulsa **Deploy** y espera a que termine la instalación de `requirements.txt`. La URL queda como `https://<subdominio>.streamlit.app`; el subdominio se puede cambiar en **Settings**.
7. Pega la URL en esta tabla (arriba) y la del repositorio en `REPO_URL` dentro de `app.py` para que aparezca en la pestaña Proyecto.

Las apps gratuitas se duermen cuando pasa un tiempo sin visitas; al abrirlas se despiertan con el botón «Yes, get this app back up!». Antes de la revisión, ábrela una vez para confirmar que está despierta.

## Lista de pruebas

| Prueba | Estado |
|---|---|
| La app arranca en local sin errores (`streamlit run app.py` y `AppTest`) | Hecha |
| Los tres modelos se cargan desde `.joblib` como pipelines completos y devuelven probabilidades que suman 1 | Hecha |
| Predicción de la app = predicción directa en Python con las mismas entradas (885 casos de Test) | Hecha, diferencia máxima ≈ 5e-16 |
| Las métricas de Test recalculadas reproducen las del proyecto | Hecha (F1 macro 0.688 / 0.716 / 0.703) |
| Entradas inválidas se rechazan con mensaje claro (edad, notas, 0/1, enteros, campos vacíos, aprobadas > inscritas) | Hecha (pruebas y navegador) |
| Modo 1 con un estudiante típico → sin alerta; con un caso de riesgo diseñado → alerta con 3 de 3 modelos | Hecha en navegador |
| Carga de CSV: separador `,` o `;`, encabezado `Nacionality` del UCI, filas con errores separadas | Hecha (pruebas) |
| Caso real del Test en el Modo developer coincide con el lote calculado antes (diferencia ≈ 1e-16) | Hecha |
| Todas las pantallas se ven bien en tema claro y en tema oscuro (cambio desde el menú ⋮) | Hecha en navegador |
| La app abre en la URL pública sin instalar nada | Pendiente de desplegar |
| Los tres modos se recorren en la URL pública (Proyecto, Consejero 1 y 2, Developer) | Pendiente de desplegar |

## Cómo interpretar los resultados y sus límites

- **Dropout** = el estudiante abandona; **Enrolled** = sigue inscrito al terminar la duración normal de la carrera (rezago); **Graduate** = se gradúa.
- La probabilidad es la estimación del modelo para cada clase y las tres suman 100 %. Los modelos se entrenaron con pesos de clase balanceados, así que **no son probabilidades calibradas**: sirven para ordenar y comparar casos, no como riesgo literal («72 % de abandonar»).
- El semáforo del Modo consejero es una regla de consenso: **2 o 3 modelos** predicen Dropout → *alerta de posible abandono*; **1** → *señal mixta*; **0** → *sin alerta*.
- La clase **Enrolled** es la más difícil (F1 de 0.49 a 0.53 en Test). Un estudiante con rezago se parece a veces al que abandona y a veces al que se gradúa.
- Los modelos se entrenaron con 4,424 estudiantes de un solo instituto portugués. No hay garantía de que generalicen a otra institución, país o periodo.
- Las variables de rendimiento académico (unidades aprobadas y promedio de los dos primeros semestres) dominan la predicción, así que la señal llega después de que el estudiante ya cursó dos semestres.
- Es una herramienta de **apoyo** para priorizar tutorías y seguimiento. No debe usarse para negar becas, inscripciones u otras decisiones sobre una persona.

## Datos

**Predict Students' Dropout and Academic Success** (UCI Machine Learning Repository, id 697), licencia CC BY 4.0.
Realinho, V., Machado, J., Baptista, L. y Martins, M. V. (2022). *Predicting Student Dropout and Academic Success*. Data, 7(11), 146.
El conjunto no contiene nombres ni identificadores personales: cada fila es un estudiante anonimizado. Los archivos `datos/train.csv` y `datos/test.csv` son la partición del proyecto.
Las etiquetas en español de los códigos (carrera, vía de ingreso, escolaridad y ocupación) se tomaron del diccionario del UCI y del artículo; los códigos de ocupación 125, 173 y 191 no estaban en la lista del artículo que se usó de referencia y se etiquetaron por separado, así que conviene contrastarlos con el diccionario original antes de citarlos.

Código de la app con fines académicos.
