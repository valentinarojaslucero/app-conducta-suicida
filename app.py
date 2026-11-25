"""
Producción – Modelo Random Forest (Conducta suicida)
Angie Paola Real · Valentina Rojas

Ejecutar:  streamlit run app.py
Requiere en la misma carpeta: modelo_rf.joblib (y opcionalmente metadata_modelo.json)
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificador de conducta suicida", page_icon="🧠", layout="centered")

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "modelo_rf.joblib")
RUTA_META = os.path.join(CARPETA, "metadata_modelo.json")


# ---------------------------------------------------------------------------
# 1. Carga del modelo y metadatos
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)


@st.cache_data
def cargar_metadata(ruta):
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {}


if not os.path.exists(RUTA_MODELO):
    st.error("No se encontró `modelo_rf.joblib` en la carpeta de la app.")
    st.stop()

modelo = cargar_modelo(RUTA_MODELO)
metadata = cargar_metadata(RUTA_META)

# ---------------------------------------------------------------------------
# 2. Configuración de variables
# ---------------------------------------------------------------------------
CATEGORICAS = metadata.get(
    "categorical_features",
    ["ciclo_vital", "sexo", "nivel_educativo", "poblacion_diferencial", "localidad"],
)
BINARIAS = metadata.get(
    "binary_features",
    ["enfermedades_dolorosas", "maltrato_sexual", "muerte_familiar", "conflicto_pareja",
     "problemas_economicos", "problemas_escolares", "problemas_juridicos",
     "problemas_laborales", "suicidio_amigo"],
)
OPCIONES_RESPALDO = {
    "ciclo_vital": ["0 – 11 Infancia", "12 – 17 Adolescencia", "18 – 28 Juventud",
                    "29 – 59 Adultez", ">60 Vejez"],
    "sexo": ["Hombre", "Mujer"],
    "nivel_educativo": ["Ninguno", "Posgrado", "Preescolar/Primaria",
                        "Secundaria", "Técnico", "Universitario"],
    "poblacion_diferencial": ["Consumidor habitual de spa", "En protección",
        "Madre cabeza de hogar", "Menor abandonado", "Ninguna / no especificada",
        "Otro tipo de víctima del conflicto", "Población lgbti",
        "Servidor público", "Trabajador informal"],
    "localidad": ["Antonio Nariño", "Barrios Unidos", "Bosa", "Chapinero",
        "Ciudad Bolívar", "Engativá", "Fontibón", "Kennedy", "La Candelaria",
        "Los Mártires", "Puente Aranda", "Rafael Uribe Uribe", "San Cristóbal",
        "Santa Fe", "Sin dato", "Suba", "Sumapaz", "Teusaquillo", "Tunjuelito",
        "Usaquén", "Usme"],
}
opc_meta = metadata.get("feature_options", metadata.get("categorical_values", {}))
OPCIONES = {c: opc_meta.get(c, OPCIONES_RESPALDO.get(c, [])) for c in CATEGORICAS}

ORDEN = metadata.get("input_features_order", CATEGORICAS + BINARIAS)
UMBRAL = float(metadata.get("threshold", 0.2160863722223444))

NOMBRES = {
    "ciclo_vital": "Ciclo vital", "sexo": "Sexo", "nivel_educativo": "Nivel educativo",
    "poblacion_diferencial": "Población", "localidad": "Localidad",
    "enfermedades_dolorosas": "Enfermedades dolorosas", "maltrato_sexual": "Maltrato sexual",
    "muerte_familiar": "Muerte de un familiar", "conflicto_pareja": "Conflicto de pareja",
    "problemas_economicos": "Problemas económicos", "problemas_escolares": "Problemas escolares",
    "problemas_juridicos": "Problemas jurídicos", "problemas_laborales": "Problemas laborales",
    "suicidio_amigo": "Suicidio de un amigo",
}

# Casos de ejemplo donde P(intento) > P(ideación)
CASOS = {
    "Caso A (≈60 % intento)": {
        "ciclo_vital": "18 – 28 Juventud", "sexo": "Hombre", "nivel_educativo": "Secundaria",
        "poblacion_diferencial": "Población lgbti", "localidad": "Bosa",
        "enfermedades_dolorosas": 1, "maltrato_sexual": 0, "muerte_familiar": 1,
        "conflicto_pareja": 1, "problemas_economicos": 1, "problemas_escolares": 1,
        "problemas_juridicos": 0, "problemas_laborales": 1, "suicidio_amigo": 1,
    },
    "Caso B (≈55 % intento)": {
        "ciclo_vital": "18 – 28 Juventud", "sexo": "Hombre", "nivel_educativo": "Preescolar/Primaria",
        "poblacion_diferencial": "Consumidor habitual de spa", "localidad": "Barrios Unidos",
        "enfermedades_dolorosas": 0, "maltrato_sexual": 0, "muerte_familiar": 1,
        "conflicto_pareja": 1, "problemas_economicos": 1, "problemas_escolares": 0,
        "problemas_juridicos": 0, "problemas_laborales": 0, "suicidio_amigo": 1,
    },
}


def cargar_caso(caso):
    for c in CATEGORICAS:
        if caso.get(c) in OPCIONES[c]:
            st.session_state[c] = caso[c]
    for b in BINARIAS:
        st.session_state[b] = bool(caso.get(b, 0))
    st.session_state.pop("resultado", None)


def limpiar():
    for c in CATEGORICAS:
        st.session_state[c] = OPCIONES[c][0]
    for b in BINARIAS:
        st.session_state[b] = False
    st.session_state.pop("resultado", None)


# ---------------------------------------------------------------------------
# 3. Interfaz
# ---------------------------------------------------------------------------
st.title("Clasificador de conducta suicida")
st.caption("Modelo Random Forest · Angie Paola Real · Valentina Rojas")

with st.sidebar:
    st.header("Casos de ejemplo")
    for nombre, caso in CASOS.items():
        st.button(nombre, on_click=cargar_caso, args=(caso,), use_container_width=True)
    st.button("Limpiar formulario", on_click=limpiar, use_container_width=True)
    st.divider()
    st.metric("Umbral de decisión", f"{UMBRAL:.1%}")
    st.caption("Se clasifica como intento si la probabilidad de intento es mayor o igual al umbral.")

with st.form("formulario"):
    st.subheader("Características del registro")
    col1, col2 = st.columns(2)
    for i, c in enumerate(CATEGORICAS):
        with (col1 if i % 2 == 0 else col2):
            st.selectbox(NOMBRES.get(c, c), OPCIONES[c], key=c)

    st.subheader("Factores presentes")
    col1, col2 = st.columns(2)
    for i, b in enumerate(BINARIAS):
        with (col1 if i % 2 == 0 else col2):
            st.checkbox(NOMBRES.get(b, b), key=b)

    enviado = st.form_submit_button("Clasificar registro", type="primary", use_container_width=True)

if enviado:
    registro = {c: st.session_state[c] for c in CATEGORICAS}
    registro.update({b: int(st.session_state[b]) for b in BINARIAS})
    X = pd.DataFrame([registro])[ORDEN]
    st.session_state["resultado"] = float(modelo.predict_proba(X)[0, 1])

if "resultado" in st.session_state:
    p_intento = st.session_state["resultado"]
    p_ideacion = 1 - p_intento
    es_intento = p_intento >= UMBRAL

    st.divider()
    st.subheader("Resultado")
    if es_intento:
        st.error("Clasificación: **INTENTO DE SUICIDIO**")
    else:
        st.info("Clasificación: **IDEACIÓN SUICIDA**")

    m1, m2 = st.columns(2)
    m1.metric("Probabilidad de intento", f"{p_intento:.1%}")
    m2.metric("Probabilidad de ideación", f"{p_ideacion:.1%}")
    st.progress(p_intento, text=f"Intento {p_intento:.1%} · umbral {UMBRAL:.1%}")

    if p_intento > p_ideacion:
        st.caption("En este registro la probabilidad de intento es mayor que la de ideación.")

    st.warning(
        "Resultado experimental de apoyo analítico. No es un diagnóstico, "
        "no descarta riesgo y siempre requiere revisión profesional."
    )
