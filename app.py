"""
Clasificador de conducta suicida – Modelo Random Forest
Angie Paola Real · Valentina Rojas

Ejecutar:  streamlit run app.py
Requiere en la misma carpeta: modelo_rf.joblib (y opcionalmente metadata_modelo.json)
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Clasificador de conducta suicida",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "modelo_rf.joblib")
RUTA_META = os.path.join(CARPETA, "metadata_modelo.json")

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #f4f6f9; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* Encabezado */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5a7f 100%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    color: #fff;
    margin-bottom: 1.6rem;
}
.hero .tag {
    display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; background: rgba(255,255,255,.14);
    padding: .3rem .7rem; border-radius: 999px; margin-bottom: .8rem;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 .35rem 0; color: #fff; }
.hero p  { margin: 0; opacity: .85; font-size: .95rem; }

/* Tarjetas */
[data-testid="stForm"], .card {
    background: #fff;
    border: 1px solid #e3e8ef;
    border-radius: 16px;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 1px 3px rgba(16,24,40,.04);
}
.sec-title {
    font-size: .78rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
    color: #5b6b80; margin: .2rem 0 .9rem 0;
}
.sec-title.mt { margin-top: 1.6rem; }

/* Botón */
.stFormSubmitButton button {
    background: #1e3a5f !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: .7rem 0 !important; font-weight: 600 !important;
    margin-top: .8rem;
}
.stFormSubmitButton button:hover { background: #2c5a7f !important; }

/* Resultado */
.res-label { font-size: .78rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: #5b6b80; }
.res-title { font-size: 1.55rem; font-weight: 700; margin: .3rem 0 1.3rem 0; }
.res-title.intento  { color: #b4533a; }
.res-title.ideacion { color: #2c5a7f; }
.chip {
    display: inline-block; font-size: .75rem; font-weight: 600; padding: .25rem .65rem;
    border-radius: 999px; margin-left: .4rem; vertical-align: middle;
}
.chip.intento  { background: #fbeae5; color: #b4533a; }
.chip.ideacion { background: #e6eef5; color: #2c5a7f; }

.bar-wrap { position: relative; margin: .4rem 0 .5rem 0; }
.bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: #eef1f5; }
.seg-int { background: #d9735a; }
.seg-ide { background: #7fa3c2; }
.umbral {
    position: absolute; top: -5px; width: 2px; height: 24px; background: #1e3a5f; border-radius: 2px;
}
.umbral-txt { position: absolute; top: 22px; font-size: .7rem; color: #5b6b80; transform: translateX(-50%); white-space: nowrap; }

.stats { display: flex; gap: 1rem; margin-top: 2.2rem; }
.stat { flex: 1; border-radius: 12px; padding: 1rem 1.1rem; }
.stat.intento  { background: #fbeae5; }
.stat.ideacion { background: #e6eef5; }
.stat .k { font-size: .8rem; color: #5b6b80; font-weight: 500; }
.stat .v { font-size: 1.9rem; font-weight: 700; margin-top: .15rem; }
.stat.intento  .v { color: #b4533a; }
.stat.ideacion .v { color: #2c5a7f; }

.nota {
    margin-top: 1.3rem; font-size: .82rem; color: #5b6b80; line-height: 1.5;
    border-top: 1px solid #eef1f5; padding-top: 1rem;
}
.aviso {
    margin-top: 1rem; background: #fff8e6; border: 1px solid #f3e2b3; color: #7a5b12;
    border-radius: 12px; padding: .9rem 1rem; font-size: .82rem; line-height: 1.5;
}
.vacio { text-align: center; padding: 2.5rem 1rem; color: #7b8a9c; }
.vacio .ico { font-size: 2.2rem; margin-bottom: .6rem; }
.vacio b { color: #1e3a5f; }

.pie { text-align: center; font-size: .78rem; color: #8a97a8; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Modelo y metadatos
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
    "poblacion_diferencial": "Población diferencial", "localidad": "Localidad",
    "enfermedades_dolorosas": "Enfermedades dolorosas", "maltrato_sexual": "Maltrato sexual",
    "muerte_familiar": "Muerte de un familiar", "conflicto_pareja": "Conflicto de pareja",
    "problemas_economicos": "Problemas económicos", "problemas_escolares": "Problemas escolares",
    "problemas_juridicos": "Problemas jurídicos", "problemas_laborales": "Problemas laborales",
    "suicidio_amigo": "Suicidio de un amigo",
}

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <span class="tag">Modelo Random Forest · Bogotá</span>
  <h1>Clasificador de conducta suicida</h1>
  <p>Estima si un registro corresponde a <b>intento</b> o a <b>ideación</b> suicida a partir de sus
  características y factores asociados. · Angie Paola Real · Valentina Rojas</p>
</div>
""",
    unsafe_allow_html=True,
)

col_form, col_res = st.columns([1.35, 1], gap="large")

# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------
with col_form:
    with st.form("formulario", border=False):
        st.markdown('<div class="sec-title">Características del registro</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        valores = {}
        for i, c in enumerate(CATEGORICAS):
            with (c1 if i % 2 == 0 else c2):
                valores[c] = st.selectbox(
                    NOMBRES.get(c, c), OPCIONES[c], index=None,
                    placeholder="Selecciona una opción", key=c,
                )

        st.markdown('<div class="sec-title mt">Factores presentes</div>', unsafe_allow_html=True)
        st.caption("Activa los factores que estén presentes en el registro.")
        b1, b2 = st.columns(2)
        for i, b in enumerate(BINARIAS):
            with (b1 if i % 2 == 0 else b2):
                valores[b] = st.toggle(NOMBRES.get(b, b), key=b)

        enviado = st.form_submit_button("Clasificar registro", use_container_width=True)

if enviado:
    faltan = [NOMBRES.get(c, c) for c in CATEGORICAS if valores[c] is None]
    if faltan:
        st.session_state.pop("resultado", None)
        st.session_state["faltan"] = faltan
    else:
        st.session_state.pop("faltan", None)
        registro = {c: valores[c] for c in CATEGORICAS}
        registro.update({b: int(valores[b]) for b in BINARIAS})
        X = pd.DataFrame([registro])[ORDEN]
        st.session_state["resultado"] = float(modelo.predict_proba(X)[0, 1])
        st.session_state["n_factores"] = sum(int(valores[b]) for b in BINARIAS)

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
with col_res:
    if "faltan" in st.session_state:
        st.markdown(
            f"""<div class="card"><div class="vacio"><div class="ico">📝</div>
            Falta completar: <b>{", ".join(st.session_state["faltan"])}</b>.</div></div>""",
            unsafe_allow_html=True,
        )
    elif "resultado" not in st.session_state:
        st.markdown(
            """<div class="card"><div class="vacio"><div class="ico">🧭</div>
            Completa el formulario y presiona <b>Clasificar registro</b><br>para ver el resultado aquí.
            </div></div>""",
            unsafe_allow_html=True,
        )
    else:
        p_int = st.session_state["resultado"]
        p_ide = 1 - p_int
        es_intento = p_int >= UMBRAL
        clase = "intento" if es_intento else "ideacion"
        titulo = "Intento de suicidio" if es_intento else "Ideación suicida"
        dominante = "intento" if p_int > p_ide else "ideación"
        n_fac = st.session_state.get("n_factores", 0)

        st.markdown(
            f"""
<div class="card">
  <div class="res-label">Clasificación del modelo
    <span class="chip {clase}">{"≥" if es_intento else "<"} umbral</span>
  </div>
  <div class="res-title {clase}">{titulo}</div>

  <div class="bar-wrap">
    <div class="bar">
      <div class="seg-int" style="width:{p_int*100:.2f}%"></div>
      <div class="seg-ide" style="width:{p_ide*100:.2f}%"></div>
    </div>
    <div class="umbral" style="left:calc({UMBRAL*100:.2f}% - 1px)"></div>
    <div class="umbral-txt" style="left:{UMBRAL*100:.2f}%">umbral {UMBRAL:.1%}</div>
  </div>

  <div class="stats">
    <div class="stat intento"><div class="k">Probabilidad de intento</div><div class="v">{p_int:.1%}</div></div>
    <div class="stat ideacion"><div class="k">Probabilidad de ideación</div><div class="v">{p_ide:.1%}</div></div>
  </div>

  <div class="nota">
    Predomina la probabilidad de <b>{dominante}</b>. Factores marcados: <b>{n_fac} de {len(BINARIAS)}</b>.<br>
    El registro se clasifica como intento cuando su probabilidad de intento es igual o mayor
    al umbral de {UMBRAL:.1%}, definido durante el entrenamiento.
  </div>

  <div class="aviso">
    ⚠️ Resultado experimental de apoyo analítico. No es un diagnóstico, no descarta riesgo
    y siempre requiere revisión profesional.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    """<div class="pie">Si tú o alguien cercano necesita apoyo en Bogotá: Línea 106 ·
    Línea nacional de salud mental 192, opción 4.</div>""",
    unsafe_allow_html=True,
)
