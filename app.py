import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NeuroVivienda | ACA 2", page_icon="🏠", layout="wide")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "modelo_mlp_california.joblib"
META_PATH = ROOT / "artifacts" / "metadata.json"

st.title("🏠 NeuroVivienda – Estimador con redes neuronales")
st.caption("ACA N.º 2 · Aprendizaje Automático · Pilar Ventas · Sector inmobiliario")
st.warning("Herramienta académica basada en California Housing (datos históricos de California). No constituye un avalúo profesional ni estima el mercado colombiano.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("No se encuentran los artefactos entrenados. Ejecute primero todas las celdas del notebook ACA2_Red_Neuronal_California_Housing.ipynb.")
    st.stop()

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta

model, meta = load_assets()
st.subheader("Ingrese las características del distrito")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        medinc = st.number_input("Ingreso medio (MedInc)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
        houseage = st.number_input("Antigüedad media (HouseAge)", min_value=1.0, max_value=60.0, value=25.0, step=1.0)
        averooms = st.number_input("Habitaciones promedio (AveRooms)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    with c2:
        avebedrms = st.number_input("Dormitorios promedio (AveBedrms)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
        population = st.number_input("Población del bloque", min_value=1.0, max_value=40000.0, value=1000.0, step=50.0)
        aveoccup = st.number_input("Ocupantes promedio (AveOccup)", min_value=0.5, max_value=20.0, value=3.0, step=0.1)
    with c3:
        latitude = st.number_input("Latitud", min_value=32.0, max_value=42.5, value=36.0, step=0.1)
        longitude = st.number_input("Longitud", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.1)
    submitted = st.form_submit_button("Calcular estimación", type="primary")

if submitted:
    row = pd.DataFrame([{
        "MedInc": medinc, "HouseAge": houseage, "AveRooms": averooms,
        "AveBedrms": avebedrms, "Population": population, "AveOccup": aveoccup,
        "Latitude": latitude, "Longitude": longitude
    }])[meta["features"]]
    pred_units = float(model.predict(row)[0])
    pred_usd = pred_units * 100000
    st.divider()
    st.subheader("Resultado de la red neuronal")
    st.metric("Valor medio estimado", f"USD ${pred_usd:,.0f}")
    st.caption(f"Predicción original: {pred_units:.3f} unidades de USD 100.000.")
    st.info("La estimación es para el valor medio del distrito representado por las variables ingresadas; no corresponde al precio garantizado de una vivienda individual.")

with st.expander("Arquitectura y métricas del modelo"):
    st.write(meta["architecture"])
    st.write("Variables:", ", ".join(meta["features"]))
    st.json(meta.get("metrics", {}))

st.markdown("---")
st.caption("Proyecto académico ACA 2 · CUN · Grupo 17")
