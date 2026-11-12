# ACA N.º 2 — Predicción del valor de viviendas con una red neuronal (MLP)

**CUN · Especialización en Inteligencia Artificial · Aprendizaje Automático**
**Pilar:** Ventas — sector inmobiliario
**Integrantes:** Anyela María Torres Sánchez · Brandon Andrés Ardila Meta · Edwin Arturo Contreras Padua

## Contenido del paquete

| Archivo / carpeta | Descripción |
|---|---|
| `ACA2_Red_Neuronal_California_Housing.ipynb` | Notebook con todo el flujo: EDA, SQL, preparación, red neuronal, evaluación y artefactos |
| `app.py` | Aplicación interactiva en Streamlit |
| `data/california_housing.db` | Base SQLite con la tabla `housing` |
| `artifacts/` | Modelo entrenado, métricas, metadatos y gráficas |
| `Poster_ACA2.pdf` | Póster tipo infografía |
| `requirements.txt` | Dependencias del proyecto |

## Cómo ejecutar

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar el notebook **de arriba hacia abajo** (genera `data/` y `artifacts/`):
   ```bash
   jupyter notebook ACA2_Red_Neuronal_California_Housing.ipynb
   ```

3. Ejecutar la aplicación Streamlit:
   ```bash
   streamlit run app.py
   ```
   La app carga `artifacts/modelo_mlp_california.joblib` y `artifacts/metadata.json`; si no existen, muestra un aviso para ejecutar primero el notebook.

## Resultados del modelo (conjunto de prueba)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| MLP (128-64-32) | 0.349 | 0.523 | 0.792 |
| Dummy (media) | 0.909 | 1.149 | ≈ 0.000 |

Las métricas están en unidades de USD 100.000. Reproducible con `random_state = 42`.

## Nota

California Housing contiene datos **históricos de California**. La salida es una estimación académica del valor medio del distrito: no representa el mercado colombiano ni sustituye un avalúo profesional.

## Enlaces

- **Aplicación desplegada:** https://aca2-neurovivienda-grupo17.streamlit.app
- Video de sustentación: https://www.youtube.com/watch?v=FP9k7KeUqiY
- Repositorio: https://github.com/edwincontri/aca2-neurovivienda
