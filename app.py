import streamlit as st
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from datetime import datetime
from utils import *

#Caracteristicas basicas de la pagina
st.set_page_config(page_icon="🤖", page_title="Proyecto_Deteccion", layout="wide")
#hora_actual = datetime.now().strftime("%H:%M:%S")
#st.subheader(f"Hora actual: {hora_actual}")
#st.title("Deteccion de anomalias en Series de Tiempo en Cardiografia")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

UMBRAL = 0.089

with c30:
    uploaded_file = st.file_uploader(
        "", type = 'pkl',
        key="1",
    )


    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        info_box_wait = st.info(
            f"""
                Realizando la clasificacion...
                """)

        # Acá viene la predicción con el modelo
        dato = leer_dato(uploaded_file)
        autoencoder = Autoencoder()
        autoencoder = cargar_modelo_preentrenado('autoencoder.pth')
        prediccion = predecir(autoencoder, dato, UMBRAL)
        categoria = obtener_categoria(prediccion)

        datos_series_temporales = leer_dato(uploaded_file)

            # Mostrar la grafica estilo cardiograma
        #st.subheader("Cardiograma")

            # Supongamos que tus datos tienen un formato de tiempo y valor
        #tiempo = np.arange(len(datos_series_temporales))
        #valores = datos_series_temporales

            # Crear un gráfico estilo cardiograma
       # fig, ax = plt.subplots(figsize=(10, 6))
        #ax.plot(tiempo, valores, color='blue', linewidth=2, label='Cardiograma')
        #ax.set_xlabel('Tiempo', fontsize=14)
        #ax.set_ylabel('Valor', fontsize=14)
        #ax.set_title('Representacion del Cardiograma', fontsize=16)
        #ax.legend()
        #ax.grid(True, linestyle='--', alpha=0.7)
        #st.pyplot(fig)

            
        info_box_result = st.info(f"""
        	El dato analizado corresponde a un sujeto: {categoria}
        	""")

    else:
        st.info(
            f"""
                👆 Debe cargar primero un dato con extension .pkl
                """
        )

        st.stop()
