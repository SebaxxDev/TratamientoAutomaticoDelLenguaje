import subprocess
import sys
import spacy

## Primero se importan las librerías necesarias y se carga el CV

import pandas as pd            # manipulación de datos
import spacy                   # procesamiento de lenguaje natural
from tqdm.auto import tqdm     # barras de progreso
import json                    # manejo de archivos JSON
import re                      # expresiones regulares
from datetime import datetime  # manejo de fechas


# Instalar spacy si no está disponible

subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "-q"])

# Descargar el modelo de spaCy para español
subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_lg", "-q"])

print("Librerías importadas correctamente.")


# Cargar el dataset

df = pd.read_csv("/workspaces/TratamientoAutomaticoDelLenguaje/Datos/dataset_proyecto_chile_septiembre2025.csv")

print(df.head())
print("Dataset cargado correctamente.")



# Limpieza de Informacion

def limpiar_dataframe_basico(df_entrada):
    """Selecciona columnas relevantes, filtra textos cortos/nulos y normaliza la fecha."""
    # Seleccionar columnas y copiar el DataFrame
    df_limpio = df_entrada[['date', 'media_outlet', 'title', 'text', 'url']].copy() 

    # Eliminar filas sin título o texto
    df_limpio = df_limpio.dropna(subset=['text', 'title'])

    # Quitar textos muy cortos o basura
    df_limpio = df_limpio[df_limpio['text'].str.len() > 200].copy()

    # parsear y ordenar la fecha ("Sep 24, 2025 @ 00:00:00.000")
    def parsear_fecha(cadena_fecha):
        if pd.isna(cadena_fecha):
            return pd.NaT
        try:
            parte_fecha = str(cadena_fecha).split('@')[0].strip()
            return pd.to_datetime(parte_fecha, format='%b %d, %Y', errors='coerce')
        except Exception:
            return pd.NaT

    # Aplicar parseo y eliminar filas sin fecha válida
    df_limpio['date'] = df_limpio['date'].apply(parsear_fecha)
    df_limpio = df_limpio.dropna(subset=['date'])

    # Columna con fecha en formato YYYY-MM-DD
    df_limpio['fecha_yyyy_mm_dd'] = df_limpio['date'].dt.strftime('%Y-%m-%d')

    print(f"Filas después de limpieza: {len(df_limpio)}")
    return df_limpio

df_limpio = limpiar_dataframe_basico(df)
print(df_limpio.head())


# ==================== spaCy + Reconocimiento de Entidades Nombradas (NER) ====================

# Configurar spaCy para NER en español
nlp = spacy.load("es_core_news_lg", disable=["tagger", "parser"])  # más rápido

def extraer_entidades(texto):
    """Extrae personas, lugares y organizaciones del texto usando NER."""
    try:
        doc = nlp(texto[:500000])  # límite razonable
        personas = list({ent.text.strip() for ent in doc.ents if ent.label_ == "PER"})
        lugares = list({ent.text.strip() for ent in doc.ents if ent.label_ in ["LOC", "GPE"]})
        organizaciones = list({ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"})
        return personas, lugares, organizaciones
    except:
        return [], [], []

print("Extrayendo entidades NER (esto puede tardar 8-15 minutos con ~50k noticias)...")
tqdm.pandas()

# Aplicamos con barra de progreso
resultados = df_limpio['text'].progress_apply(extraer_entidades)

# Desempaquetamos la tupla correctamente
df_limpio[['personas', 'lugares', 'organizaciones']] = pd.DataFrame(
    resultados.tolist(),
    columns=['personas', 'lugares', 'organizaciones'],
    index=df_limpio.index
)

print("Entidades extraídas correctamente.")
print(df_limpio[['title', 'personas', 'lugares', 'organizaciones']].head())
