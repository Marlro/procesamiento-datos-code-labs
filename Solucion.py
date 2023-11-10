from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")

data = dataset["train"]



import numpy as np

edades = data["age"]


edades_np = np.array(edades)


promedio_edad = np.mean(edades_np)

print(f"El promedio de edad de las personas participantes en el estudio es: {promedio_edad} años")

import pandas as pd
from datasets import load_dataset

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Convertir a DataFrame de Pandas
df = pd.DataFrame(data)

# Separar en dos DataFrames
df_fallecidos = df[df['is_dead'] == 1]
df_sobrevivientes = df[df['is_dead'] == 0]

# Calcular el promedio de edades para cada DataFrame
promedio_edad_fallecidos = df_fallecidos['age'].mean()
promedio_edad_sobrevivientes = df_sobrevivientes['age'].mean()

# Imprimir los resultados
print(f"Promedio de edad de personas fallecidas: {promedio_edad_fallecidos} años")
print(f"Promedio de edad de personas sobrevivientes: {promedio_edad_sobrevivientes} años")

import pandas as pd

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

df = pd.DataFrame(data)

# Verificar tipos de datos
for col in df.columns:
    print(f"La columna {col} tiene tipo de dato {df[col].dtype}")

# Calcular cantidad de hombres fumadores vs mujeres fumadoras
hombres_fumadores = df[df["is_male"] == 1][["is_smoker"]].sum()
mujeres_fumadoras = df[df["is_male"] == 0][["is_smoker"]].sum()

print(f"La cantidad de hombres fumadores es: {hombres_fumadores}")
print(f"La cantidad de mujeres fumadoras es: {mujeres_fumadoras}")

import requests
import io

def descargar_datos(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error al descargar los datos: {response.status_code}")

    with io.open("datos.csv", "w", encoding="utf-8") as f:
        f.write(response.text)

if __name__ == "__main__":
    descargar_datos("https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv")

import pandas as pd
import numpy as np

def limpiar_datos(df):
    # Verificar valores faltantes
    if df.isnull().values.any():
        raise Exception("Existen valores faltantes en los datos")

    # Verificar filas repetidas
    if df.duplicated().values.any():
        raise Exception("Existen filas repetidas en los datos")

    # Verificar valores atípicos
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Eliminar valores fuera de 3 desviaciones estándar
            df = df[np.abs(df[col] - df[col].mean()) <= 3 * df[col].std()]

    # Crear columna de categorías de edad
    df["edad_categoria"] = pd.cut(df["age"], [0, 12, 19, 39, 59, np.inf], labels=["Niño", "Adolescente", "Jóvenes adulto", "Adulto", "Adulto mayor"])

    return df

if __name__ == "__main__":
    # Descargar datos
    descargar_datos("https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv")

    # Cargar datos
    df = pd.read_csv("datos.csv")

    # Limpiar datos
    df = limpiar_datos(df)

    # Guardar datos
    df.to_csv("datos_limpios.csv")

import sys

def descargar_datos(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error al descargar los datos: {response.status_code}")

    with open("datos.csv", "w", encoding="utf-8") as f:
        f.write(response.text)

if __name__ == "__main__":
    # Obtener la URL de los datos de la línea de comandos
    if len(sys.argv) < 2:
        print("Por favor, pasa la URL de los datos como argumento.")
        exit(1)

    url = sys.argv[1]

    # Verificar que la URL tenga un esquema
    if not url.startswith("http://") or not url.startswith("https://"):
        raise Exception("La URL debe tener un esquema (http:// o https://)")

    # Descargar los datos
    descargar_datos(url)

import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("datos.csv")

# Obtener la distribución de edades
edades = df["age"]

# Graficar el histograma
plt.hist(edades, bins=100)
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()
