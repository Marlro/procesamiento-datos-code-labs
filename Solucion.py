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

import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("datos.csv")

# Obtener las cantidades por categoría
anémicos = df["anemia"].sum()
diabéticos = df["diabetes"].sum()
fumadores = df["is_smoker"].sum()
muertos = df["is_dead"].sum()

# Crear las gráficas de pastel
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].pie([anémicos], labels=["Anémicos"], colors=["#ff0000"])
ax[0, 1].pie([diabéticos], labels=["Diabéticos"], colors=["#00ff00"])
ax[1, 0].pie([fumadores], labels=["Fumadores"], colors=["#0000ff"])
ax[1, 1].pie([muertos], labels=["Muertos"], colors=["#ffff00"])

# Agregar título y leyenda a las gráficas
fig.suptitle("Distribución de condiciones médicas", fontsize=20)
ax[0, 0].set_title("Anémicos")
ax[0, 1].set_title("Diabéticos")
ax[1, 0].set_title("Fumadores")
ax[1, 1].set_title("Muertos")

# Mostrar la gráfica
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Cargar los datos
df = pd.read_csv("datos.csv")

# Eliminar las columnas objetivo y categoria_edad
df = df.drop(columns=["is_dead", "edad_categoría"])

# Convertir el DataFrame a un NumPy array
X = df.values

# Obtener el vector objetivo
y = df["is_dead"].values

# Realizar la reducción de dimensionalidad
X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

# Crear el gráfico de dispersión 3D
fig = px.scatter_3d(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    z=X_embedded[:, 2],
    color=y,
    color_continuous_scale="Spectral",
    title="Distribución de pacientes con insuficiencia cardíaca"
)

# Mostrar el gráfico
fig.show()

import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar los datos
df = pd.read_csv("datos.csv")

# Eliminar las columnas objetivo y edad
df = df.drop(columns=["is_dead", "edad", "edad_categoría"])

# Convertir el DataFrame a un NumPy array
X = df.values

# Obtener el vector objetivo
y = df["edad"].values

# Ajustar el modelo de regresión lineal
reg = LinearRegression()
reg.fit(X, y)

# Predecir las edades
y_pred = reg.predict(X)

# Calcular el error cuadrático medio
mse = np.mean((y_pred - y)**2)

print(f"Error cuadrático medio: {mse}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Cargar los datos
df = pd.read_csv("datos.csv")

# Eliminar la columna categoria_edad
df = df.drop(columns=["edad_categoría"])

# Obtener el vector objetivo
y = df["is_dead"].values

# Graficar la distribución de clases
plt.hist(y)
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.show()

# Partición del dataset en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df.values, y, test_size=0.25, stratify=y
)

# Ajustar un árbol de decisión
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)

# Calcular el accuracy sobre el conjunto de test
y_pred = tree.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Accuracy: {accuracy}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar los datos
df = pd.read_csv("datos.csv")

# Eliminar la columna categoria_edad
df = df.drop(columns=["edad_categoría"])

# Obtener el vector objetivo
y = df["is_dead"].values

# Partición del dataset en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df.values, y, test_size=0.25, stratify=y
)

# Ajustar un random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
rf.fit(X_train, y_train)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, rf.predict(X_test))

# Calcular F1-Score
f1 = f1_score(y_test, rf.predict(X_test))

# Imprimir el accuracy
print(f"Accuracy: {accuracy_score(y_test, rf.predict(X_test))}")

# Imprimir la matriz de confusión
print(cm)

# Imprimir el F1-Score
print(f"F1-Score: {f1}")
