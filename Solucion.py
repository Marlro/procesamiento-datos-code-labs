from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")

data = dataset["train"]



import numpy as np

edades = data["age"]


edades_np = np.array(edades)


promedio_edad = np.mean(edades_np)

print(f"El promedio de edad de las personas participantes en el estudio es: {promedio_edad} años")
