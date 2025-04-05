"""
Autor: Georgina Castorena
Fecha: 04/04/2025
Proyecto: VisionMedic - Detección de fatiga mediante análisis facial

Descripción:
Este script forma parte del proyecto VisionMedic, una solución basada en visión por computadora para
evaluar el nivel de fatiga en rostros humanos a través del análisis de imágenes. El código utiliza redes 
neuronales convolucionales (CNN) y técnicas de preprocesamiento facial con DeepFace, además de datasets especializados.

Este codigo evalúa el modelo entrenado, calcula métricas de desempeño y muestra visualizaciones.
"""

import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 🔹 Cargar el modelo entrenado
model = tf.keras.models.load_model("fatigue_model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# 🔹 Definir la ruta de la carpeta de prueba (AJUSTAR según tu ruta)
test_dir = r'C:\Users\Lenovo\OneDrive\Escritorio\Samsung\NewProof\TestDB'

# 🔹 Etiquetas: 1 para fatiga, 0 para activo
categories = ["fatigue", "active"]
labels = {"fatigue": 1, "active": 0}

# 🔹 Preprocesar imágenes de prueba
X_test = []
y_test = []

for category in categories:
    folder = os.path.join(test_dir, category)
    if not os.path.exists(folder):
        print(f"⚠️ Carpeta no encontrada: {folder}")
        continue
    
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        # Cargar imagen en escala de grises y redimensionar a (128,128)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ No se pudo cargar: {img_path}")
            continue

        img = cv2.resize(img, (128, 128))  
        img = img / 255.0  # Normalización
        
        X_test.append(img)
        y_test.append(labels[category])  # 1 si es "fatigue", 0 si es "active"

# 🔹 Convertir listas a arrays de NumPy
X_test = np.array(X_test).reshape(-1, 128, 128, 1)  
y_test = np.array(y_test)

print(f"✅ Datos de prueba cargados: {len(X_test)} imágenes")

# 🔹 Realizar predicciones
predictions = model.predict(X_test)
predictions_bin = (predictions > 0.5).astype(int)  # Umbral de 0.5 para binarizar

# 🔹 Calcular métricas de evaluación
accuracy = accuracy_score(y_test, predictions_bin)
precision = precision_score(y_test, predictions_bin)
recall = recall_score(y_test, predictions_bin)
f1 = f1_score(y_test, predictions_bin)

print(f"📊 Resultados de Evaluación:")
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"✅ Precisión: {precision:.2%}")
print(f"✅ Recall (Sensibilidad): {recall:.2%}")
print(f"✅ F1-Score: {f1:.2%}")

# 🔹 Matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions_bin)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Active", "Fatigue"], yticklabels=["Active", "Fatigue"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# 🔹 Mostrar algunos ejemplos con predicciones
for i in range(5):  # Muestra 5 ejemplos
    plt.imshow(X_test[i].reshape(128, 128), cmap='gray')
    pred_label = "Fatigue" if predictions_bin[i] == 1 else "Active"
    true_label = "Fatigue" if y_test[i] == 1 else "Active"
    plt.title(f"Predicción: {pred_label} | Real: {true_label}")
    plt.show()

# 🔹 Graficar la distribución de las predicciones
plt.figure(figsize=(6, 4))
sns.countplot(x=predictions_bin.flatten(), palette="Set2")
plt.title("Distribución de las Predicciones")
plt.xlabel("Predicción (0: Activo, 1: Fatiga)")
plt.ylabel("Cantidad de Imágenes")
plt.show()

# 🔹 Graficar la distribución de las etiquetas reales
plt.figure(figsize=(6, 4))
sns.countplot(x=y_test, palette="Set1")
plt.title("Distribución de las Etiquetas Reales")
plt.xlabel("Etiqueta Real (0: Activo, 1: Fatiga)")
plt.ylabel("Cantidad de Imágenes")
plt.show()
