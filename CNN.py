"""
Autor: Georgina Castorena
Fecha: 04/04/2025
Proyecto: VisionMedic - Detección de fatiga mediante análisis facial

Descripción:
Este script forma parte del proyecto VisionMedic, una solución basada en visión por computadora para
evaluar el nivel de fatiga en rostros humanos a través del análisis de imágenes. El código utiliza redes 
neuronales convolucionales (CNN) y técnicas de preprocesamiento facial con DeepFace, además de datasets especializados.

 Este codigo entrena una CNN con salida de regresión para predecir el nivel de fatiga (0.0 a 1.0).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 🔹 Cargar los datos preprocesados
X = np.load("X.npy")
y = np.load("y.npy")

# 🔹 Normalización de etiquetas (Asegurar que estén entre 0 y 1)
y = y / np.max(y)

# 🔹 Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Construcción de la CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Salida en rango [0,1]
])

# 🔹 Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 🔹 Entrenar el modelo
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32)

# 🔹 Guardar el modelo entrenado
model.save("fatigue_model.h5")

# 🔹 Graficar la pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

print("✅ Entrenamiento completado y modelo guardado como 'fatigue_model.h5'")
