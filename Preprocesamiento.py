"""
Autor: Georgina Castorena
Fecha: 04/04/2025
Proyecto: VisionMedic - Detección de fatiga mediante análisis facial

Descripción:
Este script forma parte del proyecto VisionMedic, una solución basada en visión por computadora para
evaluar el nivel de fatiga en rostros humanos a través del análisis de imágenes. El código utiliza redes 
neuronales convolucionales (CNN) y técnicas de preprocesamiento facial con DeepFace, además de datasets especializados.

Este codigo preprocesa imágenes, detecta rostros, aplica normalización y genera las etiquetas aumtomaticas asi como manuales.
"""

import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import random

# 🔹 Configuración
IMG_SIZE = (128, 128)
INPUT_FOLDER = "0"
OUTPUT_IMAGES = "X.npy"
OUTPUT_LABELS = "y.npy"
OUTPUT_CSV = "labels.csv"

# 🔹 Rango de cansancio por emoción
EMOTION_RANGES = {
    "angry": (0.80, 0.90),
    "fear": (0.85, 0.95),
    "sad": (0.70, 0.80),
    "neutral": (0.45, 0.55),
    "happy": (0.05, 0.15),
    "surprise": (0.45, 0.55),
    "disgust": (0.65, 0.75)
}

# 🔹 Etiquetas manuales por nombre de carpeta
CUSTOM_LABELS = {
    "fatigue": (0.85, 1.0),
    "active": (0.0, 0.2)
}

def get_random_in_range(value_range):
    return round(random.uniform(*value_range), 2)

def assign_manual_label_from_folder(folder_path):
    for keyword, val_range in CUSTOM_LABELS.items():
        if keyword in folder_path.lower():
            return get_random_in_range(val_range)
    return None

def analyze_emotion(img_path):
    try:
        analysis = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        value_range = EMOTION_RANGES.get(emotion, (0.45, 0.55))
        fatigue_level = get_random_in_range(value_range)
        print(f"🧠 Emoción detectada: {emotion} → Cansancio estimado: {fatigue_level*100:.0f}%")
        return fatigue_level
    except Exception as e:
        print(f"⚠️ Error analizando imagen {img_path}: {e}")
        return None

def preprocess_images(folder):
    images, labels, filenames = [], [], []

    if not os.path.exists(folder):
        print("⚠️ La carpeta no existe. Verifica la ruta.")
        return None, None, None

    print("🔍 Buscando imágenes en carpetas y subcarpetas...")
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(dirpath, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ No se pudo leer la imagen {filename}.")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, IMG_SIZE)
            img_normalized = img_resized / 255.0

            label = assign_manual_label_from_folder(dirpath)
            if label is None:
                label = analyze_emotion(img_path)

            if label is not None:
                images.append(img_normalized.reshape(128, 128, 1))
                labels.append(label)
                filenames.append(os.path.relpath(img_path, folder))  # guarda ruta relativa
            else:
                print(f"⚠️ Imagen sin etiqueta válida: {filename}")

            if len(images) % 10 == 0:
                print(f"📦 Procesadas {len(images)} imágenes...")

    print(f"✅ Total de imágenes procesadas: {len(images)}")
    return np.array(images), np.array(labels), filenames

# 🔹 Ejecutar preprocesamiento
X, y, filenames = preprocess_images(INPUT_FOLDER)

if X is not None and y is not None:
    np.save(OUTPUT_IMAGES, X)
    np.save(OUTPUT_LABELS, y)

    df = pd.DataFrame({"filename": filenames, "fatigue_level": y})
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Preprocesamiento completado. Guardado en {OUTPUT_IMAGES}, {OUTPUT_LABELS}, {OUTPUT_CSV}.")
else:
    print("❌ Error: No se pudieron procesar las imágenes.")
