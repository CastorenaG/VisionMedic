**VisionMedic** es un sistema innovador basado en inteligencia artificial y reconocimiento facial, diseñado para evaluar el estado físico y emocional del personal médico antes de realizar procedimientos quirúrgicos.

Este proyecto fue desarrollado como parte del programa de innovación tecnológica impartido por **Samsung Innovation Campus** en colaboración con la **Universidad de Monterrey (UDEM)**.

## 🎯 Objetivo

El objetivo principal de VisionMedic es **mejorar la seguridad del paciente** mediante la **detección no invasiva de signos de fatiga y estrés** en los médicos, contribuyendo así a la **optimización de la atención médica**.

## 🧠 ¿Cómo funciona?

VisionMedic utiliza redes neuronales convolucionales (CNN) y modelos de visión por computadora para analizar expresiones faciales y otros indicadores biométricos. El sistema:

- Evalúa imágenes del rostro del personal médico.
- Estima niveles de **cansancio**, **estrés** y **estado emocional**.
- Proporciona retroalimentación antes de que el médico entre a cirugía.
- Funciona como herramienta **continua y no invasiva** para mejorar la toma de decisiones clínicas.

## 💡 Beneficios

- ✅ Mejora la seguridad en quirófano.
- ✅ Promueve el bienestar del personal médico.
- ✅ Integra IA de manera ética y responsable en entornos clínicos.
- ✅ Apoya decisiones en tiempo real sin interferir con el flujo de trabajo hospitalario.

## 🧪 Tecnologías utilizadas

- Python
- OpenCV
- DeepFace / Dlib
- TensorFlow / Keras
- CNN (Convolutional Neural Networks)
- Dataset: AffectNet, UTA-RLDD, Face Mask Detection Dataset

## 📁 Estructura del Proyecto
VisionMedic/ 
│ ├── preprocessing.py # Script de preprocesamiento con DeepFace y etiquetas manuales 
├── train_cnn.py # Entrenamiento del modelo CNN con salida de regresión 
├── test_model.py # Pruebas del modelo y visualización de métricas 
│ ├── X.npy # Imágenes procesadas (grises, 128x128) 
├── y.npy # Etiquetas de nivel de fatiga [0.0, 1.0] 
├── labels.csv # Archivo CSV con etiquetas y nombres de imagen 
├── fatigue_model.h5 # Modelo entrenado guardado 
│ └── README.md # Documentación del proyecto (este archivo)


## 🧠 Tecnologías y Librerías
- Python 3.8+
- [DeepFace](https://github.com/serengil/deepface)
- OpenCV
- TensorFlow / Keras
- NumPy,Pandas, Matplotlib, Seaborn
- scikit-learn


🚀 Fases del Proyecto
🔹 Fase 1: Preprocesamiento de Imágenes
Script: preprocessing.py

Objetivo:

Detectar rostros con DeepFace.

Redimensionar imágenes a 128x128 px en escala de grises.

Asignar etiquetas manuales de fatiga en un rango de 0.0 a 1.0.

Salidas:

X.npy: Imágenes procesadas.

y.npy: Etiquetas numéricas.

labels.csv: Relación imagen-etiqueta.

🔹 Fase 2: Entrenamiento del Modelo
Script: train_cnn.py

Objetivo:

Entrenar una red neuronal convolucional (CNN) con salida de regresión.

Predecir el nivel de fatiga a partir de imágenes faciales.

Modelo guardado:

fatigue_model.h5: Archivo del modelo entrenado.

🔹 Fase 3: Evaluación y Pruebas
Script: test_model.py

Objetivo:

Cargar el modelo entrenado.

Evaluar el desempeño con métricas como MAE, RMSE, R², etc.

Visualizar predicciones vs. valores reales.

🔹 Fase 4: Documentación
Archivo: README.md

Contiene:

Descripción general del proyecto.

Estructura de archivos.

Instrucciones de uso.

Referencias de datasets y modelos utilizados.

🎯 Futuras Mejoras
Integración con video en tiempo real.

Consideración del uso de cubrebocas y equipo quirúrgico.

Entrenamiento con datasets clínicos reales y etiquetado profesional.

Panel clínico para visualización rápida del estado del equipo médico.

👨‍⚕️ Contribución
Este proyecto fue desarrollado por Georgina Elizabeth Castorena Rojas, como parte del curso de Samsung Innovation Campus - UDEM 2024 con el objetivo de aplicar lo aprendido a través del curso inteligencia artificial y liderazgo , junto con herramientas tecnológicas responsables y humanas.

📄 Licencia
Este proyecto es de uso académico. Para uso comercial, por favor contacta al equipo desarrollador.

