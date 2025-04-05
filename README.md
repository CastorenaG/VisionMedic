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

📂 Datasets Utilizados
1. AffectNet
Descripción general: AffectNet es uno de los conjuntos de datos más grandes y completos para el análisis de emociones faciales. Contiene más de 1 millón de imágenes de rostros humanos, obtenidas mediante búsquedas en internet y etiquetadas automáticamente y manualmente.

Etiquetas: Las imágenes están clasificadas en 7 emociones básicas: felicidad, tristeza, enojo, miedo, sorpresa, disgusto y neutralidad. Además, incluye etiquetas de valencia (placer) y arousal (nivel de activación), lo cual permite realizar análisis más complejos como la inferencia del estado emocional general.

Referencia: Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2017). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. IEEE Transactions on Affective Computing.

2. UTA-RLDD (University of Texas at Arlington - Real-Life Drowsiness Dataset)
Descripción general: UTA-RLDD es un dataset especializado en la detección de somnolencia en escenarios reales. Fue capturado en entornos naturales y contiene datos multimodales, incluyendo videos, secuencias de imágenes y anotaciones temporales sobre los niveles de somnolencia.

Etiquetas: Las anotaciones están divididas por niveles de alerta, incluyendo alerta, transición a somnolencia, y somnoliento. Las anotaciones se basan en marcadores como cierre ocular, cabeceo y duración del parpadeo.

Referencia: Vural, E., Cetin, M., & Littlewort, G. (2019). Real-life Drowsiness Detection Dataset: Collection and Analysis. IEEE International Conference on Automatic Face and Gesture Recognition (FG).

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


##🚀 Fases del Proyecto
🔹 Fase 1: Preprocesamiento de Imágenes
Script: Preprocesamiento.py
  Objetivo:    
    -Detectar rostros con DeepFace.
    -Redimensionar imágenes a 128x128 px en escala de grises.
    -Asignar etiquetas manuales de fatiga en un rango de 0.0 a 1.0.
  Salidas:
    -X.npy: Imágenes procesadas.
    -y.npy: Etiquetas numéricas.
    -labels.csv: Relación imagen-etiqueta.

🔹 Fase 2: Entrenamiento del Modelo
Script: CNN.py
  Objetivo:
    -Entrenar una red neuronal convolucional (CNN) con salida de regresión.
    -Predecir el nivel de fatiga a partir de imágenes faciales.
Modelo guardado:
    -fatigue_model.h5: Archivo del modelo entrenado.

🔹 Fase 3: Evaluación y Pruebas
  Script: Prueba.py
  Objetivo:
    -Cargar el modelo entrenado.
    -Evaluar el desempeño con métricas como Accuracy, Precisión,  Recall (Sensibilidad),  F1-Score.
    -Visualizar predicciones vs. valores reales.


🎯 Futuras Mejoras
Integración con video en tiempo real.

Consideración del uso de cubrebocas y equipo quirúrgico.

Entrenamiento con datasets clínicos reales y etiquetado profesional.

Panel clínico para visualización rápida del estado del equipo médico.

👨‍⚕️ Contribución
Este proyecto fue desarrollado por Georgina Elizabeth Castorena Rojas, como parte del curso de Samsung Innovation Campus - UDEM 2024 con el objetivo de aplicar lo aprendido a través del curso inteligencia artificial y liderazgo , junto con herramientas tecnológicas responsables y humanas.

📄 Licencia
Este proyecto es de uso académico. Para uso comercial, por favor contacta al equipo desarrollador.

