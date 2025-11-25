# setup-project.ps1
# Script para crear la estructura del proyecto CoreFace-AI

Write-Host "Creando estructura del proyecto CoreFace-AI..." -ForegroundColor Green
Write-Host ""

# Crear estructura de directorios
$directories = @(
    "docker",
    "data/raw",
    "data/processed/train",
    "data/processed/validation",
    "data/processed/test",
    "data/augmented",
    "models/saved_models",
    "models/checkpoints",
    "models/pretrained",
    "models/metadata",
    "src/data",
    "src/models",
    "src/training",
    "src/inference",
    "src/utils",
    "notebooks",
    "tests",
    "scripts",
    "logs",
    "outputs/predictions",
    "outputs/visualizations"
)

Write-Host "Creando directorios..." -ForegroundColor Cyan
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "  - $dir" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Creando archivos..." -ForegroundColor Cyan

# ============== DOCKER ==============

# Dockerfile
$dockerfileContent = @'
# docker/Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Guayaquil

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888 6006

CMD ["python", "src/inference/recognize.py"]
'@
$dockerfileContent | Out-File -FilePath "docker/Dockerfile" -Encoding UTF8 -NoNewline

# Dockerfile.gpu
$dockerfileGpuContent = @'
# docker/Dockerfile.gpu
FROM tensorflow/tensorflow:2.14.0-gpu

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Guayaquil

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-gpu.txt

COPY . .

EXPOSE 8888 6006

CMD ["python", "src/training/train.py"]
'@
$dockerfileGpuContent | Out-File -FilePath "docker/Dockerfile.gpu" -Encoding UTF8 -NoNewline

# docker-compose.yml
$dockerComposeContent = @'
version: '3.8'

services:
  coreface-cpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: coreface-cpu
    volumes:
      - ../:/app
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
    ports:
      - "8888:8888"
      - "6006:6006"
    environment:
      - DISPLAY=host.docker.internal:0
    command: python src/inference/recognize.py
    
  coreface-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu
    container_name: coreface-gpu
    runtime: nvidia
    volumes:
      - ../:/app
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
    ports:
      - "8889:8888"
      - "6007:6006"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: python src/training/train.py
    
  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: coreface-jupyter
    volumes:
      - ../:/app
    ports:
      - "8888:8888"
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    
  tensorboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: coreface-tensorboard
    volumes:
      - ../logs:/app/logs
    ports:
      - "6006:6006"
    command: tensorboard --logdir=/app/logs --host=0.0.0.0
'@
$dockerComposeContent | Out-File -FilePath "docker/docker-compose.yml" -Encoding UTF8 -NoNewline

# requirements.txt
$requirementsContent = @'
# Core
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0

# Deep Learning
tensorflow==2.14.0
keras==2.14.0

# Computer Vision
opencv-python==4.8.0.76
opencv-contrib-python==4.8.0.76
mtcnn==0.1.1

# Utils
scikit-learn==1.3.0
scipy==1.11.2
tqdm==4.66.1
python-dotenv==1.0.0

# Visualization
tensorboard==2.14.0
jupyter==1.0.0
ipykernel==6.25.1

# API
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0

# Testing
pytest==7.4.2
pytest-cov==4.1.0
'@
$requirementsContent | Out-File -FilePath "requirements.txt" -Encoding UTF8 -NoNewline

# requirements-gpu.txt
$requirementsGpuContent = @'
# Core
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0

# Deep Learning (GPU)
tensorflow[and-cuda]==2.14.0

# Computer Vision
opencv-python==4.8.0.76
opencv-contrib-python==4.8.0.76
mtcnn==0.1.1

# Utils
scikit-learn==1.3.0
scipy==1.11.2
tqdm==4.66.1
python-dotenv==1.0.0

# Visualization
tensorboard==2.14.0
jupyter==1.0.0
ipykernel==6.25.1

# API
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0

# Testing
pytest==7.4.2
pytest-cov==4.1.0
'@
$requirementsGpuContent | Out-File -FilePath "requirements-gpu.txt" -Encoding UTF8 -NoNewline

# src/__init__.py
$srcInitContent = @'
"""
CoreFace-AI: Sistema de Reconocimiento Facial con TensorFlow
"""

__version__ = "1.0.0"
__author__ = "Cristian"
'@
$srcInitContent | Out-File -FilePath "src/__init__.py" -Encoding UTF8 -NoNewline

# src/config.py
$configContent = @'
"""
Configuracion general del proyecto
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Configuracion de datos
IMG_SIZE = 160
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Configuracion de entrenamiento
EPOCHS = 50
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 10

# Configuracion de modelo
MODEL_NAME = 'facenet'
USE_PRETRAINED = True

# Configuracion de captura
FOTOS_POR_VARIACION = {
    'frontal': 30,
    'lejos': 20,
    'cerca': 20,
    'perfil_izq': 15,
    'perfil_der': 15,
    'luz_baja': 15,
    'luz_alta': 15
}

# Configuracion de camara
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Configuracion de reconocimiento
CONFIDENCE_THRESHOLD = 0.6
RECOGNITION_THRESHOLD = 0.5

# TensorBoard
TENSORBOARD_LOG_DIR = LOGS_DIR / 'tensorboard'

# Variables de entorno
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
'@
$configContent | Out-File -FilePath "src/config.py" -Encoding UTF8 -NoNewline

# Crear archivos __init__.py vacíos
"" | Out-File -FilePath "src/data/__init__.py" -Encoding UTF8 -NoNewline
"" | Out-File -FilePath "src/models/__init__.py" -Encoding UTF8 -NoNewline
"" | Out-File -FilePath "src/training/__init__.py" -Encoding UTF8 -NoNewline
"" | Out-File -FilePath "src/inference/__init__.py" -Encoding UTF8 -NoNewline
"" | Out-File -FilePath "src/utils/__init__.py" -Encoding UTF8 -NoNewline

# src/data/capture.py
$captureContent = @'
"""
Sistema de captura inteligente de rostros
"""
import cv2
import os
import time
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import FOTOS_POR_VARIACION, CAMERA_INDEX, DATA_DIR

class CapturadorInteligente:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.variaciones = FOTOS_POR_VARIACION
        
    def capturar_dataset(self, nombre_persona):
        """Captura dataset completo con variaciones"""
        print(f"Iniciando captura para: {nombre_persona}")
        # TODO: Implementar logica de captura
        pass

if __name__ == "__main__":
    capturador = CapturadorInteligente()
    nombre = input("Nombre de la persona: ")
    capturador.capturar_dataset(nombre)
'@
$captureContent | Out-File -FilePath "src/data/capture.py" -Encoding UTF8 -NoNewline

# src/data/preprocessing.py
$preprocessingContent = @'
"""
Preprocesamiento de imagenes
"""
import cv2
import numpy as np
from pathlib import Path

def detectar_rostro(imagen):
    """Detecta y extrae el rostro de una imagen"""
    # TODO: Implementar deteccion
    pass

def normalizar_imagen(imagen, target_size=(160, 160)):
    """Normaliza imagen al tamano objetivo"""
    # TODO: Implementar normalizacion
    pass

def aplicar_augmentation(imagen):
    """Aplica data augmentation"""
    # TODO: Implementar augmentation
    pass
'@
$preprocessingContent | Out-File -FilePath "src/data/preprocessing.py" -Encoding UTF8 -NoNewline

# src/data/augmentation.py
$augmentationContent = @'
"""
Data Augmentation para rostros
"""
import tensorflow as tf
from tensorflow.keras import layers

def crear_augmentation_pipeline():
    """Crea pipeline de augmentation"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2)
    ])
'@
$augmentationContent | Out-File -FilePath "src/data/augmentation.py" -Encoding UTF8 -NoNewline

# src/models/facenet.py
$facenetContent = @'
"""
Implementacion de FaceNet
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionResNetV2

def crear_facenet(input_shape=(160, 160, 3), embedding_size=128):
    """Crea modelo FaceNet basado en InceptionResNetV2"""
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    embeddings = layers.Dense(embedding_size)(x)
    
    model = Model(inputs, embeddings, name='facenet')
    
    return model
'@
$facenetContent | Out-File -FilePath "src/models/facenet.py" -Encoding UTF8 -NoNewline

# src/models/arcface.py
$arcfaceContent = @'
"""
Implementacion de ArcFace
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def crear_arcface(input_shape=(160, 160, 3), num_classes=10):
    """Crea modelo ArcFace"""
    # TODO: Implementar ArcFace
    pass
'@
$arcfaceContent | Out-File -FilePath "src/models/arcface.py" -Encoding UTF8 -NoNewline

# src/models/custom_model.py
$customModelContent = @'
"""
Modelo personalizado
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def crear_modelo_custom(input_shape=(160, 160, 3), num_classes=10):
    """Crea modelo personalizado"""
    inputs = layers.Input(shape=input_shape)
    
    # TODO: Implementar arquitectura personalizada
    
    outputs = layers.Dense(num_classes, activation='softmax')(inputs)
    return Model(inputs, outputs, name='custom_model')
'@
$customModelContent | Out-File -FilePath "src/models/custom_model.py" -Encoding UTF8 -NoNewline

# src/training/train.py
$trainContent = @'
"""
Script principal de entrenamiento
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import *
from src.models.facenet import crear_facenet
import tensorflow as tf

def entrenar():
    """Funcion principal de entrenamiento"""
    print("Iniciando entrenamiento...")
    
    # TODO: Implementar entrenamiento completo
    
    print("Entrenamiento completado")

if __name__ == "__main__":
    entrenar()
'@
$trainContent | Out-File -FilePath "src/training/train.py" -Encoding UTF8 -NoNewline

# src/training/callbacks.py
$callbacksContent = @'
"""
Callbacks personalizados para entrenamiento
"""
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    """Callback personalizado"""
    
    def on_epoch_end(self, epoch, logs=None):
        # TODO: Implementar logica
        pass
'@
$callbacksContent | Out-File -FilePath "src/training/callbacks.py" -Encoding UTF8 -NoNewline

# src/training/metrics.py
$metricsContent = @'
"""
Metricas personalizadas
"""
import tensorflow as tf

def calcular_accuracy(y_true, y_pred):
    """Calcula accuracy"""
    # TODO: Implementar metrica
    pass
'@
$metricsContent | Out-File -FilePath "src/training/metrics.py" -Encoding UTF8 -NoNewline

# src/inference/recognize.py
$recognizeContent = @'
"""
Reconocimiento facial en tiempo real
"""
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CAMERA_INDEX

def reconocer_tiempo_real():
    """Reconoce rostros en tiempo real"""
    print("Iniciando reconocimiento en tiempo real...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # TODO: Implementar reconocimiento
        
        cv2.imshow('CoreFace-AI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocer_tiempo_real()
'@
$recognizeContent | Out-File -FilePath "src/inference/recognize.py" -Encoding UTF8 -NoNewline

# src/inference/predictor.py
$predictorContent = @'
"""
Predictor para inferencia
"""
import tensorflow as tf
import numpy as np

class FacePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
    def predict(self, imagen):
        """Realiza prediccion"""
        # TODO: Implementar prediccion
        pass
'@
$predictorContent | Out-File -FilePath "src/inference/predictor.py" -Encoding UTF8 -NoNewline

# src/utils/face_detector.py
$faceDetectorContent = @'
"""
Detector de rostros
"""
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detectar(self, imagen):
        """Detecta rostros en la imagen"""
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
'@
$faceDetectorContent | Out-File -FilePath "src/utils/face_detector.py" -Encoding UTF8 -NoNewline

# src/utils/visualization.py
$visualizationContent = @'
"""
Utilidades de visualizacion
"""
import matplotlib.pyplot as plt
import cv2

def mostrar_imagen(imagen, titulo="Imagen"):
    """Muestra una imagen"""
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def plot_history(history):
    """Grafica historial de entrenamiento"""
    # TODO: Implementar visualizacion
    pass
'@
$visualizationContent | Out-File -FilePath "src/utils/visualization.py" -Encoding UTF8 -NoNewline

# src/utils/logger.py
$loggerContent = @'
"""
Sistema de logging
"""
import logging
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    """Configura logger"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
'@
$loggerContent | Out-File -FilePath "src/utils/logger.py" -Encoding UTF8 -NoNewline

# tests/test_capture.py
$testCaptureContent = @'
"""
Tests para captura de rostros
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.capture import CapturadorInteligente

def test_capturador():
    capturador = CapturadorInteligente()
    assert capturador is not None
'@
$testCaptureContent | Out-File -FilePath "tests/test_capture.py" -Encoding UTF8 -NoNewline

# tests/test_model.py
$testModelContent = @'
"""
Tests para modelos
"""
import pytest

def test_model_creation():
    # TODO: Implementar tests
    pass
'@
$testModelContent | Out-File -FilePath "tests/test_model.py" -Encoding UTF8 -NoNewline

# tests/test_inference.py
$testInferenceContent = @'
"""
Tests para inferencia
"""
import pytest

def test_prediccion():
    # TODO: Implementar tests
    pass
'@
$testInferenceContent | Out-File -FilePath "tests/test_inference.py" -Encoding UTF8 -NoNewline

# .env
$envContent = @'
# Variables de entorno
DEBUG=False
USE_GPU=True
CAMERA_INDEX=0
MODEL_NAME=facenet
'@
$envContent | Out-File -FilePath ".env" -Encoding UTF8 -NoNewline

# .gitignore
$gitignoreContent = @'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/augmented/*
!data/augmented/.gitkeep

# Models
models/saved_models/*
!models/saved_models/.gitkeep
models/checkpoints/*
!models/checkpoints/.gitkeep
models/pretrained/*
!models/pretrained/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Outputs
outputs/*
!outputs/.gitkeep

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
'@
$gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8 -NoNewline

# README.md
$readmeContent = @'
# CoreFace-AI

Sistema inteligente de reconocimiento facial usando TensorFlow y Deep Learning.

## Caracteristicas

- Captura inteligente con multiples variaciones (distancia, angulo, iluminacion)
- Entrenamiento con TensorFlow/Keras
- Arquitecturas: FaceNet, ArcFace
- Reconocimiento en tiempo real
- Dockerizado (CPU y GPU)
- TensorBoard para monitoreo
- API REST (opcional)

## Instalacion

### Opcion 1: Local
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/CoreFace-AI.git
cd CoreFace-AI

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opcion 2: Docker
```bash
# CPU
docker-compose up coreface-cpu

# GPU
docker-compose up coreface-gpu
```

## Uso Rapido

### 1. Capturar rostros
```bash
python src/data/capture.py
```

### 2. Entrenar modelo
```bash
python src/training/train.py
```

### 3. Reconocimiento en tiempo real
```bash
python src/inference/recognize.py
```

## Estructura del Proyecto
```
CoreFace-AI/
├── docker/          # Dockerfiles y docker-compose
├── data/            # Datasets
├── models/          # Modelos entrenados
├── src/             # Codigo fuente
├── notebooks/       # Jupyter notebooks
├── tests/           # Tests unitarios
└── scripts/         # Scripts de utilidad
```

## Tecnologias

- **Deep Learning**: TensorFlow 2.14, Keras
- **Computer Vision**: OpenCV, MTCNN
- **Containerizacion**: Docker, Docker Compose
- **Visualizacion**: TensorBoard, Matplotlib
- **Testing**: Pytest

## Configuracion

Edita `src/config.py` para personalizar parametros.

## Contribuciones

Las contribuciones son bienvenidas!

## Licencia

MIT License

## Autor

**Cristian** - Universidad Politecnica Estatal del Carchi
'@
$readmeContent | Out-File -FilePath "README.md" -Encoding UTF8 -NoNewline

# Crear archivos .gitkeep
$gitkeepDirs = @(
    "data/raw",
    "data/processed/train",
    "data/processed/validation",
    "data/processed/test",
    "data/augmented",
    "models/saved_models",
    "models/checkpoints",
    "models/pretrained",
    "models/metadata",
    "logs",
    "outputs/predictions",
    "outputs/visualizations"
)

foreach ($dir in $gitkeepDirs) {
    "" | Out-File -FilePath "$dir/.gitkeep" -Encoding UTF8 -NoNewline
}

Write-Host ""
Write-Host "Estructura del proyecto creada exitosamente!" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Yellow
Write-Host "  1. python -m venv venv" -ForegroundColor Cyan
Write-Host "  2. venv\Scripts\activate" -ForegroundColor Cyan
Write-Host "  3. pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para subir a GitHub:" -ForegroundColor Yellow
Write-Host "  git add ." -ForegroundColor Cyan
Write-Host "  git commit -m 'Initial commit: CoreFace-AI structure'" -ForegroundColor Cyan
Write-Host "  git branch -M main" -ForegroundColor Cyan
Write-Host "  git remote add origin https://github.com/tu-usuario/CoreFace-AI.git" -ForegroundColor Cyan
Write-Host "  git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "Listo para empezar a desarrollar!" -ForegroundColor Green