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