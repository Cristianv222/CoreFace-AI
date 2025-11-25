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