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