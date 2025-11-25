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