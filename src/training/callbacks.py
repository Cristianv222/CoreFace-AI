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