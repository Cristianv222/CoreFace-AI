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