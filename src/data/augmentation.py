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