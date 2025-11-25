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