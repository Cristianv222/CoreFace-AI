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