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