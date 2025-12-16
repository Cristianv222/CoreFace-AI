
import cv2
import os
import time
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CAMERA_INDEX, DATA_DIR


class QualityChecker:
    """Verifica la calidad de las imágenes"""
    
    @staticmethod
    def check_blur(image, threshold=100):
        """Detecta si está borrosa"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > threshold, laplacian_var
    
    @staticmethod
    def check_brightness(image):
        """Verifica iluminación"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        avg_brightness = np.mean(gray)
        return 70 <= avg_brightness <= 180, avg_brightness
    
    @staticmethod
    def check_face_size(w, h, frame_w, frame_h):
        """Verifica tamaño del rostro"""
        face_area = w * h
        frame_area = frame_w * frame_h
        ratio = face_area / frame_area
        return 0.08 <= ratio <= 0.65, ratio * 100


class CapturadorInteligenteV2:
    """Sistema de captura con guía paso a paso"""
    
    def __init__(self):
        # Detector de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Verificador de calidad
        self.quality = QualityChecker()
        
        # Definir variaciones de captura
        self.variations = {
            'frontal': {
                'name': 'FRONTAL NORMAL',
                'icon': '😐',
                'count': 15,
                'instructions': [
                    'Mira directo a la cámara',
                    'Expresión neutral',
                    'Sin gafas ni accesorios'
                ],
                'color': (0, 255, 0)  # Verde
            },
            'sonrisa': {
                'name': 'SONRIENDO',
                'icon': '😊',
                'count': 10,
                'instructions': [
                    'Sonríe a la cámara',
                    'Muestra los dientes',
                    'Expresión feliz'
                ],
                'color': (0, 255, 255)  # Amarillo
            },
            'perfil_izq': {
                'name': 'PERFIL IZQUIERDO',
                'icon': '👈',
                'count': 10,
                'instructions': [
                    'Gira tu cabeza a la IZQUIERDA',
                    'Muestra el lado izquierdo',
                    'Aproximadamente 45 grados'
                ],
                'color': (255, 128, 0)  # Naranja
            },
            'perfil_der': {
                'name': 'PERFIL DERECHO',
                'icon': '👉',
                'count': 10,
                'instructions': [
                    'Gira tu cabeza a la DERECHA',
                    'Muestra el lado derecho',
                    'Aproximadamente 45 grados'
                ],
                'color': (255, 128, 0)  # Naranja
            },
            'arriba': {
                'name': 'MIRANDO ARRIBA',
                'icon': '👆',
                'count': 8,
                'instructions': [
                    'Levanta tu cabeza',
                    'Mira hacia arriba',
                    'Mantén el rostro visible'
                ],
                'color': (255, 0, 255)  # Magenta
            },
            'abajo': {
                'name': 'MIRANDO ABAJO',
                'icon': '👇',
                'count': 8,
                'instructions': [
                    'Baja tu cabeza',
                    'Mira hacia abajo',
                    'Mantén el rostro visible'
                ],
                'color': (255, 0, 255)  # Magenta
            },
            'con_gafas': {
                'name': 'CON GAFAS',
                'icon': '🤓',
                'count': 12,
                'instructions': [
                    'Ponte gafas o lentes',
                    'Mira a la cámara',
                    'Si no tienes, presiona ESPACIO para saltar'
                ],
                'color': (128, 0, 255),  # Púrpura
                'optional': True
            },
            'tapado': {
                'name': 'CARA TAPADA (50%)',
                'icon': '😷',
                'count': 12,
                'instructions': [
                    'Tápate la boca y nariz',
                    'Usa tu mano o tapabocas',
                    'Deja los ojos visibles'
                ],
                'color': (0, 128, 255)  # Azul claro
            },
            'diferentes': {
                'name': 'VARIACIONES LIBRES',
                'icon': '🎭',
                'count': 15,
                'instructions': [
                    'Haz diferentes expresiones',
                    'Muévete naturalmente',
                    'Diferentes ángulos'
                ],
                'color': (180, 180, 180)  # Gris
            }
        }
        
        # Estado de captura
        self.current_variation = None
        self.variation_index = 0
        self.photos_in_variation = 0
        self.total_photos = 0
        self.quality_rejects = 0
        
        # Control de tiempo
        self.last_capture_time = 0
        self.capture_interval = 0.4  # Segundos entre capturas
        
        # Preparación antes de empezar variación
        self.preparation_time = 3  # Segundos de preparación
        self.preparation_start = None
        self.is_preparing = True
    
    def count_existing_photos(self, person_dir):
        """Cuenta fotos existentes de una persona"""
        if not person_dir.exists():
            return 0, {}
        
        counts = {}
        total = 0
        
        for var_name in self.variations.keys():
            var_photos = list(person_dir.glob(f"*_{var_name}_*.jpg"))
            counts[var_name] = len(var_photos)
            total += len(var_photos)
        
        return total, counts
    
    def draw_ui(self, frame, face_detected, quality_status):
        """Dibuja la interfaz de usuario"""
        h, w = frame.shape[:2]
        
        # Obtener variación actual
        var = self.get_current_variation()
        if var is None:
            return frame
        
        var_info = self.variations[var]
        
        # Fondo semitransparente arriba
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Si está en preparación
        if self.is_preparing and self.preparation_start is not None:
            elapsed = time.time() - self.preparation_start
            remaining = max(0, self.preparation_time - elapsed)
            
            if remaining > 0:
                # Mostrar cuenta regresiva GRANDE
                countdown = int(remaining) + 1
                text = f"PREPARATE: {countdown}"
                font_scale = 3.0
                thickness = 8
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_BOLD, font_scale, thickness
                )
                
                x = (w - text_w) // 2
                y = h // 2
                
                # Sombra
                cv2.putText(frame, text, (x+5, y+5),
                           cv2.FONT_HERSHEY_BOLD, font_scale, (0, 0, 0), thickness+2)
                # Texto
                cv2.putText(frame, text, (x, y),
                           cv2.FONT_HERSHEY_BOLD, font_scale, (0, 255, 255), thickness)
                
                return frame
        
        # Título de variación + icono
        title = f"{var_info['icon']} {var_info['name']}"
        cv2.putText(frame, title, (20, 50),
                   cv2.FONT_HERSHEY_BOLD, 1.5, var_info['color'], 3)
        
        # Progreso de variación
        progress_text = f"{self.photos_in_variation}/{var_info['count']}"
        cv2.putText(frame, progress_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Barra de progreso
        bar_x, bar_y = 20, 110
        bar_w, bar_h = 300, 25
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (50, 50, 50), -1)
        
        # Progreso
        progress = self.photos_in_variation / var_info['count']
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                     var_info['color'], -1)
        
        # Borde
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (255, 255, 255), 2)
        
        # Instrucciones (GRANDES)
        y_offset = 160
        for i, instruction in enumerate(var_info['instructions']):
            cv2.putText(frame, instruction, (20, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Indicador de calidad (esquina superior derecha)
        status_x = w - 300
        status_y = 30
        
        if not face_detected:
            status_text = "NO SE DETECTA ROSTRO"
            status_color = (0, 0, 255)  # Rojo
        elif not quality_status['all_good']:
            issues = []
            if not quality_status['sharp']:
                issues.append("BORROSO")
            if not quality_status['bright']:
                issues.append("MALA LUZ")
            if not quality_status['size_ok']:
                issues.append("MUY LEJOS/CERCA")
            
            status_text = " - ".join(issues)
            status_color = (0, 165, 255)  # Naranja
        else:
            status_text = "CALIDAD OK!"
            status_color = (0, 255, 0)  # Verde
        
        cv2.putText(frame, status_text, (status_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Estadísticas generales (abajo)
        stats_y = h - 80
        cv2.rectangle(frame, (0, stats_y - 10), (w, h), (0, 0, 0), -1)
        
        total_needed = sum(v['count'] for v in self.variations.values())
        stats_text = f"TOTAL: {self.total_photos}/{total_needed} | RECHAZADAS: {self.quality_rejects}"
        cv2.putText(frame, stats_text, (20, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Controles
        controls = "Q: Salir | ESPACIO: Saltar variacion"
        cv2.putText(frame, controls, (20, stats_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        return frame
    
    def get_current_variation(self):
        """Obtiene la variación actual"""
        var_names = list(self.variations.keys())
        if self.variation_index >= len(var_names):
            return None
        return var_names[self.variation_index]
    
    def next_variation(self):
        """Avanza a la siguiente variación"""
        self.variation_index += 1
        self.photos_in_variation = 0
        self.is_preparing = True
        self.preparation_start = time.time()
        
        var = self.get_current_variation()
        if var:
            print(f"\n{'='*60}")
            print(f"📸 Nueva variación: {self.variations[var]['name']}")
            print(f"   Fotos necesarias: {self.variations[var]['count']}")
            for instruction in self.variations[var]['instructions']:
                print(f"   • {instruction}")
            print(f"{'='*60}\n")
    
    def check_quality(self, face_img, face_w, face_h, frame_w, frame_h):
        """Verifica calidad de la imagen"""
        is_sharp, blur_score = self.quality.check_blur(face_img)
        is_bright, brightness = self.quality.check_brightness(face_img)
        size_ok, size_ratio = self.quality.check_face_size(face_w, face_h, frame_w, frame_h)
        
        return {
            'sharp': is_sharp,
            'bright': is_bright,
            'size_ok': size_ok,
            'all_good': is_sharp and is_bright and size_ok,
            'blur_score': blur_score,
            'brightness': brightness,
            'size_ratio': size_ratio
        }
    
    def capture_dataset(self, person_name):
        """Captura dataset completo con guía paso a paso"""
        print(f"\n{'🎯'*30}")
        print(f"   CAPTURA INTELIGENTE - {person_name.upper()}")
        print(f"{'🎯'*30}\n")
        
        # Crear/verificar directorio
        person_dir = DATA_DIR / 'raw' / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Contar fotos existentes
        existing_photos, existing_counts = self.count_existing_photos(person_dir)
        
        if existing_photos > 0:
            print(f"📂 Se encontraron {existing_photos} fotos existentes:")
            for var_name, count in existing_counts.items():
                if count > 0:
                    var_info = self.variations[var_name]
                    print(f"   {var_info['icon']} {var_info['name']}: {count}/{var_info['count']}")
            
            print(f"\n💡 Las nuevas fotos se agregarán a las existentes")
            continuar = input("\n¿Continuar capturando? (s/n): ").strip().lower()
            if continuar != 's':
                print("❌ Captura cancelada")
                return
            print()
        
        # Abrir cámara
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("📋 INSTRUCCIONES GENERALES:")
        print("   • Mantén tu rostro visible en el recuadro")
        print("   • Sigue las instrucciones en pantalla")
        print("   • Las fotos se capturan automáticamente")
        print("   • Solo se guardan fotos de BUENA CALIDAD")
        print("   • Q = Salir | ESPACIO = Saltar variación\n")
        
        input("Presiona ENTER para comenzar...")
        print()
        
        # Iniciar primera variación
        self.next_variation()
        
        # Loop principal
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Error leyendo frame")
                break
            
            # Obtener variación actual
            current_var = self.get_current_variation()
            
            # Si terminamos todas las variaciones
            if current_var is None:
                print("\n🎉 ¡TODAS LAS VARIACIONES COMPLETADAS!")
                break
            
            var_info = self.variations[current_var]
            
            # Detectar rostros
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(80, 80)
            )
            
            face_detected = len(faces) > 0
            quality_status = {'all_good': False}
            
            # Procesar rostros
            for (x, y, w, h) in faces:
                # Verificar calidad
                face_img = gray[y:y+h, x:x+w]
                quality_status = self.check_quality(
                    face_img, w, h, frame.shape[1], frame.shape[0]
                )
                
                # Color del rectángulo según calidad
                if quality_status['all_good']:
                    rect_color = var_info['color']
                    thickness = 3
                else:
                    rect_color = (0, 165, 255)  # Naranja
                    thickness = 2
                
                # Dibujar rectángulo
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, thickness)
                
                # Si terminó la preparación y hay buena calidad
                if (not self.is_preparing and 
                    quality_status['all_good'] and 
                    time.time() - self.last_capture_time > self.capture_interval):
                    
                    # Capturar foto
                    if self.photos_in_variation < var_info['count']:
                        # Guardar rostro completo con margen
                        margin = int(0.2 * w)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        
                        face_with_margin = frame[y1:y2, x1:x2]
                        
                        # Nombre del archivo
                        timestamp = int(time.time() * 1000)
                        filename = f"{person_name}_{current_var}_{timestamp}.jpg"
                        filepath = person_dir / filename
                        
                        # Guardar
                        cv2.imwrite(str(filepath), face_with_margin)
                        
                        self.photos_in_variation += 1
                        self.total_photos += 1
                        self.last_capture_time = time.time()
                        
                        print(f"✅ [{self.total_photos}] Capturada: {filename}")
                        
                        # Si completó esta variación
                        if self.photos_in_variation >= var_info['count']:
                            print(f"🎉 ¡Variación '{var_info['name']}' completada!\n")
                            time.sleep(0.5)
                            self.next_variation()
                            break
                
                # Rechazar foto de mala calidad
                elif (not self.is_preparing and 
                      not quality_status['all_good'] and
                      time.time() - self.last_capture_time > self.capture_interval):
                    self.quality_rejects += 1
                    self.last_capture_time = time.time()
            
            # Verificar si terminó preparación
            if self.is_preparing and self.preparation_start is not None:
                elapsed = time.time() - self.preparation_start
                if elapsed >= self.preparation_time:
                    self.is_preparing = False
            
            # Dibujar UI
            frame = self.draw_ui(frame, face_detected, quality_status)
            
            # Mostrar
            cv2.imshow('CoreFace-AI - Captura Inteligente', frame)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⚠️ Captura cancelada por el usuario")
                break
            elif key == ord(' '):  # Espacio para saltar
                if var_info.get('optional', False):
                    print(f"⏭️  Saltando variación opcional: {var_info['name']}\n")
                    self.next_variation()
                else:
                    print("⚠️  Esta variación no es opcional")
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        # Resumen final
        print(f"\n{'='*60}")
        print("📊 RESUMEN DE CAPTURA")
        print(f"{'='*60}")
        print(f"   Persona: {person_name}")
        print(f"   Fotos capturadas en esta sesión: {self.total_photos}")
        print(f"   Fotos rechazadas (calidad): {self.quality_rejects}")
        
        # Contar total final
        final_total, final_counts = self.count_existing_photos(person_dir)
        print(f"   Total de fotos ahora: {final_total}")
        print(f"\n   Desglose por variación:")
        for var_name, count in final_counts.items():
            if count > 0:
                var_info = self.variations[var_name]
                status = "✅" if count >= var_info['count'] else "⚠️"
                print(f"   {status} {var_info['icon']} {var_info['name']}: {count}/{var_info['count']}")
        
        print(f"\n   📁 Ubicación: {person_dir}")
        print(f"{'='*60}\n")
        
        # Sugerencia
        if final_total >= 80:
            print("🎉 ¡Excelente! Tienes suficientes fotos para entrenar")
            print("   Puedes ejecutar: python entrenamiento_mejorado.py\n")
        elif final_total >= 50:
            print("✅ Tienes buena cantidad de fotos")
            print("   Para mejor precisión, considera capturar más\n")
        else:
            print("⚠️  Se recomienda capturar más fotos para mejor precisión\n")


def main():
    """Función principal"""
    print(f"\n{'🎥'*30}")
    print("   SISTEMA DE CAPTURA INTELIGENTE")
    print(f"{'🎥'*30}\n")
    
    # Obtener nombre
    nombre = input("👤 Nombre de la persona: ").strip()
    
    if not nombre:
        print("❌ Error: Debes ingresar un nombre válido")
        return
    
    # Verificar si ya existe
    person_dir = DATA_DIR / 'raw' / nombre
    if person_dir.exists():
        fotos_existentes = len(list(person_dir.glob("*.jpg")))
        if fotos_existentes > 0:
            print(f"\n💡 Ya existen {fotos_existentes} fotos de '{nombre}'")
            print("   Se continuará agregando más fotos\n")
    
    # Crear capturador
    capturador = CapturadorInteligenteV2()
    
    # Capturar
    capturador.capture_dataset(nombre)
    
    print("✅ Proceso finalizado\n")


if __name__ == "__main__":
    main()