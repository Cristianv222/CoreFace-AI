"""
Script de Entrenamiento Mejorado - CoreFace-AI
================================================

Mejoras implementadas:
1. Data Augmentation (rotación, brillo, contraste)
2. Validación cruzada (train/test split)
3. Métricas de rendimiento
4. Detección de calidad de imágenes
5. Balanceo de clases
6. Soporte para múltiples modelos (LBPH, EigenFaces, FisherFaces)
7. Visualización de resultados
8. Exportación de reportes
"""

import cv2
import os
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR


class DataAugmentation:
    """Clase para aumentar el dataset con transformaciones"""
    
    @staticmethod
    def rotate(image, angle):
        """Rota la imagen"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Ajusta brillo de la imagen"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(image, factor):
        """Ajusta contraste de la imagen"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    @staticmethod
    def flip_horizontal(image):
        """Voltea la imagen horizontalmente"""
        return cv2.flip(image, 1)
    
    @staticmethod
    def add_noise(image, intensity=10):
        """Agrega ruido gaussiano"""
        noise = np.random.normal(0, intensity, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    @staticmethod
    def augment_face(face_gray, num_augmentations=5):
        """
        Genera múltiples versiones augmentadas de un rostro
        
        Args:
            face_gray: Rostro en escala de grises
            num_augmentations: Número de augmentations a generar
            
        Returns:
            Lista de rostros augmentados
        """
        augmented_faces = [face_gray]  # Original
        
        # Convertir a BGR para algunas transformaciones
        face_bgr = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
        
        transformations = [
            lambda img: DataAugmentation.rotate(img, 5),
            lambda img: DataAugmentation.rotate(img, -5),
            lambda img: DataAugmentation.adjust_brightness(img, 1.2),
            lambda img: DataAugmentation.adjust_brightness(img, 0.8),
            lambda img: DataAugmentation.adjust_contrast(img, 1.2),
            lambda img: DataAugmentation.adjust_contrast(img, 0.8),
            lambda img: DataAugmentation.flip_horizontal(img),
            lambda img: DataAugmentation.add_noise(img, 5),
        ]
        
        # Seleccionar transformaciones aleatorias
        np.random.shuffle(transformations)
        
        for i, transform in enumerate(transformations[:num_augmentations]):
            try:
                augmented = transform(face_bgr)
                augmented_gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                augmented_faces.append(augmented_gray)
            except Exception as e:
                print(f"    ⚠️  Error en augmentation: {e}")
                continue
        
        return augmented_faces


class ImageQualityChecker:
    """Verifica la calidad de las imágenes"""
    
    @staticmethod
    def check_blur(image, threshold=100):
        """
        Detecta si la imagen está borrosa usando varianza de Laplacian
        
        Args:
            image: Imagen en escala de grises
            threshold: Umbral (valores bajos = borrosa)
            
        Returns:
            (es_nitida, score)
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var > threshold, laplacian_var
    
    @staticmethod
    def check_brightness(image):
        """
        Verifica si la imagen tiene buena iluminación
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            (es_adecuada, promedio_brillo)
        """
        avg_brightness = np.mean(image)
        # Rango óptimo: 70-180
        is_good = 70 <= avg_brightness <= 180
        return is_good, avg_brightness
    
    @staticmethod
    def check_size(face_region, min_size=60):
        """Verifica que el rostro sea suficientemente grande"""
        h, w = face_region.shape[:2]
        return min(h, w) >= min_size, (h, w)


class ModelTrainer:
    """Clase para entrenar modelos de reconocimiento facial"""
    
    def __init__(self, model_type='lbph', use_augmentation=True, train_split=0.8):
        """
        Inicializa el entrenador
        
        Args:
            model_type: Tipo de modelo ('lbph', 'eigenfaces', 'fisherfaces')
            use_augmentation: Usar data augmentation
            train_split: Proporción de datos para entrenamiento (0-1)
        """
        self.model_type = model_type
        self.use_augmentation = use_augmentation
        self.train_split = train_split
        
        # Inicializar modelo según tipo
        if model_type == 'lbph':
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8
            )
        elif model_type == 'eigenfaces':
            self.recognizer = cv2.face.EigenFaceRecognizer_create()
        elif model_type == 'fisherfaces':
            self.recognizer = cv2.face.FisherFaceRecognizer_create()
        else:
            raise ValueError(f"Modelo desconocido: {model_type}")
        
        # Detector de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Estadísticas
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'low_quality_rejected': 0,
            'augmented_faces': 0,
            'persons': {},
            'training_time': 0
        }
        
        self.quality_checker = ImageQualityChecker()
        self.augmentor = DataAugmentation()
    
    def extract_faces_from_image(self, img_path, check_quality=True):
        """
        Extrae rostros de una imagen con verificación de calidad
        
        Args:
            img_path: Ruta de la imagen
            check_quality: Verificar calidad de imagen
            
        Returns:
            Lista de tuplas (face_gray, quality_score)
        """
        try:
            img = cv2.imread(str(img_path))
            
            if img is None:
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Ecualización de histograma para mejorar contraste
            gray = cv2.equalizeHist(gray)
            
            # Detectar rostros
            detected_faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            valid_faces = []
            
            for (x, y, w, h) in detected_faces:
                face = gray[y:y+h, x:x+w]
                
                if check_quality:
                    # Verificar calidad
                    is_sharp, blur_score = self.quality_checker.check_blur(face)
                    is_bright, brightness = self.quality_checker.check_brightness(face)
                    is_large, size = self.quality_checker.check_size(face)
                    
                    quality_score = {
                        'blur': blur_score,
                        'brightness': brightness,
                        'size': size,
                        'is_valid': is_sharp and is_bright and is_large
                    }
                    
                    if not quality_score['is_valid']:
                        self.stats['low_quality_rejected'] += 1
                        continue
                else:
                    quality_score = {'is_valid': True}
                
                # Redimensionar a tamaño estándar
                face_resized = cv2.resize(face, (200, 200))
                
                # IMPORTANTE: Asegurar tipo uint8
                if face_resized.dtype != np.uint8:
                    face_resized = face_resized.astype(np.uint8)
                
                valid_faces.append((face_resized, quality_score))
            
            return valid_faces
            
        except Exception as e:
            print(f"    ⚠️  Error procesando imagen: {e}")
            return []
    
    def load_dataset(self, data_dir):
        """
        Carga el dataset completo
        
        Args:
            data_dir: Directorio con carpetas de personas
            
        Returns:
            (faces, labels, label_dict)
        """
        print(f"\n{'='*60}")
        print("📂 Cargando dataset...")
        print(f"{'='*60}\n")
        
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        raw_dir = Path(data_dir) / 'raw'
        
        if not raw_dir.exists():
            print(f"❌ Error: No existe {raw_dir}")
            return None, None, None
        
        # Procesar cada persona
        person_dirs = [d for d in os.listdir(raw_dir) 
                      if (raw_dir / d).is_dir() and not d.startswith('.')]
        
        if len(person_dirs) == 0:
            print("❌ No se encontraron carpetas de personas")
            return None, None, None
        
        for persona in person_dirs:
            persona_path = raw_dir / persona
            label_dict[current_label] = persona
            
            print(f"👤 {persona} (Label: {current_label})")
            
            person_faces = []
            person_stats = {
                'images': 0,
                'faces': 0,
                'augmented': 0,
                'rejected': 0
            }
            
            # Procesar imágenes
            image_files = [f for f in os.listdir(persona_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in image_files:
                img_path = persona_path / img_name
                self.stats['total_images'] += 1
                person_stats['images'] += 1
                
                # Extraer rostros
                extracted_faces = self.extract_faces_from_image(img_path)
                
                for face, quality in extracted_faces:
                    person_faces.append(face)
                    person_stats['faces'] += 1
                    
                    # Data augmentation
                    if self.use_augmentation:
                        augmented = self.augmentor.augment_face(
                            face, 
                            num_augmentations=3
                        )
                        person_faces.extend(augmented[1:])  # Excluir original
                        person_stats['augmented'] += len(augmented) - 1
                        self.stats['augmented_faces'] += len(augmented) - 1
            
            # Agregar todas las caras de esta persona
            for face in person_faces:
                faces.append(face)
                labels.append(current_label)
            
            self.stats['total_faces'] += len(person_faces)
            self.stats['persons'][persona] = person_stats
            
            print(f"    Imágenes: {person_stats['images']}")
            print(f"    Rostros válidos: {person_stats['faces']}")
            
            if self.use_augmentation:
                print(f"    Augmentados: {person_stats['augmented']}")
            
            print(f"    Total para entrenamiento: {len(person_faces)}\n")
            
            current_label += 1
        
        return faces, labels, label_dict
    
    def split_train_test(self, faces, labels):
        """
        Divide dataset en train/test
        
        Args:
            faces: Lista de rostros
            labels: Lista de etiquetas
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        # Convertir a numpy con tipos correctos
        X = np.array(faces, dtype=np.uint8)
        y = np.array(labels, dtype=np.int32)
        
        # Shuffle
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Split
        split_idx = int(len(X) * self.train_split)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, faces, labels):
        """
        Entrena el modelo
        
        Args:
            faces: Lista de rostros o numpy array
            labels: Lista de etiquetas o numpy array
        """
        print(f"{'='*60}")
        print(f" Entrenando modelo {self.model_type.upper()}...")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        # Convertir a numpy arrays con tipos correctos
        if not isinstance(faces, np.ndarray):
            faces_array = np.array(faces, dtype=np.uint8)
        else:
            faces_array = faces.astype(np.uint8)
        
        if not isinstance(labels, np.ndarray):
            labels_array = np.array(labels, dtype=np.int32)
        else:
            labels_array = labels.astype(np.int32)
        
        # Verificar dimensiones
        if len(faces_array.shape) != 3:
            print(f" Error: Shape incorrecto de faces: {faces_array.shape}")
            print(f"   Esperado: (n_samples, height, width)")
            return
        
        if len(labels_array.shape) != 1:
            print(f" Error: Shape incorrecto de labels: {labels_array.shape}")
            print(f"   Esperado: (n_samples,)")
            return
        
        # Debug: Mostrar información
        print(f" Datos de entrenamiento:")
        print(f"   Rostros: {len(faces_array)}")
        print(f"   Tamaño por rostro: {faces_array.shape[1:]} pixels")
        print(f"   Tipo de datos faces: {faces_array.dtype}")
        print(f"   Tipo de datos labels: {labels_array.dtype}")
        print(f"   Rango de labels: {labels_array.min()} - {labels_array.max()}")
        print()
        
        try:
            # Entrenar modelo
            self.recognizer.train(faces_array, labels_array)
            
            end_time = datetime.now()
            self.stats['training_time'] = (end_time - start_time).total_seconds()
            
            print(f"✅ Entrenamiento completado en {self.stats['training_time']:.2f}s\n")
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            print(f"\nDebug adicional:")
            print(f"   Primer rostro dtype: {faces_array[0].dtype}")
            print(f"   Primer rostro shape: {faces_array[0].shape}")
            print(f"   Primer rostro min/max: {faces_array[0].min()}/{faces_array[0].max()}")
            raise
    
    def evaluate(self, X_test, y_test, label_dict):
        """
        Evalúa el modelo con el conjunto de prueba
        
        Args:
            X_test: Rostros de prueba
            y_test: Etiquetas de prueba
            label_dict: Diccionario de etiquetas
            
        Returns:
            Diccionario con métricas
        """
        print(f"{'='*60}")
        print(" Evaluando modelo...")
        print(f"{'='*60}\n")
        
        correct = 0
        total = len(X_test)
        
        # Métricas por persona
        person_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for i, face in enumerate(X_test):
            true_label = y_test[i]
            
            try:
                pred_label, confidence = self.recognizer.predict(face)
                
                if pred_label == true_label:
                    correct += 1
                    person_metrics[label_dict[true_label]]['correct'] += 1
                
                person_metrics[label_dict[true_label]]['total'] += 1
                
            except Exception as e:
                print(f"  ⚠️  Error en predicción: {e}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f" Precisión Global: {accuracy:.2f}% ({correct}/{total})\n")
        
        print(" Precisión por Persona:")
        for person, metrics in person_metrics.items():
            person_acc = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
            print(f"   {person}: {person_acc:.2f}% ({metrics['correct']}/{metrics['total']})")
        
        print()
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'person_metrics': dict(person_metrics)
        }
    
    def save_model(self, output_dir):
        """
        Guarda el modelo entrenado
        
        Args:
            output_dir: Directorio de salida
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Guardar modelo
        model_path = output_dir / f'face_{self.model_type}.yml'
        self.recognizer.write(str(model_path))
        
        print(f"💾 Modelo guardado: {model_path}")
        
        return model_path
    
    def save_labels(self, label_dict, output_dir):
        """
        Guarda el diccionario de etiquetas
        
        Args:
            label_dict: Diccionario de etiquetas
            output_dir: Directorio de salida
        """
        output_dir = Path(output_dir)
        labels_path = output_dir / 'labels.pkl'
        
        with open(labels_path, 'wb') as f:
            pickle.dump(label_dict, f)
        
        print(f"💾 Etiquetas guardadas: {labels_path}")
        
        return labels_path
    
    def save_report(self, label_dict, metrics, output_dir):
        """
        Guarda reporte de entrenamiento
        
        Args:
            label_dict: Diccionario de etiquetas
            metrics: Métricas de evaluación
            output_dir: Directorio de salida
        """
        output_dir = Path(output_dir)
        report_path = output_dir / 'training_report.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'use_augmentation': self.use_augmentation,
            'train_split': self.train_split,
            'statistics': self.stats,
            'label_dict': label_dict,
            'metrics': metrics
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f" Reporte guardado: {report_path}")
        
        # También guardar versión legible
        txt_path = output_dir / 'training_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE ENTRENAMIENTO - CoreFace-AI\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo: {self.model_type.upper()}\n")
            f.write(f"Data Augmentation: {'Sí' if self.use_augmentation else 'No'}\n")
            f.write(f"Train/Test Split: {self.train_split*100:.0f}%/{(1-self.train_split)*100:.0f}%\n\n")
            
            f.write("="*60 + "\n")
            f.write("ESTADÍSTICAS DEL DATASET\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Imágenes totales: {self.stats['total_images']}\n")
            f.write(f"Rostros extraídos: {self.stats['total_faces']}\n")
            f.write(f"Rostros rechazados (calidad): {self.stats['low_quality_rejected']}\n")
            f.write(f"Rostros augmentados: {self.stats['augmented_faces']}\n")
            f.write(f"Tiempo de entrenamiento: {self.stats['training_time']:.2f}s\n\n")
            
            f.write("="*60 + "\n")
            f.write("ESTADÍSTICAS POR PERSONA\n")
            f.write("="*60 + "\n\n")
            
            for person, stats in self.stats['persons'].items():
                f.write(f"{person}:\n")
                f.write(f"  Imágenes: {stats['images']}\n")
                f.write(f"  Rostros válidos: {stats['faces']}\n")
                f.write(f"  Augmentados: {stats['augmented']}\n\n")
            
            if metrics:
                f.write("="*60 + "\n")
                f.write("MÉTRICAS DE EVALUACIÓN\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Precisión Global: {metrics['accuracy']:.2f}%\n")
                f.write(f"Correctos: {metrics['correct']}/{metrics['total']}\n\n")
                
                f.write("Precisión por Persona:\n")
                for person, person_metrics in metrics['person_metrics'].items():
                    person_acc = (person_metrics['correct'] / person_metrics['total'] * 100) if person_metrics['total'] > 0 else 0
                    f.write(f"  {person}: {person_acc:.2f}% ({person_metrics['correct']}/{person_metrics['total']})\n")
        
        print(f" Reporte TXT guardado: {txt_path}")


def entrenar_modelo_mejorado(
    data_dir=None,
    output_dir=None,
    model_type='lbph',
    use_augmentation=True,
    train_split=0.8,
    evaluate_model=True
):
    """
    Función principal de entrenamiento mejorado
    
    Args:
        data_dir: Directorio con datos de entrenamiento
        output_dir: Directorio para guardar modelo
        model_type: Tipo de modelo ('lbph', 'eigenfaces', 'fisherfaces')
        use_augmentation: Usar data augmentation
        train_split: Proporción train/test
        evaluate_model: Evaluar modelo con conjunto de prueba
    """
    print("\n" + ""*30)
    print("   CoreFace-AI - Entrenamiento Mejorado")
    print(""*30 + "\n")
    
    # Configurar directorios
    if data_dir is None:
        data_dir = DATA_DIR
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'models'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Inicializar entrenador
    trainer = ModelTrainer(
        model_type=model_type,
        use_augmentation=use_augmentation,
        train_split=train_split
    )
    
    # Cargar dataset
    faces, labels, label_dict = trainer.load_dataset(data_dir)
    
    if faces is None or len(faces) == 0:
        print(" No se pudo cargar el dataset")
        return
    
    # Split train/test
    if evaluate_model and train_split < 1.0:
        X_train, X_test, y_train, y_test = trainer.split_train_test(faces, labels)
        
        print(f"{'='*60}")
        print(f" Dataset dividido:")
        print(f"   Entrenamiento: {len(X_train)} rostros ({train_split*100:.0f}%)")
        print(f"   Prueba: {len(X_test)} rostros ({(1-train_split)*100:.0f}%)")
        print(f"{'='*60}\n")
        
        # Entrenar (pasar arrays directamente, ya están en formato correcto)
        trainer.train(X_train, y_train)
        
        # Evaluar
        metrics = trainer.evaluate(X_test, y_test, label_dict)
    else:
        print(f"{'='*60}")
        print(f" Usando todo el dataset para entrenamiento: {len(faces)} rostros")
        print(f"{'='*60}\n")
        
        # Convertir a arrays antes de entrenar
        faces_array = np.array(faces, dtype=np.uint8)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Entrenar con todo
        trainer.train(faces_array, labels_array)
        metrics = None
    
    # Guardar modelo
    print(f"\n{'='*60}")
    print(" Guardando modelo...")
    print(f"{'='*60}\n")
    
    model_path = trainer.save_model(output_dir)
    labels_path = trainer.save_labels(label_dict, output_dir)
    trainer.save_report(label_dict, metrics, output_dir)
    
    # Resumen final
    print(f"\n{'='*60}")
    print(" ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"\n Resumen:")
    print(f"   Modelo: {model_type.upper()}")
    print(f"   Personas: {len(label_dict)}")
    print(f"   Rostros totales: {trainer.stats['total_faces']}")
    print(f"   Augmentation: {'Sí' if use_augmentation else 'No'}")
    
    if metrics:
        print(f"   Precisión: {metrics['accuracy']:.2f}%")
    
    print(f"\n Archivos generados:")
    print(f"   {model_path}")
    print(f"   {labels_path}")
    print(f"   {output_dir / 'training_report.json'}")
    print(f"   {output_dir / 'training_report.txt'}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento Mejorado - CoreFace-AI')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directorio con datos de entrenamiento')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directorio para guardar modelo')
    parser.add_argument('--model', choices=['lbph', 'eigenfaces', 'fisherfaces'],
                       default='lbph', help='Tipo de modelo')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Desactivar data augmentation')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Proporción train/test (0-1)')
    parser.add_argument('--no-eval', action='store_true',
                       help='No evaluar modelo')
    
    args = parser.parse_args()
    
    entrenar_modelo_mejorado(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        use_augmentation=not args.no_augmentation,
        train_split=args.train_split,
        evaluate_model=not args.no_eval
    )