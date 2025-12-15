

import cv2
import numpy as np
import sys
from pathlib import Path
import pickle
import json
from collections import deque, defaultdict
from datetime import datetime
import time
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))


class DeepFaceRecognizer:
    """
    Reconocedor facial usando deep learning con embeddings
    """
    
    def __init__(self, model_type='facenet', distance_metric='cosine'):
        """
        Inicializa el reconocedor con deep learning
        
        Args:
            model_type: 'facenet', 'arcface', o 'vggface2'
            distance_metric: 'cosine', 'euclidean', o 'euclidean_l2'
        """
        self.model_type = model_type
        self.distance_metric = distance_metric
        
        print(f"🔧 Inicializando {model_type.upper()}...")
        
        # Cargar modelos
        self._load_face_detector()
        self._load_landmark_detector()
        self._load_embedding_model()
        
        # Base de datos de embeddings
        self.known_embeddings = {}  # {name: [embedding1, embedding2, ...]}
        self.known_names = []
        
        # Umbrales de distancia según el modelo
        self.thresholds = {
            'facenet': {'cosine': 0.40, 'euclidean': 10.0, 'euclidean_l2': 0.8},
            'arcface': {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
            'vggface2': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86}
        }
        
        # Tracking
        self.tracked_faces = {}  # {track_id: {'bbox', 'embeddings', 'name', 'confidence'}}
        self.next_track_id = 0
        
        # Métricas
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Configuración
        self.embedding_buffer_size = 5  # Promediar últimos N embeddings
        self.unknown_threshold_multiplier = 1.2  # Margen para "desconocido"
        
    def _load_face_detector(self):
        """Carga detector de rostros DNN (más preciso)"""
        try:
            # Usar RetinaFace o MTCNN si está disponible
            # Por ahora usamos DNN de OpenCV
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            
            # Intentar cargar
            base_path = Path(__file__).parent.parent.parent / 'models' / 'detection'
            base_path.mkdir(parents=True, exist_ok=True)
            
            model_path = base_path / model_file
            config_path = base_path / config_file
            
            if model_path.exists() and config_path.exists():
                self.face_detector = cv2.dnn.readNetFromCaffe(
                    str(config_path), 
                    str(model_path)
                )
                print("✅ Detector DNN cargado")
            else:
                # Fallback a Haar
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("⚠️  Usando Haar Cascades (DNN no disponible)")
                
        except Exception as e:
            print(f"⚠️  Error cargando detector: {e}")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_landmark_detector(self):
        """Carga detector de landmarks faciales (68 puntos)"""
        try:
            # Intentar cargar dlib shape predictor
            predictor_path = Path(__file__).parent.parent.parent / 'models' / 'shape_predictor_68_face_landmarks.dat'
            
            if predictor_path.exists():
                import dlib
                self.landmark_detector = dlib.shape_predictor(str(predictor_path))
                self.face_detector_dlib = dlib.get_frontal_face_detector()
                self.use_landmarks = True
                print("✅ Detector de landmarks cargado (68 puntos)")
            else:
                self.use_landmarks = False
                print("⚠️  Landmarks no disponibles (dlib no configurado)")
                
        except ImportError:
            self.use_landmarks = False
            print("⚠️  dlib no instalado, landmarks deshabilitados")
    
    def _load_embedding_model(self):
        """Carga modelo de embeddings"""
        try:
            if self.model_type == 'facenet':
                self._load_facenet()
            elif self.model_type == 'arcface':
                self._load_arcface()
            elif self.model_type == 'vggface2':
                self._load_vggface()
            else:
                raise ValueError(f"Modelo desconocido: {self.model_type}")
                
        except Exception as e:
            print(f"❌ Error cargando modelo de embeddings: {e}")
            print("   Intentando con facenet-keras como fallback...")
            self._load_facenet_keras()
    
    def _load_facenet(self):
        """Carga FaceNet (Inception ResNet V1)"""
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.embedding_size = 512
            self.use_pytorch = True
            
            print(f"✅ FaceNet cargado (PyTorch - {self.device})")
            
        except ImportError:
            raise Exception("facenet-pytorch no instalado")
    
    def _load_facenet_keras(self):
        """Carga FaceNet usando Keras (fallback)"""
        try:
            from keras_facenet import FaceNet
            
            self.embedding_model = FaceNet()
            self.embedding_size = 128
            self.use_pytorch = False
            
            print("✅ FaceNet cargado (Keras)")
            
        except ImportError:
            raise Exception("keras-facenet no instalado")
    
    def _load_arcface(self):
        """Carga ArcFace"""
        # Implementación con InsightFace o modelo ONNX
        print("⚠️  ArcFace en desarrollo, usando FaceNet")
        self._load_facenet()
    
    def _load_vggface(self):
        """Carga VGGFace2"""
        # Implementación con DeepFace
        print("⚠️  VGGFace2 en desarrollo, usando FaceNet")
        self._load_facenet()
    
    def detect_faces(self, frame):
        """
        Detecta rostros en el frame
        
        Args:
            frame: Frame BGR
            
        Returns:
            Lista de (x, y, w, h)
        """
        if isinstance(self.face_detector, cv2.dnn_Net):
            return self._detect_faces_dnn(frame)
        else:
            return self._detect_faces_haar(frame)
    
    def _detect_faces_dnn(self, frame):
        """Detecta rostros con DNN"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.6:  # Umbral más alto para mejor calidad
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                x, y = max(0, startX), max(0, startY)
                w_face = min(w - x, endX - startX)
                h_face = min(h - y, endY - startY)
                
                if w_face > 30 and h_face > 30:  # Filtrar rostros muy pequeños
                    faces.append((x, y, w_face, h_face))
        
        return faces
    
    def _detect_faces_haar(self, frame):
        """Detecta rostros con Haar"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return faces
    
    def detect_landmarks(self, frame, x, y, w, h):
        """
        Detecta landmarks faciales (68 puntos)
        
        Args:
            frame: Frame BGR
            x, y, w, h: Bounding box del rostro
            
        Returns:
            Array de (68, 2) con coordenadas de landmarks o None
        """
        if not self.use_landmarks:
            return None
        
        try:
            import dlib
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect = dlib.rectangle(x, y, x+w, y+h)
            shape = self.landmark_detector(gray, rect)
            
            # Convertir a numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks
            
        except Exception as e:
            return None
    
    def align_face(self, frame, landmarks):
        """
        Alinea el rostro basándose en landmarks (ojos principalmente)
        
        Args:
            frame: Frame BGR
            landmarks: Array de landmarks (68, 2)
            
        Returns:
            Frame alineado
        """
        if landmarks is None or len(landmarks) < 68:
            return frame
        
        # Puntos de los ojos
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calcular ángulo
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calcular centro entre ojos
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)
        
        # Matriz de rotación
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Rotar frame
        (h, w) = frame.shape[:2]
        aligned = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def extract_embedding(self, frame, x, y, w, h):
        """
        Extrae embedding del rostro
        
        Args:
            frame: Frame BGR
            x, y, w, h: Bounding box del rostro
            
        Returns:
            Vector de embedding normalizado
        """
        # Extraer ROI con margen
        margin = 0.2
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(frame.shape[1], int(x + w * (1 + margin)))
        y2 = min(frame.shape[0], int(y + h * (1 + margin)))
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        # Detectar y alinear con landmarks si está disponible
        landmarks = self.detect_landmarks(frame, x, y, w, h)
        if landmarks is not None:
            face_roi = self.align_face(face_roi, landmarks - [x1, y1])
        
        try:
            if self.use_pytorch:
                return self._extract_embedding_pytorch(face_roi)
            else:
                return self._extract_embedding_keras(face_roi)
        except Exception as e:
            print(f"⚠️  Error extrayendo embedding: {e}")
            return None
    
    def _extract_embedding_pytorch(self, face_roi):
        """Extrae embedding con PyTorch (FaceNet)"""
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # Preprocesar
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
        
        # Extraer embedding
        with torch.no_grad():
            embedding = self.embedding_model(face_tensor).cpu().numpy().flatten()
        
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _extract_embedding_keras(self, face_roi):
        """Extrae embedding con Keras (FaceNet)"""
        # Preprocesar
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_array = np.expand_dims(face_resized, axis=0)
        
        # Extraer embedding
        embedding = self.embedding_model.embeddings(face_array)[0]
        
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def calculate_distance(self, embedding1, embedding2):
        """
        Calcula distancia entre dos embeddings
        
        Args:
            embedding1, embedding2: Vectores de embeddings
            
        Returns:
            Distancia según la métrica configurada
        """
        if self.distance_metric == 'cosine':
            # Distancia coseno (1 - similitud)
            similarity = np.dot(embedding1, embedding2)
            distance = 1 - similarity
            
        elif self.distance_metric == 'euclidean':
            # Distancia euclidiana
            distance = np.linalg.norm(embedding1 - embedding2)
            
        elif self.distance_metric == 'euclidean_l2':
            # Distancia euclidiana normalizada
            distance = np.linalg.norm(embedding1 - embedding2)
            distance = distance / np.sqrt(len(embedding1))
        
        return distance
    
    def recognize_face(self, embedding):
        """
        Reconoce un rostro comparando con la base de datos
        
        Args:
            embedding: Vector de embedding del rostro
            
        Returns:
            (name, confidence, distance)
        """
        if embedding is None or len(self.known_embeddings) == 0:
            return "Desconocido", 0.0, float('inf')
        
        # Obtener umbral del modelo actual
        threshold = self.thresholds[self.model_type][self.distance_metric]
        
        # Buscar el match más cercano
        min_distance = float('inf')
        best_match = None
        
        for name, embeddings_list in self.known_embeddings.items():
            for known_embedding in embeddings_list:
                distance = self.calculate_distance(embedding, known_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
        
        # Decidir si es conocido o desconocido
        if min_distance < threshold:
            # Convertir distancia a confianza (0-100%)
            confidence = max(0, (1 - min_distance / threshold) * 100)
            return best_match, confidence, min_distance
        else:
            return "Desconocido", 0.0, min_distance
    
    def load_database(self, db_path):
        """
        Carga base de datos de embeddings
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        db_path = Path(db_path)
        
        if not db_path.exists():
            print(f"⚠️  Base de datos no encontrada: {db_path}")
            return False
        
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.known_embeddings = data['embeddings']
        self.known_names = list(self.known_embeddings.keys())
        
        total_embeddings = sum(len(embs) for embs in self.known_embeddings.values())
        print(f"✅ Base de datos cargada:")
        print(f"   Personas: {len(self.known_names)}")
        print(f"   Embeddings totales: {total_embeddings}")
        print(f"   Nombres: {', '.join(self.known_names)}")
        
        return True
    
    def save_database(self, db_path):
        """
        Guarda base de datos de embeddings
        
        Args:
            db_path: Ruta donde guardar
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embeddings': self.known_embeddings,
            'model_type': self.model_type,
            'distance_metric': self.distance_metric,
            'embedding_size': self.embedding_size
        }
        
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Base de datos guardada en: {db_path}")
    
    def add_person(self, name, embedding):
        """
        Agrega una persona a la base de datos
        
        Args:
            name: Nombre de la persona
            embedding: Vector de embedding
        """
        if name not in self.known_embeddings:
            self.known_embeddings[name] = []
            self.known_names.append(name)
        
        self.known_embeddings[name].append(embedding)
    
    def match_tracked_face(self, x, y, w, h):
        """
        Asocia detección con rostro rastreado
        
        Args:
            x, y, w, h: Bounding box del rostro
            
        Returns:
            track_id
        """
        center = np.array([x + w/2, y + h/2])
        
        # Buscar rostro cercano
        min_distance = float('inf')
        matched_id = None
        
        for track_id, data in self.tracked_faces.items():
            tracked_bbox = data['bbox']
            tracked_center = np.array([
                tracked_bbox[0] + tracked_bbox[2]/2,
                tracked_bbox[1] + tracked_bbox[3]/2
            ])
            
            distance = np.linalg.norm(center - tracked_center)
            
            if distance < max(w, h) * 0.5 and distance < min_distance:
                min_distance = distance
                matched_id = track_id
        
        if matched_id is not None:
            # Actualizar bbox
            self.tracked_faces[matched_id]['bbox'] = (x, y, w, h)
            return matched_id
        else:
            # Nuevo rostro
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracked_faces[track_id] = {
                'bbox': (x, y, w, h),
                'embeddings': deque(maxlen=self.embedding_buffer_size),
                'name': None,
                'confidence': 0.0,
                'distance': float('inf')
            }
            
            return track_id
    
    def cleanup_old_tracks(self, active_tracks):
        """Limpia tracking de rostros que ya no están"""
        to_remove = [tid for tid in self.tracked_faces if tid not in active_tracks]
        for tid in to_remove:
            del self.tracked_faces[tid]
    
    def calculate_fps(self):
        """Calcula FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)


def train_database(data_dir, output_path, model_type='facenet'):
    """
    Entrena base de datos de embeddings desde un directorio de imágenes
    
    Estructura esperada:
    data_dir/
        persona1/
            foto1.jpg
            foto2.jpg
        persona2/
            foto1.jpg
    
    Args:
        data_dir: Directorio con carpetas de personas
        output_path: Donde guardar la base de datos
        model_type: Tipo de modelo a usar
    """
    print(f"\n{'='*60}")
    print("🎓 Entrenando base de datos de embeddings")
    print(f"{'='*60}\n")
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Directorio no encontrado: {data_dir}")
        return
    
    # Inicializar reconocedor
    recognizer = DeepFaceRecognizer(model_type=model_type)
    
    # Procesar cada persona
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if len(person_dirs) == 0:
        print("❌ No se encontraron carpetas de personas")
        return
    
    print(f"📁 Personas encontradas: {len(person_dirs)}\n")
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        print(f"👤 Procesando: {person_name}")
        
        # Obtener imágenes
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(person_dir.glob(ext))
        
        if len(image_files) == 0:
            print(f"   ⚠️  No se encontraron imágenes, saltando...")
            continue
        
        print(f"   📷 Imágenes: {len(image_files)}")
        
        embeddings_count = 0
        
        for img_path in image_files:
            # Cargar imagen
            frame = cv2.imread(str(img_path))
            
            if frame is None:
                continue
            
            # Detectar rostros
            faces = recognizer.detect_faces(frame)
            
            if len(faces) == 0:
                print(f"   ⚠️  No se detectó rostro en {img_path.name}")
                continue
            
            # Tomar el rostro más grande (asumimos que es la persona)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Extraer embedding
            embedding = recognizer.extract_embedding(frame, x, y, w, h)
            
            if embedding is not None:
                recognizer.add_person(person_name, embedding)
                embeddings_count += 1
        
        print(f"   ✅ Embeddings extraídos: {embeddings_count}\n")
    
    # Guardar base de datos
    if len(recognizer.known_embeddings) > 0:
        recognizer.save_database(output_path)
        print(f"\n{'='*60}")
        print("✅ Entrenamiento completado")
        print(f"{'='*60}\n")
    else:
        print("\n❌ No se pudo generar ningún embedding\n")


def reconocimiento_avanzado(db_path, camera_index=0, model_type='facenet'):
    """
    Ejecuta reconocimiento facial avanzado en tiempo real
    
    Args:
        db_path: Ruta a la base de datos de embeddings
        camera_index: Índice de la cámara
        model_type: Tipo de modelo ('facenet', 'arcface', 'vggface2')
    """
    print(f"\n{'='*60}")
    print("🎥 CoreFace-AI v2.0 - Reconocimiento Avanzado")
    print(f"{'='*60}\n")
    
    # Inicializar reconocedor
    recognizer = DeepFaceRecognizer(model_type=model_type)
    
    # Cargar base de datos
    if not recognizer.load_database(db_path):
        print("\n❌ No se pudo cargar la base de datos")
        print("   Primero entrena el modelo con: train_database()")
        return
    
    # Abrir cámara
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    # Configuración de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n📌 Controles:")
    print("   'q' - Salir")
    print("   's' - Screenshot")
    print("   'i' - Toggle info")
    print("   'd' - Toggle debug")
    print("   'r' - Reset tracking")
    print("   'l' - Toggle landmarks")
    print(f"{'='*60}\n")
    
    screenshot_count = 0
    show_info = True
    show_debug = False
    show_landmarks = True
    frame_count = 0
    
    # Estadísticas
    recognition_stats = defaultdict(int)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detectar rostros
            faces = recognizer.detect_faces(frame)
            
            # Tracking y reconocimiento
            active_tracks = set()
            
            for (x, y, w, h) in faces:
                # Tracking
                track_id = recognizer.match_tracked_face(x, y, w, h)
                active_tracks.add(track_id)
                
                track_data = recognizer.tracked_faces[track_id]
                
                # Extraer embedding (cada N frames)
                if frame_count % 3 == 0:
                    embedding = recognizer.extract_embedding(frame, x, y, w, h)
                    
                    if embedding is not None:
                        track_data['embeddings'].append(embedding)
                        
                        # Promediar embeddings recientes
                        if len(track_data['embeddings']) > 0:
                            avg_embedding = np.mean(track_data['embeddings'], axis=0)
                            
                            # Reconocer
                            name, confidence, distance = recognizer.recognize_face(avg_embedding)
                            
                            track_data['name'] = name
                            track_data['confidence'] = confidence
                            track_data['distance'] = distance
                            
                            # Estadísticas
                            recognition_stats[name] += 1
                
                # Obtener info del tracking
                name = track_data.get('name', 'Procesando...')
                confidence = track_data.get('confidence', 0.0)
                distance = track_data.get('distance', 0.0)
                
                # Determinar color
                if name == "Desconocido":
                    color = (0, 0, 255)  # Rojo
                    status = "❌"
                elif confidence > 70:
                    color = (0, 255, 0)  # Verde
                    status = "✅"
                elif confidence > 40:
                    color = (0, 165, 255)  # Naranja
                    status = "⚠️"
                else:
                    color = (255, 0, 0)  # Azul
                    status = "🔍"
                
                # Dibujar rectángulo
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Fondo del texto
                cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
                
                # Texto
                text = f"{status} {name}"
                cv2.putText(frame, text, (x+5, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                conf_text = f"{confidence:.1f}%"
                cv2.putText(frame, conf_text, (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug info
                if show_debug:
                    debug_text = f"D:{distance:.3f} T:{track_id}"
                    cv2.putText(frame, debug_text, (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Landmarks (si están disponibles)
                if show_landmarks and recognizer.use_landmarks:
                    landmarks = recognizer.detect_landmarks(frame, x, y, w, h)
                    if landmarks is not None:
                        for (lx, ly) in landmarks:
                            cv2.circle(frame, (lx, ly), 1, (0, 255, 255), -1)
            
            # Limpiar tracking
            recognizer.cleanup_old_tracks(active_tracks)
            
            # FPS
            fps = recognizer.calculate_fps()
            
            # Info en pantalla
            if show_info:
                info_y = 30
                
                # Línea 1: FPS y rostros
                cv2.putText(frame, f"FPS: {fps:.1f} | Rostros: {len(faces)}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 30
                
                # Línea 2: Modelo
                cv2.putText(frame, f"Modelo: {model_type.upper()}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                info_y += 25
                
                # Línea 3: Controles
                cv2.putText(frame, "q:Salir s:Screenshot i:Info d:Debug l:Landmarks",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mostrar frame
            cv2.imshow('CoreFace-AI v2.0 - Reconocimiento Avanzado', frame)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot: {filename}")
            elif key == ord('i'):
                show_info = not show_info
            elif key == ord('d'):
                show_debug = not show_debug
            elif key == ord('l'):
                show_landmarks = not show_landmarks
            elif key == ord('r'):
                recognizer.tracked_faces.clear()
                recognizer.next_track_id = 0
                print("🔄 Tracking reseteado")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
    
    finally:
        # Estadísticas finales
        print(f"\n{'='*60}")
        print("📊 Estadísticas de la sesión:")
        print(f"   Frames: {frame_count}")
        print(f"   FPS promedio: {np.mean(recognizer.fps_buffer):.1f}")
        print(f"\n   Reconocimientos:")
        for name, count in sorted(recognition_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {name}: {count}")
        print(f"{'='*60}\n")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CoreFace-AI v2.0 - Reconocimiento Avanzado')
    parser.add_argument('--mode', choices=['train', 'recognize'], required=True,
                       help='Modo: train (entrenar) o recognize (reconocer)')
    parser.add_argument('--data-dir', type=str, default='data/faces',
                       help='Directorio con imágenes para entrenar')
    parser.add_argument('--db-path', type=str, default='models/face_embeddings.pkl',
                       help='Ruta de la base de datos')
    parser.add_argument('--camera', type=int, default=0,
                       help='Índice de la cámara')
    parser.add_argument('--model', choices=['facenet', 'arcface', 'vggface2'],
                       default='facenet', help='Modelo de embeddings')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_database(args.data_dir, args.db_path, args.model)
    else:
        reconocimiento_avanzado(args.db_path, args.camera, args.model)