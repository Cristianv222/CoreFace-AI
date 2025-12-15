
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import pickle
import base64
from pathlib import Path
import os
from datetime import datetime
import json
import sys

# Agregar path para importar módulos personalizados
sys.path.append(str(Path(__file__).parent.parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'coreface-ai-secret-key-2024'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent.parent / 'data' / 'raw'
app.config['MODELS_FOLDER'] = Path(__file__).parent.parent.parent / 'models'

# Crear directorios si no existen
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['MODELS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Variables globales
recognizer_lbph = None
recognizer_deep = None
label_dict = {}
modelo_lbph_cargado = False
modelo_deep_cargado = False
modelo_activo = 'lbph'  # 'lbph' o 'deep'

# Detector de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def cargar_modelo_lbph():
    """Carga el modelo LBPH"""
    global recognizer_lbph, label_dict, modelo_lbph_cargado
    
    try:
        model_path = app.config['MODELS_FOLDER'] / 'face_lbph.yml'
        labels_path = app.config['MODELS_FOLDER'] / 'labels.pkl'
        
        # Intentar con el nombre antiguo si no existe el nuevo
        if not model_path.exists():
            model_path = app.config['MODELS_FOLDER'] / 'face_trained.yml'
        
        if model_path.exists() and labels_path.exists():
            recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
            recognizer_lbph.read(str(model_path))
            
            with open(labels_path, 'rb') as f:
                label_dict = pickle.load(f)
            
            modelo_lbph_cargado = True
            print("✅ Modelo LBPH cargado correctamente")
            print(f"   Personas: {list(label_dict.values())}")
            return True
        else:
            print("⚠️  No se encontró el modelo LBPH entrenado")
            modelo_lbph_cargado = False
            return False
    except Exception as e:
        print(f"❌ Error cargando modelo LBPH: {e}")
        modelo_lbph_cargado = False
        return False


def cargar_modelo_deep():
    """Carga el modelo de Deep Learning"""
    global recognizer_deep, modelo_deep_cargado
    
    try:
        # Importar el sistema de deep learning
        from src.training.train_simple import DeepFaceRecognizer
        
        db_path = app.config['MODELS_FOLDER'] / 'face_embeddings.pkl'
        
        if db_path.exists():
            recognizer_deep = DeepFaceRecognizer(model_type='facenet')
            recognizer_deep.load_database(db_path)
            
            modelo_deep_cargado = True
            print("✅ Modelo Deep Learning cargado correctamente")
            print(f"   Personas: {recognizer_deep.known_names}")
            return True
        else:
            print("⚠️  No se encontró el modelo Deep Learning")
            modelo_deep_cargado = False
            return False
    except ImportError:
        print("⚠️  Módulo de Deep Learning no disponible")
        modelo_deep_cargado = False
        return False
    except Exception as e:
        print(f"❌ Error cargando modelo Deep Learning: {e}")
        modelo_deep_cargado = False
        return False


def intentar_cargar_modelos():
    """Intenta cargar todos los modelos disponibles"""
    lbph_ok = cargar_modelo_lbph()
    deep_ok = cargar_modelo_deep()
    
    # Establecer modelo activo según disponibilidad
    global modelo_activo
    if deep_ok:
        modelo_activo = 'deep'
    elif lbph_ok:
        modelo_activo = 'lbph'
    else:
        modelo_activo = None


# ============================================================
# RUTAS DE PÁGINAS
# ============================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', 
                         modelo_activo=modelo_activo,
                         lbph_cargado=modelo_lbph_cargado,
                         deep_cargado=modelo_deep_cargado)


@app.route('/capture')
def capture_page():
    """Página de captura de rostros"""
    return render_template('capture.html')


@app.route('/recognize')
def recognize_page():
    """Página de reconocimiento"""
    personas = []
    
    if modelo_deep_cargado and recognizer_deep:
        personas = recognizer_deep.known_names
    elif modelo_lbph_cargado and label_dict:
        personas = list(label_dict.values())
    
    # AGREGAR ESTA VARIABLE
    modelo_cargado = modelo_lbph_cargado or modelo_deep_cargado
    
    return render_template('recognize.html', 
                         personas=personas,
                         modelo_cargado=modelo_cargado,  # ← AGREGAR ESTO
                         modelo_activo=modelo_activo,
                         lbph_cargado=modelo_lbph_cargado,
                         deep_cargado=modelo_deep_cargado)


@app.route('/admin')
def admin_page():
    """Página de administración"""
    # Listar personas registradas
    raw_dir = app.config['UPLOAD_FOLDER']
    personas = []
    
    if raw_dir.exists():
        for persona_dir in raw_dir.iterdir():
            if persona_dir.is_dir() and not persona_dir.name.startswith('.'):
                num_fotos = len(list(persona_dir.glob('*.jpg'))) + \
                           len(list(persona_dir.glob('*.jpeg'))) + \
                           len(list(persona_dir.glob('*.png')))
                
                # Contar por variación
                variaciones = {}
                for foto in persona_dir.glob('*.jpg'):
                    # Extraer tipo de variación del nombre
                    parts = foto.stem.split('_')
                    if len(parts) >= 2:
                        var_type = parts[1] if len(parts) > 1 else 'general'
                        variaciones[var_type] = variaciones.get(var_type, 0) + 1
                
                personas.append({
                    'nombre': persona_dir.name,
                    'fotos': num_fotos,
                    'variaciones': variaciones
                })
    
    return render_template('admin.html', 
                         personas=personas,
                         modelo_activo=modelo_activo,
                         lbph_cargado=modelo_lbph_cargado,
                         deep_cargado=modelo_deep_cargado)


# ============================================================
# API - CAPTURA
# ============================================================

@app.route('/api/save_capture', methods=['POST'])
def save_capture():
    """Guarda imágenes capturadas desde el navegador"""
    try:
        data = request.json
        nombre = data.get('nombre')
        image_data = data.get('image')
        tipo = data.get('tipo', 'normal')
        
        if not nombre or not image_data:
            return jsonify({'success': False, 'error': 'Datos incompletos'})
        
        # Decodificar imagen base64
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar rostro
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No se detectó ningún rostro'})
        
        # Verificar calidad de imagen (importar verificador)
        from src.training.train_simple import ImageQualityChecker
        quality_checker = ImageQualityChecker()
        
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        is_sharp, blur_score = quality_checker.check_blur(face_roi)
        is_bright, brightness = quality_checker.check_brightness(face_roi)
        
        if not is_sharp:
            return jsonify({'success': False, 'error': 'Imagen borrosa, intenta de nuevo'})
        
        if not is_bright:
            return jsonify({'success': False, 'error': 'Mala iluminación, ajusta la luz'})
        
        # Crear directorio para la persona
        persona_dir = app.config['UPLOAD_FOLDER'] / nombre
        persona_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar imagen completa con margen
        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        face_with_margin = img[y1:y2, x1:x2]
        
        # Nombre del archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{nombre}_{tipo}_{timestamp}.jpg"
        filepath = persona_dir / filename
        
        cv2.imwrite(str(filepath), face_with_margin)
        
        # Contar fotos actuales
        total_fotos = len(list(persona_dir.glob('*.jpg')))
        
        # Contar por tipo
        tipo_fotos = len(list(persona_dir.glob(f'*_{tipo}_*.jpg')))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'total_fotos': total_fotos,
            'tipo_fotos': tipo_fotos,
            'quality': {
                'blur_score': float(blur_score),
                'brightness': float(brightness)
            }
        })
        
    except Exception as e:
        print(f"Error en save_capture: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# API - RECONOCIMIENTO
# ============================================================

@app.route('/api/recognize_frame', methods=['POST'])
def recognize_frame():
    """Reconoce rostro en un frame"""
    try:
        if not modelo_lbph_cargado and not modelo_deep_cargado:
            return jsonify({'success': False, 'error': 'No hay modelo cargado'})
        
        data = request.json
        image_data = data.get('image')
        usar_deep = data.get('use_deep', modelo_activo == 'deep')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No se recibió imagen'})
        
        # Decodificar imagen
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        resultados = []
        
        # Usar Deep Learning si está disponible y se solicita
        if usar_deep and modelo_deep_cargado and recognizer_deep:
            # Detectar rostros
            faces = recognizer_deep.detect_faces(img)
            
            for (x, y, w, h) in faces:
                # Extraer embedding
                embedding = recognizer_deep.extract_embedding(img, x, y, w, h)
                
                if embedding is not None:
                    # Reconocer
                    name, confidence, distance = recognizer_deep.recognize_face(embedding)
                    
                    # Determinar status
                    if name == "Desconocido":
                        status = "unknown"
                    elif confidence > 70:
                        status = "high"
                    elif confidence > 40:
                        status = "medium"
                    else:
                        status = "low"
                        name = "Desconocido"
                    
                    resultados.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'name': name,
                        'confidence': float(confidence),
                        'distance': float(distance),
                        'status': status,
                        'model': 'deep'
                    })
        
        # Usar LBPH si no se usó deep learning o como fallback
        elif modelo_lbph_cargado and recognizer_lbph:
            # Detectar rostros
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, confidence = recognizer_lbph.predict(face)
                name = label_dict.get(label, "Desconocido")
                
                # LBPH: menor confianza = mejor match (invertido)
                if confidence < 50:
                    status = "high"
                elif confidence < 80:
                    status = "medium"
                else:
                    status = "low"
                    name = "Desconocido"
                
                # Convertir confianza de LBPH a porcentaje (invertir)
                confidence_percent = max(0, 100 - confidence)
                
                resultados.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'name': name,
                    'confidence': float(confidence_percent),
                    'lbph_distance': float(confidence),
                    'status': status,
                    'model': 'lbph'
                })
        
        return jsonify({
            'success': True,
            'faces': resultados,
            'model_used': 'deep' if usar_deep and modelo_deep_cargado else 'lbph'
        })
        
    except Exception as e:
        print(f"Error en recognize_frame: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# API - ENTRENAMIENTO
# ============================================================

@app.route('/api/train', methods=['POST'])
def train_model():
    """Entrena los modelos con las imágenes capturadas"""
    try:
        data = request.json
        tipo_modelo = data.get('model_type', 'lbph')  # 'lbph', 'deep', o 'both'
        use_augmentation = data.get('augmentation', True)
        
        resultados = {}
        
        # Entrenar LBPH (siempre más rápido)
        if tipo_modelo in ['lbph', 'both']:
            try:
                from src.training.train_simple import entrenar_modelo_mejorado
                
                print("🧠 Entrenando modelo LBPH...")
                
                entrenar_modelo_mejorado(
                    data_dir=app.config['UPLOAD_FOLDER'].parent,
                    output_dir=app.config['MODELS_FOLDER'],
                    model_type='lbph',
                    use_augmentation=use_augmentation,
                    train_split=0.8,
                    evaluate_model=True
                )
                
                # Recargar modelo
                cargar_modelo_lbph()
                
                resultados['lbph'] = {
                    'success': True,
                    'message': 'Modelo LBPH entrenado correctamente'
                }
                
            except Exception as e:
                print(f"❌ Error entrenando LBPH: {e}")
                resultados['lbph'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Entrenar Deep Learning
        if tipo_modelo in ['deep', 'both']:
            try:
                from src.training.train_simple import train_database
                
                print("🧠 Entrenando modelo Deep Learning...")
                
                train_database(
                    data_dir=app.config['UPLOAD_FOLDER'],
                    output_path=app.config['MODELS_FOLDER'] / 'face_embeddings.pkl',
                    model_type='facenet'
                )
                
                # Recargar modelo
                cargar_modelo_deep()
                
                resultados['deep'] = {
                    'success': True,
                    'message': 'Modelo Deep Learning entrenado correctamente'
                }
                
            except Exception as e:
                print(f"❌ Error entrenando Deep Learning: {e}")
                resultados['deep'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Determinar éxito general
        success = any(r.get('success', False) for r in resultados.values())
        
        return jsonify({
            'success': success,
            'results': resultados,
            'modelo_activo': modelo_activo
        })
        
    except Exception as e:
        print(f"Error en train_model: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """Cambia entre modelo LBPH y Deep Learning"""
    global modelo_activo
    
    try:
        data = request.json
        nuevo_modelo = data.get('model', 'lbph')
        
        if nuevo_modelo == 'deep' and not modelo_deep_cargado:
            return jsonify({'success': False, 'error': 'Modelo Deep Learning no disponible'})
        
        if nuevo_modelo == 'lbph' and not modelo_lbph_cargado:
            return jsonify({'success': False, 'error': 'Modelo LBPH no disponible'})
        
        modelo_activo = nuevo_modelo
        
        return jsonify({
            'success': True,
            'modelo_activo': modelo_activo
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# API - ADMINISTRACIÓN
# ============================================================

@app.route('/api/delete_person', methods=['POST'])
def delete_person():
    """Elimina una persona del dataset"""
    try:
        data = request.json
        nombre = data.get('nombre')
        
        if not nombre:
            return jsonify({'success': False, 'error': 'Nombre no proporcionado'})
        
        persona_dir = app.config['UPLOAD_FOLDER'] / nombre
        
        if persona_dir.exists():
            import shutil
            shutil.rmtree(persona_dir)
            
            print(f"✅ Persona '{nombre}' eliminada")
            
            return jsonify({'success': True, 'message': f'Persona {nombre} eliminada'})
        else:
            return jsonify({'success': False, 'error': 'Persona no encontrada'})
            
    except Exception as e:
        print(f"Error en delete_person: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    """Obtiene estadísticas del sistema"""
    try:
        raw_dir = app.config['UPLOAD_FOLDER']
        
        total_personas = 0
        total_fotos = 0
        personas_detalle = []
        
        if raw_dir.exists():
            for persona_dir in raw_dir.iterdir():
                if persona_dir.is_dir() and not persona_dir.name.startswith('.'):
                    num_fotos = len(list(persona_dir.glob('*.jpg')))
                    total_personas += 1
                    total_fotos += num_fotos
                    
                    personas_detalle.append({
                        'nombre': persona_dir.name,
                        'fotos': num_fotos
                    })
        
        return jsonify({
            'success': True,
            'stats': {
                'total_personas': total_personas,
                'total_fotos': total_fotos,
                'modelo_activo': modelo_activo,
                'lbph_cargado': modelo_lbph_cargado,
                'deep_cargado': modelo_deep_cargado,
                'personas': personas_detalle
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# INICIALIZACIÓN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CoreFace-AI v2.0 - Servidor Web Mejorado")
    print("="*60)
    print("\n📱 Accede desde cualquier dispositivo:")
    print(f"   🏠 Local:      http://localhost:5000")
    print(f"   🌐 Red local:  http://TU_IP_LOCAL:5000")
    print(f"   🔗 ngrok:      ngrok http 5000")
    print("\n💡 Para encontrar tu IP:")
    print("   Windows: ipconfig")
    print("   Linux/Mac: ifconfig")
    print("\n🎨 Características:")
    print("   ✅ Captura inteligente con guía visual")
    print("   ✅ Entrenamiento con data augmentation")
    print("   ✅ Reconocimiento con Deep Learning")
    print("   ✅ Fallback a LBPH si es necesario")
    print("\n⚠️  Para detener: Ctrl+C")
    print("="*60 + "\n")
    
    # Intentar cargar modelos existentes
    print("🔍 Buscando modelos entrenados...")
    intentar_cargar_modelos()
    
    if modelo_activo:
        print(f"\n✅ Modelo activo: {modelo_activo.upper()}")
    else:
        print("\n⚠️  No hay modelos entrenados. Entrena primero desde /admin")
    
    print("\n" + "="*60 + "\n")
    
    # Iniciar servidor
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)