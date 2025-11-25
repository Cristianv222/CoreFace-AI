# CoreFace-AI

Sistema inteligente de reconocimiento facial usando TensorFlow y Deep Learning.

## Caracteristicas

- Captura inteligente con multiples variaciones (distancia, angulo, iluminacion)
- Entrenamiento con TensorFlow/Keras
- Arquitecturas: FaceNet, ArcFace
- Reconocimiento en tiempo real
- Dockerizado (CPU y GPU)
- TensorBoard para monitoreo
- API REST (opcional)

## Instalacion

### Opcion 1: Local
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/CoreFace-AI.git
cd CoreFace-AI

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opcion 2: Docker
```bash
# CPU
docker-compose up coreface-cpu

# GPU
docker-compose up coreface-gpu
```

## Uso Rapido

### 1. Capturar rostros
```bash
python src/data/capture.py
```

### 2. Entrenar modelo
```bash
python src/training/train.py
```

### 3. Reconocimiento en tiempo real
```bash
python src/inference/recognize.py
```

## Estructura del Proyecto
```
CoreFace-AI/
â”œâ”€â”€ docker/          # Dockerfiles y docker-compose
â”œâ”€â”€ data/            # Datasets
â”œâ”€â”€ models/          # Modelos entrenados
â”œâ”€â”€ src/             # Codigo fuente
â”œâ”€â”€ notebooks/       # Jupyter notebooks
â”œâ”€â”€ tests/           # Tests unitarios
â””â”€â”€ scripts/         # Scripts de utilidad
```

## Tecnologias

- **Deep Learning**: TensorFlow 2.14, Keras
- **Computer Vision**: OpenCV, MTCNN
- **Containerizacion**: Docker, Docker Compose
- **Visualizacion**: TensorBoard, Matplotlib
- **Testing**: Pytest

## Configuracion

Edita `src/config.py` para personalizar parametros.

## Contribuciones

Las contribuciones son bienvenidas!

## Licencia

MIT License

## Autor

**Cristian** - Universidad Politecnica Estatal del Carchi