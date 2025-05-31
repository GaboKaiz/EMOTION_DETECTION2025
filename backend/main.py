from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import cv2
import base64
import numpy as np
from emotion_detection import EmotionDetector
import json
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar variables de entorno
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI)
    db = client["emotions_db"]
    collection = db["emotions"]
    logger.info("Conexión a MongoDB establecida")
except Exception as e:
    logger.error(f"Error al conectar a MongoDB: {e}")
    raise

# Inicializar detector de emociones
try:
    detector = EmotionDetector()
    logger.info("Detector de emociones inicializado")
except Exception as e:
    logger.error(f"Error al inicializar el detector: {e}")
    raise


# Función para convertir numpy.int32 a int
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


@app.websocket("/ws/emotions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Conexión WebSocket aceptada")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Frame recibido vía WebSocket")
            # Decodificar imagen desde base64
            try:
                img_data = base64.b64decode(data.split(",")[1])
                if not img_data:
                    logger.warning("Datos base64 vacíos")
                    continue
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Frame decodificado nulo")
                    continue
            except Exception as e:
                logger.error(f"Error al decodificar frame: {e}")
                continue

            # Detectar emociones
            results = detector.detect_emotions(frame)
            logger.info(f"Resultados de detección: {results}")

            # Convertir resultados para que sean serializables
            results = [
                {
                    "box": convert_to_serializable(result["box"]),
                    "emotion": result["emotion"],
                    "score": result["score"]
                }
                for result in results
            ]

            # Guardar en MongoDB
            if results:
                for result in results:
                    try:
                        collection.insert_one({
                            "emotion": result["emotion"],
                            "score": result["score"],
                            "box": result["box"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Emoción guardada: {result['emotion']}")
                    except Exception as e:
                        logger.error(f"Error al guardar en MongoDB: {e}")

            # Enviar resultados al frontend
            await websocket.send_json(results)
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        logger.info("Conexión WebSocket cerrada")
        await websocket.close()


@app.get("/emotions")
async def get_emotions():
    try:
        emotions = list(collection.find({}, {"_id": 0}))
        logger.info(f"Emociones recuperadas: {len(emotions)}")
        return emotions
    except Exception as e:
        logger.error(f"Error al obtener emociones: {e}")
        return []
