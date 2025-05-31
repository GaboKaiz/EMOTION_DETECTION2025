import cv2
import mediapipe as mp
from fer import FER
import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:

    def __init__(self):
        # Inicializar MediaPipe para detección de rostros
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.4)  # Reducir aún más
        
        # Inicializar FER para detección de emociones
        try:
            self.emotion_detector = FER(mtcnn=True)
        except Exception as e:
            logger.error(f"Error al inicializar FER: {e}")
            raise

        # Lista de emociones
        self.emotion_labels = ["happy", "angry", "surprised", "sad", "disgust", "fear", "neutral"]
        
        # Diccionario para conteo de emociones
        self.emotions_count = {emotion: 0 for emotion in self.emotion_labels}

        # Dimensiones del frame
        self.frame_width, self.frame_height = 480, 360  # Reducir aún más resolución

    def preprocess_roi(self, roi):
        """Preprocesa la ROI para mejorar la detección de FER."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Ecualizar histograma para mejorar contraste
        gray = cv2.equalizeHist(gray)
        # Suavizar para reducir ruido
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Convertir de vuelta a RGB
        roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return roi

    def detect_emotions(self, frame):
        """Detecta rostros y estima emociones en un frame."""
        try:
            # Redimensionar el frame
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            logger.debug(f"Frame redimensionado: {frame.shape}")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectar rostros
            results = self.face_detection.process(image)
            person_count = 0
            detection_results = []

            # Reiniciar el conteo de emociones
            self.emotions_count = {key: 0 for key in self.emotions_count}

            if results.detections:
                person_count = len(results.detections)
                logger.debug(f"Rostros detectados: {person_count}")

                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = (
                        int(bbox.xmin * self.frame_width),
                        int(bbox.ymin * self.frame_height),
                        int(bbox.width * self.frame_width),
                        int(bbox.height * self.frame_height)
                    )

                    # Aumentar el área de la región de interés (ROI)
                    margin = 60
                    face_roi = frame[
                        max(0, y - margin):min(self.frame_height, y + h + margin),
                        max(0, x - margin):min(self.frame_width, x + w + margin)
                    ]

                    if face_roi.size > 0:
                        # Asegurarse de que la ROI sea lo suficientemente grande
                        min_size = 48
                        if face_roi.shape[0] < min_size or face_roi.shape[1] < min_size:
                            logger.debug(f"ROI demasiado pequeño: {face_roi.shape}, omitiendo detección")
                            continue

                        # Preprocesar la ROI
                        face_roi = self.preprocess_roi(face_roi)
                        face_roi_resized = cv2.resize(face_roi, (224, 224))
                        logger.debug(f"ROI redimensionada para FER: {face_roi_resized.shape}")

                        # Detectar emociones con FER
                        emotion_results = self.emotion_detector.detect_emotions(face_roi_resized)

                        if emotion_results:
                            logger.debug(f"Emociones detectadas por FER: {emotion_results}")
                            max_emotion = max(emotion_results[0]["emotions"], key=emotion_results[0]["emotions"].get)
                            score = emotion_results[0]["emotions"][max_emotion]
                            if score > 0.3:  # Umbral mínimo para aceptar detección
                                if max_emotion in self.emotions_count:
                                    self.emotions_count[max_emotion] += 1

                                detection_results.append({
                                    "box": [x, y, w, h],
                                    "emotion": max_emotion,
                                    "score": float(score)
                                })
                        else:
                            logger.debug("No se detectaron emociones por FER en este rostro")
                    else:
                        logger.debug("ROI vacía, omitiendo detección")

            return {
                "detections": detection_results,
                "person_count": person_count,
                "emotions_count": self.emotions_count
            }
        except Exception as e:
            logger.error(f"Error al detectar emociones: {e}")
            return {
                "detections": [],
                "person_count": 0,
                "emotions_count": {emotion: 0 for emotion in self.emotion_labels}
            }
