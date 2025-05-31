import cv2
import numpy as np
from collections import deque


class EmotionDetector:

    def __init__(self):
        # Cargar clasificadores Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Verificar que los clasificadores se carguen correctamente
        if self.face_cascade.empty():
            raise Exception("Error: No se pudo cargar el clasificador Haar Cascade para rostros")
        if self.smile_cascade.empty():
            raise Exception("Error: No se pudo cargar el clasificador de sonrisas")
        if self.eye_cascade.empty():
            raise Exception("Error: No se pudo cargar el clasificador de ojos")
        
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']
        self.emotion_history = deque(maxlen=5)
        self.weights = [0.1, 0.15, 0.2, 0.25, 0.3]

    def estimate_emotion(self, face_img, gray_face, x, y, w, h):
        """Estima la emoción basada en la detección de sonrisas, ojos, cejas y forma de la boca."""
        try:
            # Validar que sea un rostro verificando la presencia de al menos un ojo
            eyes = self.eye_cascade.detectMultiScale(gray_face[0:int(h * 0.6)], scaleFactor=1.1, minNeighbors=25, minSize=(25, 25))
            if len(eyes) < 1:
                return None, None

            # Detectar sonrisas en la región inferior del rostro
            smiles = self.smile_cascade.detectMultiScale(gray_face[int(h * 0.5):h], scaleFactor=1.6, minNeighbors=35, minSize=(30, 30))

            # Extraer región de la boca
            mouth_region = gray_face[int(h * 0.6):h, 0:w]
            if mouth_region.size == 0:
                return "neutral", 0.6

            # Aplicar umbral adaptativo para detectar bordes de la boca
            thresh = cv2.adaptiveThreshold(mouth_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mouth_shape = None
            mouth_symmetry = 1.0
            mouth_curvature = 0.0
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 150:
                    x_m, y_m, w_m, h_m = cv2.boundingRect(max_contour)
                    mouth_shape = h_m / w_m
                    # Calcular simetría
                    left_half = mouth_region[:,:w_m // 2]
                    right_half = mouth_region[:, w_m // 2:]
                    left_sum = np.sum(left_half)
                    right_sum = np.sum(right_half)
                    if left_sum > 0 and right_sum > 0:
                        mouth_symmetry = min(left_sum, right_sum) / max(left_sum, right_sum)
                    # Calcular curvatura con ajuste polinómico
                    contour_points = max_contour.reshape(-1, 2)
                    if len(contour_points) > 5:
                        x_coords, y_coords = contour_points[:, 0], contour_points[:, 1]
                        try:
                            coeffs = np.polyfit(x_coords, y_coords, 2)
                            mouth_curvature = abs(coeffs[0])
                        except np.linalg.LinAlgError:
                            mouth_curvature = 0.0

            # Estimar apertura de ojos y distancia entre ellos
            eye_openness = 0.0
            eye_distance = 0.0
            if len(eyes) >= 2:
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_openness += eh / ew
                eye_openness /= 2
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]
                eye_distance = abs((ex1 + ew1 / 2) - (ex2 + ew2 / 2)) / w
            elif len(eyes) == 1:
                ex, ey, ew, eh = eyes[0]
                eye_openness = eh / ew
                eye_distance = 0.5

            # Detectar cejas en la región superior
            eyebrow_region = gray_face[int(h * 0.15):int(h * 0.35), 0:w]
            eyebrow_angle = 0.0
            if eyebrow_region.size > 0:
                thresh_eyebrow = cv2.adaptiveThreshold(eyebrow_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                contours_eyebrow, _ = cv2.findContours(thresh_eyebrow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_eyebrow:
                    contours_eyebrow = sorted(contours_eyebrow, key=cv2.contourArea, reverse=True)[:2]
                    angles = []
                    for contour in contours_eyebrow:
                        if cv2.contourArea(contour) > 60:
                            rect = cv2.minAreaRect(contour)
                            angles.append(rect[2])
                    eyebrow_angle = np.mean(angles) if angles else 0.0

            # Lógica heurística refinada (idéntica al código de referencia)
            if len(smiles) > 0 and mouth_shape is not None and mouth_shape > 0.45 and mouth_symmetry > 0.9 and mouth_curvature < 0.2:
                return "happy", 0.9
            elif eye_openness > 0.65 and mouth_shape is not None and mouth_shape > 0.55 and eye_distance > 0.3:
                return "surprise", 0.85
            elif eye_openness > 0.65 and mouth_shape is not None and mouth_shape < 0.35 and mouth_symmetry > 0.85:
                return "fear", 0.8
            elif mouth_shape is not None:
                if mouth_shape < 0.35 and eye_openness < 0.3 and len(eyes) >= 2 and eyebrow_angle > 15:
                    return "angry", 0.8
                elif mouth_shape < 0.35 and eyebrow_angle < -15:
                    return "sad", 0.75
                elif mouth_shape > 0.75 and mouth_symmetry < 0.6 and eye_distance < 0.25:
                    return "disgust", 0.75
                elif mouth_shape > 0.75 and mouth_symmetry > 0.85:
                    return "fear", 0.7
            return "neutral", 0.7
        except Exception as e:
            print(f"Error al estimar emoción: {e}")
            return None, None

    def detect_emotions(self, frame):
        """Detecta rostros y estima emociones en un frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=12, minSize=(70, 70))
            results = []

            for (x, y, w, h) in faces:
                # Filtrar rostros con proporciones inusuales
                if w / h < 0.7 or w / h > 1.3:
                    continue

                face_img = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                dominant_emotion, score = self.estimate_emotion(face_img, gray_face, x, y, w, h)
                if dominant_emotion is None:
                    continue

                # Suavizar detecciones con promedio ponderado
                self.emotion_history.append((dominant_emotion, score))
                if len(self.emotion_history) >= 5:
                    emotion_scores = {label: 0.0 for label in self.emotion_labels}
                    for i, (emo, s) in enumerate(self.emotion_history):
                        emotion_scores[emo] += s * self.weights[i]
                    most_common = max(emotion_scores, key=emotion_scores.get)
                    score = emotion_scores[most_common] / sum(self.weights)
                else:
                    most_common, score = dominant_emotion, score

                results.append({
                    "box": [x, y, w, h],
                    "emotion": most_common,
                    "score": score
                })

            return results
        except Exception as e:
            print(f"Error al detectar emociones: {e}")
            return []
