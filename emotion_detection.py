import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

# Clase para detección de emociones con deepface
class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def detect_emotions(self, frame):
        try:
            # Analizar el frame con deepface
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(result, list):
                return [self.format_emotion(r) for r in result]
            return [self.format_emotion(result)]
        except Exception as e:
            print(f"Error en detección: {e}")
            return []

    def format_emotion(self, result):
        x, y, w, h = result.get('region', {'x': 0, 'y': 0, 'w': 0, 'h': 0})
        emotions = result.get('emotion', {})
        return {
            'box': [x, y, w, h],
            'emotions': {label: float(emotions.get(label, 0)) for label in self.emotion_labels}
        }

# Inicializar el detector de emociones
detector = EmotionDetector()

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam")
    exit()

# Diccionario para contar la frecuencia de emociones
emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

# Configurar la figura para el gráfico en tiempo real
plt.ion()
fig, ax = plt.subplots()
emotions = list(emotion_counts.keys())
counts = [0] * len(emotions)
bars = ax.bar(emotions, counts)
ax.set_ylim(0, 100)
ax.set_title("Frecuencia de Emociones Detectadas")
ax.set_ylabel("Frecuencia")
plt.xticks(rotation=45)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame")
        break

    # Detectar emociones
    results = detector.detect_emotions(frame)

    for face in results:
        emotions_detected = face['emotions']
        dominant_emotion = max(emotions_detected, key=emotions_detected.get)
        score = emotions_detected[dominant_emotion]

        # Dibujar rectángulo y etiqueta
        box = face['box']
        x, y, w, h = box
        if w > 0 and h > 0:  # Verificar que el cuadro sea válido
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        emotion_counts[dominant_emotion] += 1

    # Actualizar el gráfico de barras
    counts = [emotion_counts[emotion] for emotion in emotions]
    for bar, count in zip(bars, counts):
        bar.set_height(count)
    ax.set_ylim(0, max(max(counts) + 10, 100))
    plt.draw()
    plt.pause(0.01)

    # Mostrar el frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()