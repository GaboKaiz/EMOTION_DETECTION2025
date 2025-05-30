import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar los clasificadores Haar Cascade para detección de rostros, sonrisas y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: No se pudo cargar el clasificador Haar Cascade para rostros")
    exit()

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
if smile_cascade.empty():
    print("Error: No se pudo cargar el clasificador de sonrisas")
    exit()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if eye_cascade.empty():
    print("Error: No se pudo cargar el clasificador de ojos")
    exit()

# Etiquetas de emociones ampliadas
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam")
    exit()

# Diccionario para contar la frecuencia de emociones
emotion_counts = {label: 0 for label in emotion_labels}

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

def estimate_emotion(face_img, gray_face, x, y, w, h):
    """Estima la emoción basada en la detección de sonrisas, ojos y forma de la boca."""
    # Detectar sonrisas en la región del rostro
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.7, minNeighbors=20, minSize=(20, 20))
    
    # Detectar ojos en la región superior del rostro (primeros 2/3)
    eyes = eye_cascade.detectMultiScale(gray_face[0:int(h*0.6)], scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
    
    # Extraer la región inferior del rostro (donde está la boca)
    mouth_region = gray_face[int(h*0.6):h, 0:w]
    if mouth_region.size == 0:
        return "neutral", 0.5
    
    # Aplicar umbral para detectar bordes de la boca
    _, thresh = cv2.threshold(mouth_region, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mouth_shape = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 50:  # Filtrar contornos pequeños
            x_m, y_m, w_m, h_m = cv2.boundingRect(max_contour)
            mouth_shape = h_m / w_m  # Relación altura/ancho de la boca
    
    # Estimar apertura de ojos
    eye_openness = 0.0
    if len(eyes) >= 2:  # Necesitamos al menos dos ojos
        for (ex, ey, ew, eh) in eyes[:2]:  # Tomar los dos primeros ojos detectados
            eye_openness += eh / ew  # Relación altura/ancho como proxy de apertura
        eye_openness /= min(len(eyes), 2)  # Promedio de apertura
    
    # Lógica heurística para clasificar emociones
    if len(smiles) > 0 and mouth_shape is not None and mouth_shape > 0.3:
        # Sonrisa detectada y boca relativamente alta -> happy
        return "happy", 0.8
    elif eye_openness > 0.5:
        # Ojos muy abiertos -> surprise o fear
        if mouth_shape is not None and mouth_shape > 0.4:
            return "surprise", 0.75  # Boca abierta
        else:
            return "fear", 0.7  # Boca cerrada o pequeña
    elif mouth_shape is not None:
        if mouth_shape < 0.3:
            # Boca ancha y plana -> sad o angry
            if eye_openness < 0.2 and len(eyes) >= 2:
                return "angry", 0.7  # Ojos entrecerrados
            return "sad", 0.7
        elif mouth_shape > 0.6:
            # Boca apretada y alta -> disgust
            return "disgust", 0.65
    return "neutral", 0.5  # Default si no se detectan características claras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame")
        break

    # Convertir a escala de grises para detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraer la región del rostro
        face_img = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        try:
            # Estimar la emoción usando heurísticas
            dominant_emotion, score = estimate_emotion(face_img, gray_face, x, y, w, h)

            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Actualizar conteo de emociones
            emotion_counts[dominant_emotion] += 1
        except Exception as e:
            print(f"Error al procesar la emoción: {e}")
            continue

    # Actualizar el gráfico de barras
    counts = [emotion_counts[emotion] for emotion in emotions]
    for bar, count in zip(bars, counts):
        bar.set_height(count)
    ax.set_ylim(0, max(max(counts, default=0) + 10, 100))
    plt.draw()
    plt.pause(0.01)

    # Mostrar el frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()