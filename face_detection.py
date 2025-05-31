import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar el clasificador Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: No se pudo cargar el clasificador Haar Cascade")
    exit()

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam")
    exit()

# Lista para contar la cantidad de rostros por frame
face_counts = []

# Configurar la figura para el gráfico en tiempo real
plt.ion()
fig, ax = plt.subplots()
frames = []
counts = []
bars = ax.bar(frames, counts)
ax.set_ylim(0, 5)  # Ajustar según el número máximo esperado de rostros
ax.set_title("Cantidad de Rostros Detectados por Frame")
ax.set_xlabel("Frame")
ax.set_ylabel("Cantidad de Rostros")

frame_count = 0

while True:
    # Capturar frame de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame")
        break

    # Convertir a escala de grises para la detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Rostro", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Contar rostros detectados en este frame
    num_faces = len(faces)
    face_counts.append(num_faces)
    frame_count += 1

    # Actualizar el gráfico de barras
    frames = list(range(1, frame_count + 1))[-50:]  # Mostrar últimos 50 frames
    counts = face_counts[-50:]  # Mantener las últimas 50 detecciones
    ax.clear()
    ax.bar(frames, counts)
    ax.set_ylim(0, max(max(counts, default=0) + 1, 5))
    ax.set_title("Cantidad de Rostros Detectados por Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cantidad de Rostros")
    plt.draw()
    plt.pause(0.01)

    # Mostrar el frame
    cv2.imshow('Face Detection', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
plt.close()