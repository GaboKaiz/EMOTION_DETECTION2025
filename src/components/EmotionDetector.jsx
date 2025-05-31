import { useEffect, useRef, useState } from "react";

// Traducción de emociones
const emotionsTranslation = {
    happy: "feliz",
    angry: "enojado",
    surprised: "sorprendido",
    sad: "triste",
    disgust: "desagrado",
    fear: "miedo",
    neutral: "tranquilo",
};

const EmotionDetector = ({ onEmotionDetected }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const [error, setError] = useState(null);
    const lastFrameTime = useRef(0);
    const frameInterval = 3000; // 1 frame cada 3 segundos para máxima estabilidad
    const isMounted = useRef(true);
    const [personCount, setPersonCount] = useState(0);
    const [emotionsSummary, setEmotionsSummary] = useState("Ninguna detectada");
    const [isCameraActive, setIsCameraActive] = useState(false);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" },
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
                console.log("Cámara iniciada correctamente");
                setIsCameraActive(true);
                setError(null);
            }
        } catch (err) {
            console.error("Error al iniciar cámara:", err);
            setError("No se pudo iniciar la cámara: " + err.message);
            setIsCameraActive(false);
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
            setIsCameraActive(false);
            console.log("Cámara detenida");
        }
    };

    useEffect(() => {
        isMounted.current = true;

        const setupWebSocket = () => {
            wsRef.current = new WebSocket("ws://localhost:8000/ws/emotions");
            console.log("Intentando conectar a WebSocket");

            wsRef.current.onopen = () => {
                console.log("WebSocket conectado");
                setError(null);
            };

            wsRef.current.onmessage = (event) => {
                if (!isMounted.current) return;
                try {
                    const results = JSON.parse(event.data);
                    console.log("Resultados recibidos:", results);

                    if (canvasRef.current) {
                        const ctx = canvasRef.current.getContext("2d");
                        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                        if (results.detections && results.detections.length > 0) {
                            results.detections.forEach((result) => {
                                const [x, y, w, h] = result.box;
                                ctx.strokeStyle = "#00FF00";
                                ctx.lineWidth = 3;
                                ctx.strokeRect(x, y, w, h);

                                const translatedEmotion = emotionsTranslation[result.emotion] || result.emotion;
                                const text = `${translatedEmotion}: ${result.score.toFixed(2)}`;
                                const textWidth = ctx.measureText(text).width;
                                ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
                                ctx.fillRect(x - 5, y - 40, textWidth + 10, 30);
                                ctx.font = "16px Arial";
                                ctx.fillStyle = "#FF0000";
                                ctx.fillText(text, x, y - 25);

                                onEmotionDetected(result.emotion);
                            });
                        }
                    }

                    setPersonCount(results.person_count || 0);
                    setEmotionsSummary(
                        results.emotions_count
                            ? Object.entries(results.emotions_count)
                                .filter(([_, count]) => count > 0)
                                .map(([emotion, count]) => `${emotionsTranslation[emotion] || emotion}: ${count}`)
                                .join(", ") || "Ninguna detectada"
                            : "Ninguna detectada"
                    );
                } catch (err) {
                    console.error("Error al procesar mensaje:", err);
                    setError("Error al procesar datos: " + err.message);
                }
            };

            wsRef.current.onclose = () => {
                console.log("WebSocket cerrado");
                if (isMounted.current) {
                    setError("Conexión WebSocket perdida. Reintentando en 15 segundos...");
                    setTimeout(() => {
                        wsRef.current = null;
                        setupWebSocket();
                    }, 15000); // 15 segundos para evitar intentos rápidos
                }
            };

            wsRef.current.onerror = (err) => {
                console.error("Error en WebSocket:", err);
                setError("Error en WebSocket: " + err.message);
            };
        };

        setupWebSocket();

        return () => {
            isMounted.current = false;
            if (wsRef.current) {
                wsRef.current.close();
            }
            stopCamera();
        };
    }, [onEmotionDetected]);

    useEffect(() => {
        const sendFrame = () => {
            if (!isMounted.current || !isCameraActive) return;

            const now = Date.now();
            if (now - lastFrameTime.current < frameInterval) {
                requestAnimationFrame(sendFrame);
                return;
            }

            if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
                canvasRef.current.width = 640;
                canvasRef.current.height = 480;
                const ctx = canvasRef.current.getContext("2d");
                ctx.drawImage(videoRef.current, 0, 0, 640, 480);
                const dataUrl = canvasRef.current.toDataURL("image/jpeg", 0.5);
                if (dataUrl.length > 100) {
                    wsRef.current.send(dataUrl);
                    console.log("Frame enviado al WebSocket");
                    lastFrameTime.current = now;
                }
            }
            requestAnimationFrame(sendFrame);
        };

        if (isCameraActive) {
            requestAnimationFrame(sendFrame);
        } else {
            stopCamera();
        }
    }, [isCameraActive]);

    return (
        <div className="relative mb-6 flex flex-col items-center">
            {error && <p className="text-red-500 text-center mb-2 bg-white bg-opacity-80 p-2 rounded">{error}</p>}
            <div className="relative w-full max-w-xl">
                <video ref={videoRef} autoPlay className="w-full rounded-lg shadow-lg" />
                <canvas ref={canvasRef} className="absolute top-0 left-0 w-full max-w-xl" />
                <div className="absolute top-2 left-2 bg-gray-900 bg-opacity-70 text-white font-bold p-3 rounded-lg shadow-md">
                    Personas: {personCount}
                </div>
                <div className="absolute top-14 left-2 bg-gray-900 bg-opacity-70 text-white text-sm p-3 rounded-lg shadow-md max-w-[calc(100%-1rem)]">
                    Emociones: {emotionsSummary}
                </div>
            </div>
            <button
                onClick={() => {
                    if (isCameraActive) {
                        stopCamera();
                    } else {
                        startCamera();
                    }
                }}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
                {isCameraActive ? "Apagar Cámara" : "Encender Cámara"}
            </button>
        </div>
    );
};

export default EmotionDetector;