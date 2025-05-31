import { useEffect, useRef, useState } from "react";

const EmotionDetector = ({ onEmotionDetected }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then((stream) => {
                videoRef.current.srcObject = stream;
                console.log("Webcam accedida correctamente");
                // Esperar a que el video tenga metadatos cargados
                videoRef.current.onloadedmetadata = () => {
                    console.log("Metadatos del video cargados");
                };
            })
            .catch((err) => {
                setError("Error al acceder a la webcam: " + err.message);
                console.error("Webcam error:", err);
            });

        wsRef.current = new WebSocket("ws://localhost:8000/ws/emotions");
        console.log("Intentando conectar a WebSocket: ws://localhost:8000/ws/emotions");

        wsRef.current.onopen = () => {
            console.log("WebSocket conectado");
        };
        wsRef.current.onmessage = (event) => {
            const results = JSON.parse(event.data);
            console.log("Resultados recibidos:", results);
            const ctx = canvasRef.current.getContext("2d");
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            if (results.length > 0) {
                results.forEach((result) => {
                    const [x, y, w, h] = result.box;
                    ctx.strokeStyle = "green";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);
                    ctx.font = "20px Arial";
                    ctx.fillStyle = "green";
                    ctx.fillText(
                        `${result.emotion}: ${result.score.toFixed(2)}`,
                        x,
                        y - 10
                    );
                    onEmotionDetected(result.emotion);
                });
            } else {
                console.log("No se detectaron rostros o emociones");
            }
        };

        wsRef.current.onclose = () => console.log("WebSocket cerrado");
        wsRef.current.onerror = (err) => {
            console.error("Error en WebSocket:", err);
            setError("Error en WebSocket: " + (err.message || "Desconocido"));
        };

        return () => {
            if (wsRef.current) wsRef.current.close();
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
            }
        };
    }, [onEmotionDetected]);

    useEffect(() => {
        const sendFrame = () => {
            if (
                videoRef.current &&
                canvasRef.current &&
                wsRef.current?.readyState === WebSocket.OPEN &&
                videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA
            ) {
                canvasRef.current.width = videoRef.current.videoWidth;
                canvasRef.current.height = videoRef.current.videoHeight;
                const ctx = canvasRef.current.getContext("2d");
                ctx.drawImage(videoRef.current, 0, 0);
                const dataUrl = canvasRef.current.toDataURL("image/jpeg", 0.7);
                // Verificar que dataUrl tenga datos válidos
                if (dataUrl.length > 100) { // Un dataURL válido debería ser mucho más largo que 100 caracteres
                    wsRef.current.send(dataUrl);
                    console.log("Frame válido enviado al WebSocket");
                } else {
                    console.log("Frame inválido, no enviado");
                }
            } else {
                console.log("No se puede enviar frame. WebSocket estado:", wsRef.current?.readyState, "Video estado:", videoRef.current?.readyState);
            }
            requestAnimationFrame(sendFrame);
        };

        requestAnimationFrame(sendFrame);
    }, []);

    return (
        <div className="relative mb-6">
            {error && <p className="text-red-500 text-center mb-2">{error}</p>}
            <video ref={videoRef} autoPlay className="w-full max-w-xl rounded-lg shadow-lg" />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-full max-w-xl" />
        </div>
    );
};

export default EmotionDetector;