import { useState } from "react";
import EmotionDetector from "./components/EmotionDetector";
import EmotionChart from "./components/EmotionChart";

function App() {
  const [emotionCounts, setEmotionCounts] = useState({
    neutral: 0,
    happy: 0,
    sad: 0,
    angry: 0,
    surprise: 0,
    disgust: 0,
    fear: 0,
  });

  const handleEmotionDetected = (emotion) => {
    setEmotionCounts((prev) => ({
      ...prev,
      [emotion]: prev[emotion] + 1,
    }));
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold mb-6 text-gray-800">Detección de Emociones</h1>
      <EmotionDetector onEmotionDetected={handleEmotionDetected} />
      <EmotionChart emotionCounts={emotionCounts} />
    </div>
  );
}

export default App;