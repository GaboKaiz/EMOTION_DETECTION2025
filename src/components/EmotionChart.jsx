import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

const EmotionChart = ({ emotionCounts }) => {
    const data = Object.keys(emotionCounts).map((emotion) => ({
        emotion,
        count: emotionCounts[emotion],
    }));

    return (
        <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Frecuencia de Emociones Detectadas</h2>
            <BarChart width={600} height={300} data={data} className="mt-4">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="emotion" angle={45} textAnchor="start" height={60} />
                <YAxis label={{ value: "Frecuencia", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Bar dataKey="count" fill="#82ca9d" />
            </BarChart>
        </div>
    );
};

export default EmotionChart;