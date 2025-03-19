import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import TimeSeriesChart from "./components/TimeSeriesChart";
import AnomalyDetector from "./components/AnomalyDetector";
import "./App.css";

const App = () => {
  const [data, setData] = useState([]);
  const [anomalies, setAnomalies] = useState([]);

  const handleDataUpload = (uploadedData) => {
    // Assuming uploadedData is an array of objects with 'time' and 'value' columns
    const formattedData = uploadedData.map((row) => ({
      time: row.time,  // Time column from CSV
      value: parseFloat(row.value),  // Value column from CSV, converting to float
    }));
    setData(formattedData);
  };

  const fetchAnomalies = async () => {
    try {
      const response = await fetch("http://localhost:4000/api/anomalies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data }), // Sending formatted data as JSON
      });

      if (!response.ok) {
        throw new Error("Anomaly detection request failed");
      }

      const result = await response.json();
      setAnomalies(result.anomalies || []);
    } catch (error) {
      console.error("Error detecting anomalies:", error);
    }
  };

  return (
    <div className="App">
      <h1>Time Series Anomaly Detection</h1>
      <FileUpload onDataUpload={handleDataUpload} />
      {data.length > 0 && (
        <>
          <TimeSeriesChart data={data} anomalies={anomalies} />
          <AnomalyDetector onDetectAnomalies={fetchAnomalies} />
        </>
      )}
    </div>
  );
};

export default App;
