import React from 'react';

const AnomalyDetector = ({ onDetectAnomalies }) => {
  return (
    <div>
      <button onClick={onDetectAnomalies}>Detect Anomalies</button>
    </div>
  );
};

export default AnomalyDetector;
