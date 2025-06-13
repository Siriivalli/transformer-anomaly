import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

const TimeSeriesChart = ({ data, anomalies }) => {
  // Assuming 'data' is an array of objects with 'time' and 'value'
  const times = data.map(item => item.time);
  const values = data.map(item => item.value);

  const anomalyIndices = anomalies.map(item => item.index);  // Assuming anomalies have an 'index' property

  const chartData = {
    labels: times,
    datasets: [
      {
        label: 'Time Series Data',
        data: values,
        borderColor: 'blue',
        fill: false,
      },
      {
        label: 'Anomalies',
        data: anomalyIndices.map(index => values[index]),
        backgroundColor: 'red',
        borderColor: 'red',
        pointRadius: 5,
        pointHoverRadius: 8,
        pointStyle: 'rect',
      },
    ],
  };

  return <Line data={chartData} />;
};

export default TimeSeriesChart;
