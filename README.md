Time Series Anomaly Detection
A full-stack web application for detecting anomalies in time series data using a Transformer-based Python model and visualizing them interactively using React.
Features
 1)Upload time-series CSV or text files
 2)Backend processes data using a Python Transformer model
 3)Detects anomalies and visualizes them
 4)Interactive line chart with anomalies and zoom
 5)Click to highlight specific anomalies
 6)Paginated anomaly list
 Custom threshold and scoring logic handled in Python
 Example visualization with anomaly detection:
 ![image](https://github.com/user-attachments/assets/797a9135-a444-4fc6-a6eb-608a556fb11e)
 ![image](https://github.com/user-attachments/assets/6f773615-1bf8-42b2-bf6c-72b3504625c1)
 Technologies Used:
Frontend (React)
 React.js
 Axios
 Chart.js
 HTML + CSS
Backend (Node + Python)
 Node.js + Express.js
 Multer for file uploads
 Python 3 (transformers, pandas, numpy, etc.)
 Child process (exec) to call Python scripts
Installation Prerequisites:
Node.js and npm
Python 3.7+
pip
Backend Setup:
cd backend
npm install
Install 
Python dependencies:
pip install -r requirements.txt
Frontend Setup:
cd frontend
npm install
npm start
Upload Format
Acceptable formats: .csv
File should have 2 columns: time and value
Anomaly Detection Logic:
Load the uploaded file
Run a transformer model over the series
Compute anomaly scores
Return indices of detected anomalies

