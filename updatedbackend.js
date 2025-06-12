const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const filePath = path.join(__dirname, 'uploads', req.file.filename);
  const scriptPath = path.join(__dirname, 'transformers3.py');
  const command = `python "${scriptPath}" "${filePath}"`;

  console.log('Running command:', command); // For debugging

  exec(command, (err, stdout, stderr) => {
    if (err) {
      console.error(`Error executing transformers3.py:\n${stderr}`);
      return res.status(500).json({ error: 'Error during anomaly detection' });
    }

    try {
      const result = JSON.parse(stdout); // Parse only valid JSON
      res.setHeader('Content-Type', 'application/json');
      return res.json({
        file: {
          filename: req.file.filename,
          path: filePath,
        },
        anomalies: result.anomalies,
        losses: result.losses,
        threshold: result.threshold,
        combined_scores: result.combined_scores,
      });
    } catch (parseError) {
      console.error('Error parsing Python output:', parseError);
      return res.status(500).json({ error: 'Failed to parse Python output' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
