import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing function to drop non-numeric columns
def preprocess_data(data):
    # Drop non-numeric columns
    data = data.select_dtypes(include=[np.number])
    return data

# Anomaly Transformer Class (as defined earlier)
class AnomalyTransformer:
    def __init__(self, input_dim, embed_dim, latent_dim, num_epochs=10, learning_rate=0.001, lambda_param=3.0):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param

        # Initialize weight matrices
        self.W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.reconstruction_W = np.random.randn(embed_dim, input_dim) * np.sqrt(2 / embed_dim)

    def positional_encoding(self, sequence_length, embed_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pos_encoding = np.zeros((sequence_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def softmax(self, X):
        max_X = np.max(X, axis=-1, keepdims=True)
        exp_X = np.exp(X - max_X)
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def anomaly_attention(self, X):
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        scaling_factor = np.sqrt(self.embed_dim)
        attention_scores = Q @ K.T / scaling_factor
        series_association = self.softmax(attention_scores)

        N = X.shape[0]
        sigma = 1.0
        temporal_dist = np.abs(np.arange(N)[:, None] - np.arange(N))
        prior_association = np.exp(-temporal_dist**2 / (2 * sigma**2))
        prior_association /= prior_association.sum(axis=1, keepdims=True)

        kl_div1 = np.sum(prior_association * np.log((prior_association + 1e-8) / (series_association + 1e-8)), axis=1)
        kl_div2 = np.sum(series_association * np.log((series_association + 1e-8) / (prior_association + 1e-8)), axis=1)
        association_discrepancy = kl_div1 + kl_div2

        return series_association @ V, association_discrepancy

    def reconstruct(self, encoded_X):
        return encoded_X @ self.reconstruction_W.T

    def calculate_loss(self, X, reconstructed_X, association_discrepancy):
        reconstruction_loss = np.mean((X - reconstructed_X) ** 2)
        discrepancy_loss = np.mean(association_discrepancy)
        return reconstruction_loss - self.lambda_param * discrepancy_loss

    def train(self, X):
        for epoch in range(self.num_epochs):
            total_loss = 0
            encoded_X, assoc_discrepancy = self.anomaly_attention(X)
            reconstructed_X = self.reconstruct(encoded_X)

            loss = self.calculate_loss(X, reconstructed_X, assoc_discrepancy)
            total_loss += loss

            grad_reconstruction_W = 2 * (reconstructed_X - X).T @ encoded_X / X.shape[0]
            self.reconstruction_W -= self.learning_rate * grad_reconstruction_W

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss:.4f}")

    def detect_anomalies(self, X, threshold_multiplier=2):
        anomalies = []
        scores = []
        encoded_X, assoc_discrepancy = self.anomaly_attention(X)
        reconstructed_X = self.reconstruct(encoded_X)

        for i in range(X.shape[0]):
            loss = self.calculate_loss(X[i:i+1], reconstructed_X[i:i+1], assoc_discrepancy[i:i+1])
            scores.append(loss)

        threshold = np.mean(scores) + threshold_multiplier * np.std(scores)
        anomalies = np.where(np.array(scores) > threshold)[0]
        return anomalies, scores, threshold

# Load and preprocess data
data = pd.read_csv("D:/ps/preprocessed_data.csv")  # Replace with your actual file path
processed_data = preprocess_data(data)

# Initialize the AnomalyTransformer
input_dim = processed_data.shape[1]
embed_dim = 16
latent_dim = 8

model = AnomalyTransformer(input_dim=input_dim, embed_dim=embed_dim, latent_dim=latent_dim, num_epochs=5)

# Prepare data for training (positional encoding + embedding)
X = processed_data.values
sequence_length = len(X)
positional_enc = model.positional_encoding(sequence_length, embed_dim=embed_dim)
X_embedded = X @ np.random.randn(X.shape[1], embed_dim) + positional_enc

# Train the model
model.train(X_embedded)

# Detect anomalies
anomalies, scores, threshold = model.detect_anomalies(X_embedded)

# Output results
print("Detected anomalies at indices:", anomalies)
print("Threshold for anomaly detection:", threshold)

# Plot anomalies
timestamps = range(len(scores))  # Use the index if timestamps are unavailable
plt.figure(figsize=(12, 6))
plt.plot(timestamps, scores, label="Anomaly Scores")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.scatter([timestamps[i] for i in anomalies], [scores[i] for i in anomalies], color='red', label="Anomalies")
plt.title("Anomaly Detection")
plt.xlabel("Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.show()
