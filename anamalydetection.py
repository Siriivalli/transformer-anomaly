'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TransformerAnomalyDetection:
    def __init__(self, input_dim, embed_dim, latent_dim, num_epochs=10):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs

        # Encoder weights
        self.encoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Decoder weights
        self.decoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Reconstruction weights
        self.reconstruction_W = np.random.randn(latent_dim, input_dim) * np.sqrt(2 / input_dim)

    def preprocess_data(self, data, columns):
        # Standardize data (zero mean, unit variance)
        self.mean = data[columns].mean()
        self.std = data[columns].std()
        data[columns] = (data[columns] - self.mean) / self.std
        return data

    def positional_encoding(self, sequence_length, embed_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pos_encoding = np.zeros((sequence_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def encode(self, X):
        Q = X @ self.encoder_W_Q
        K = X @ self.encoder_W_K
        V = X @ self.encoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ V, attention_scores

    def decode(self, encoded_X):
        Q = encoded_X @ self.decoder_W_Q
        K = encoded_X @ self.decoder_W_K
        V = encoded_X @ self.decoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        decoded_X = attention_weights @ V
        
        # Project decoded_X to latent_dim
        projected_decoded_X = decoded_X @ np.random.randn(self.embed_dim, self.latent_dim)
        return projected_decoded_X

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def reconstruct(self, decoded_X):
        # If you want to reconstruct back to the original input dimensions (8),
        # you need to project the decoded_X back to the input space.
        reconstructed_X = decoded_X @ self.reconstruction_W.T  # (batch_size, latent_dim) @ (input_dim, latent_dim)
        return reconstructed_X

    def calculate_loss(self, X, reconstructed_X):
        # Project X to match the latent dimension (shape 1, latent_dim)
        X_projected = X[:self.latent_dim]  # Take only the first 'latent_dim' features from X if necessary
        # Calculate loss
        return np.mean((X_projected - reconstructed_X) ** 2)

    def min_max_scale(self, losses):
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        return (losses - min_loss) / (max_loss - min_loss)

    def association_discrepancy(self, attention_scores):
        # Calculate association discrepancy (simple difference from mean)
        avg_attention = np.mean(attention_scores, axis=0)
        discrepancy = np.abs(attention_scores - avg_attention)
        return np.mean(discrepancy, axis=-1)

    def detect_anomalies(self, X, threshold_multiplier=2):
        losses = []
        attention_scores_all = []
        association_discrepancies = []

        for i in range(X.shape[0]):
            # Reshape X[i] to a 2D array of shape (1, embed_dim)
            encoded_X, attention_scores = self.encode(X[i:i+1])  # X[i:i+1] ensures a 2D shape
            decoded_X = self.decode(encoded_X)
            reconstructed_X = self.reconstruct(decoded_X)
            loss = self.calculate_loss(X[i], reconstructed_X)
            losses.append(loss)

            # Calculate association discrepancy
            association_discrepancy_value = self.association_discrepancy(attention_scores)
            association_discrepancies.append(np.mean(association_discrepancy_value))

            attention_scores_all.append(attention_scores)
        
        # Apply Min-Max scaling to the reconstruction losses
        scaled_losses = self.min_max_scale(np.array(losses))

        # Calculate anomaly threshold based on association discrepancy and reconstruction loss
        mean_loss = np.mean(scaled_losses)
        std_loss = np.std(scaled_losses)
        threshold = mean_loss + threshold_multiplier * std_loss
        
        # Combine anomaly scores from both losses and attention discrepancies
        combined_scores = scaled_losses + np.array(association_discrepancies)
        anomalies = np.where(combined_scores > threshold)[0]
        
        return anomalies, scaled_losses, threshold, combined_scores

# Load and preprocess data
data = pd.read_csv("D:/ps/preprocessed_data.csv")
data = data.dropna()

selected_columns = ['Open', 'High', 'Low', 'Close']

model = TransformerAnomalyDetection(input_dim=4, embed_dim=8, latent_dim=4, num_epochs=10)
data = model.preprocess_data(data, selected_columns)

# Extract normalized embeddings
sequence_length = 15000
X = data[selected_columns].iloc[:sequence_length].values

# Project X to match embed_dim
projection_matrix = np.random.randn(X.shape[1], model.embed_dim)  # Random projection
X_projected = X @ projection_matrix  # Now X_projected has shape (1000, embed_dim)

# Add positional encoding
positional_enc = model.positional_encoding(sequence_length, embed_dim=model.embed_dim)
X_embedded = X_projected + positional_enc

# Training loop
for epoch in range(1, model.num_epochs + 1):
    anomalies, losses, threshold, combined_scores = model.detect_anomalies(X_embedded)
    
    # Calculate average loss for the epoch
    avg_loss = np.mean(losses)
    
    # Print epoch information
    print(f"Epoch {epoch}/{model.num_epochs}, Loss: {avg_loss}")

# Display anomalies detected
print(f"Anomalies detected at indices: {anomalies}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Reconstruction Loss')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies, [losses[i] for i in anomalies], color='red', label='Anomalies')
plt.legend()
plt.title('Anomaly Detection with Transformer and Anomaly Attention')
plt.show()'''

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TransformerAnomalyDetection:
    def __init__(self, input_dim, embed_dim, latent_dim, num_epochs=10, learning_rate=0.001):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Encoder weights
        self.encoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Decoder weights
        self.decoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Reconstruction weights
        self.reconstruction_W = np.random.randn(latent_dim, input_dim) * np.sqrt(2 / input_dim)

    def preprocess_data(self, data, columns):
        # Standardize data (zero mean, unit variance)
        self.mean = data[columns].mean()
        self.std = data[columns].std()
        data[columns] = (data[columns] - self.mean) / self.std
        return data

    def positional_encoding(self, sequence_length, embed_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pos_encoding = np.zeros((sequence_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def encode(self, X):
        Q = X @ self.encoder_W_Q
        K = X @ self.encoder_W_K
        V = X @ self.encoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ V, attention_scores

    def decode(self, encoded_X):
        Q = encoded_X @ self.decoder_W_Q
        K = encoded_X @ self.decoder_W_K
        V = encoded_X @ self.decoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        decoded_X = attention_weights @ V
        
        # Project decoded_X to latent_dim
        projected_decoded_X = decoded_X @ np.random.randn(self.embed_dim, self.latent_dim)
        return projected_decoded_X

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def reconstruct(self, decoded_X):
        # If you want to reconstruct back to the original input dimensions (8),
        # you need to project the decoded_X back to the input space.
        reconstructed_X = decoded_X @ self.reconstruction_W.T  # (batch_size, latent_dim) @ (input_dim, latent_dim)
        return reconstructed_X

    def calculate_loss(self, X, reconstructed_X):
        # Project X to match the latent dimension (shape 1, latent_dim)
        X_projected = X[:self.latent_dim]  # Take only the first 'latent_dim' features from X if necessary
        # Calculate loss
        return np.mean((X_projected - reconstructed_X) ** 2)

    def min_max_scale(self, losses):
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        return (losses - min_loss) / (max_loss - min_loss)

    def association_discrepancy(self, attention_scores):
        # Calculate association discrepancy (simple difference from mean)
        avg_attention = np.mean(attention_scores, axis=0)
        discrepancy = np.abs(attention_scores - avg_attention)
        return np.mean(discrepancy, axis=-1)

    def backpropagate(self, X, reconstructed_X, encoded_X, attention_scores):
        # Gradient of loss with respect to reconstruction weights
        reconstruction_loss_grad = 2 * (reconstructed_X - X)
        reconstruction_grad = reconstruction_loss_grad @ self.reconstruction_W  # Backpropagate through reconstruction layer
        
        # Simple gradient updates (without full backpropagation chain)
        self.reconstruction_W -= self.learning_rate * reconstruction_grad

        # Add any additional backpropagation for attention layers here
        # For simplicity, we will not add the full chain, but you can expand it as needed
        # This might involve computing gradients with respect to Q, K, V matrices
        pass

    def detect_anomalies(self, X, threshold_multiplier=2):
        losses = []
        attention_scores_all = []
        association_discrepancies = []

        for i in range(X.shape[0]):
            # Reshape X[i] to a 2D array of shape (1, embed_dim)
            encoded_X, attention_scores = self.encode(X[i:i+1])  # X[i:i+1] ensures a 2D shape
            decoded_X = self.decode(encoded_X)
            reconstructed_X = self.reconstruct(decoded_X)
            loss = self.calculate_loss(X[i], reconstructed_X)
            losses.append(loss)

            # Calculate association discrepancy
            association_discrepancy_value = self.association_discrepancy(attention_scores)
            association_discrepancies.append(np.mean(association_discrepancy_value))

            attention_scores_all.append(attention_scores)
        
        # Apply Min-Max scaling to the reconstruction losses
        scaled_losses = self.min_max_scale(np.array(losses))

        # Calculate anomaly threshold based on association discrepancy and reconstruction loss
        mean_loss = np.mean(scaled_losses)
        std_loss = np.std(scaled_losses)
        threshold = mean_loss + threshold_multiplier * std_loss
        
        # Combine anomaly scores from both losses and attention discrepancies
        combined_scores = scaled_losses + np.array(association_discrepancies)
        anomalies = np.where(combined_scores > threshold)[0]
        
        return anomalies, scaled_losses, threshold, combined_scores

    def train(self, X):
        for epoch in range(1, self.num_epochs + 1):
            anomalies, losses, threshold, combined_scores = self.detect_anomalies(X)
            
            # Average loss for the epoch
            avg_loss = np.mean(losses)

            # Backpropagation step
            # We need to pass the encoded data and attention scores here
            for i in range(X.shape[0]):
                encoded_X, attention_scores = self.encode(X[i:i+1])  # Encode the data
                reconstructed_X = self.reconstruct(self.decode(encoded_X))  # Reconstruct the data
                
                # Perform backpropagation based on the current data
                self.backpropagate(X[i], reconstructed_X, encoded_X, attention_scores)

            print(f"Epoch {epoch}/{self.num_epochs}, Loss: {avg_loss}")
        return anomalies, losses

# Load and preprocess data
data = pd.read_csv("D:/ps/preprocessed_data.csv")
data = data.dropna()

selected_columns = ['Open', 'High', 'Low', 'Close']

model = TransformerAnomalyDetection(input_dim=4, embed_dim=8, latent_dim=4, num_epochs=10)
data = model.preprocess_data(data, selected_columns)

# Extract normalized embeddings
sequence_length = 1000
X = data[selected_columns].iloc[:sequence_length].values

# Project X to match embed_dim
projection_matrix = np.random.randn(X.shape[1], model.embed_dim)  # Random projection
X_projected = X @ projection_matrix  # Now X_projected has shape (1000, embed_dim)

# Add positional encoding
positional_enc = model.positional_encoding(sequence_length, embed_dim=model.embed_dim)
X_embedded = X_projected + positional_enc

# Training loop
anomalies, losses = model.train(X_embedded)

# Display anomalies detected
print(f"Anomalies detected at indices: {anomalies}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Reconstruction Loss')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies, [losses[i] for i in anomalies], color='red', label='Anomalies')
plt.legend()
plt.title('Anomaly Detection with Transformer and Anomaly Attention')
plt.show()
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class TransformerAnomalyDetection:
    def __init__(self, input_dim, embed_dim, latent_dim, num_epochs=10):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        # Encoder weights
        self.encoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Decoder weights
        self.decoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)

        # Reconstruction weights
        self.reconstruction_W = np.random.randn(latent_dim, input_dim) * np.sqrt(2 / input_dim)

    def preprocess_data(self, data, columns):
        # Standardize data (zero mean, unit variance)
        self.mean = data[columns].mean()
        self.std = data[columns].std()
        data[columns] = (data[columns] - self.mean) / self.std
        return data

    def positional_encoding(self, sequence_length, embed_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pos_encoding = np.zeros((sequence_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def encode(self, X):
        Q = X @ self.encoder_W_Q
        K = X @ self.encoder_W_K
        V = X @ self.encoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ V, attention_scores

    def decode(self, encoded_X):
        Q = encoded_X @ self.decoder_W_Q
        K = encoded_X @ self.decoder_W_K
        V = encoded_X @ self.decoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        decoded_X = attention_weights @ V
        
        # Project decoded_X to latent_dim
        projected_decoded_X = decoded_X @ np.random.randn(self.embed_dim, self.latent_dim)
        return projected_decoded_X

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def reconstruct(self, decoded_X):
        reconstructed_X = decoded_X @ self.reconstruction_W.T
        return reconstructed_X

    def calculate_loss(self, X, reconstructed_X):
        return np.mean((X[:self.latent_dim] - reconstructed_X) ** 2)

    def min_max_scale(self, losses):
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        return (losses - min_loss) / (max_loss - min_loss)

    def association_discrepancy(self, attention_scores):
        avg_attention = np.mean(attention_scores, axis=0)
        discrepancy = np.abs(attention_scores - avg_attention)
        return np.mean(discrepancy, axis=-1)

    def detect_anomalies(self, X, threshold_multiplier=2):
        losses = []
        attention_scores_all = []
        association_discrepancies = []

        for i in range(X.shape[0]):
            encoded_X, attention_scores = self.encode(X[i:i+1])
            decoded_X = self.decode(encoded_X)
            reconstructed_X = self.reconstruct(decoded_X)
            loss = self.calculate_loss(X[i], reconstructed_X)
            losses.append(loss)

            association_discrepancy_value = self.association_discrepancy(attention_scores)
            association_discrepancies.append(np.mean(association_discrepancy_value))

            attention_scores_all.append(attention_scores)
        
        scaled_losses = self.min_max_scale(np.array(losses))
        mean_loss = np.mean(scaled_losses)
        std_loss = np.std(scaled_losses)
        threshold = mean_loss + threshold_multiplier * std_loss
        
        combined_scores = scaled_losses + np.array(association_discrepancies)
        anomalies = np.where(combined_scores > threshold)[0]
        
        return anomalies, scaled_losses, threshold, combined_scores

# Load and preprocess data
data = pd.read_csv("D:/ps/preprocessed_data.csv")
data = data.dropna()

selected_columns = ['Open', 'High', 'Low', 'Close']

model = TransformerAnomalyDetection(input_dim=4, embed_dim=8, latent_dim=4, num_epochs=10)
data = model.preprocess_data(data, selected_columns)

# Extract normalized embeddings
X = data[selected_columns].values  # Use all rows dynamically
sequence_length = X.shape[0]       # Dynamically set sequence length

# Project X to match embed_dim
projection_matrix = np.random.randn(X.shape[1], model.embed_dim)
X_projected = X @ projection_matrix

# Add positional encoding
positional_enc = model.positional_encoding(sequence_length, embed_dim=model.embed_dim)
X_embedded = X_projected + positional_enc

# Simulate True Labels (for demonstration purposes)
true_labels = np.zeros(X.shape[0], dtype=int)
# Simulate anomalies at random indices (for example, 10% of the data)
anomaly_indices = np.random.choice(X.shape[0], size=int(0.1 * X.shape[0]), replace=False)
true_labels[anomaly_indices] = 1

# Training loop
for epoch in range(1, model.num_epochs + 1):
    anomalies, losses, threshold, combined_scores = model.detect_anomalies(X_embedded)
    avg_loss = np.mean(losses)
    print(f"Epoch {epoch}/{model.num_epochs}, Loss: {avg_loss}")

# Evaluate metrics
predictions = np.zeros_like(true_labels)
predictions[anomalies] = 1

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)
roc_auc = roc_auc_score(true_labels, combined_scores)

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Reconstruction Loss')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies, [losses[i] for i in anomalies], color='red', label='Anomalies')
plt.legend()
plt.title('Anomaly Detection with Transformer and Anomaly Attention')
plt.show()
