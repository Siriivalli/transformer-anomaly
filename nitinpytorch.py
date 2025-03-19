import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.N = N
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)  # For sigma in prior association

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        sigma = torch.clamp(self.Ws(x), min=1e-3)  # For sigma in prior association

        # Prior association
        P = self.prior_association(sigma)

        # Series association
        S = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_model)
        S = (S - torch.mean(S, dim=-1, keepdim=True)) / (torch.std(S, dim=-1, keepdim=True) + 1e-5)
        S = torch.softmax(S, dim=-1)

        Z = S @ V

        return Z, P, S

    @staticmethod
    def prior_association(sigma):
        N = sigma.shape[0]
        p = torch.abs(torch.arange(N).unsqueeze(0) - torch.arange(N).unsqueeze(1)).float()
        gaussian = torch.exp(-0.5 * (p / sigma).pow(2)) / (torch.sqrt(2 * math.pi * sigma))
        prior_ass = gaussian / gaussian.sum(dim=1, keepdim=True)
        return prior_ass


class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
        super(AnomalyTransformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.lambda_ = lambda_
        self.hidden_dim = hidden_dim

        self.attention_layer = AnomalyAttention(N, d_model)
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        Z, P, S = self.attention_layer(x)
        hidden = torch.relu(self.hidden_layer(Z))
        x_hat = self.output_layer(hidden)
        return x_hat, P, S

    def layer_association_discrepancy(self, P, S):
        epsilon = 1e-10
        kl_div_sum = (F.kl_div((P + epsilon).log(), S + epsilon, reduction="sum") +
                      F.kl_div((S + epsilon).log(), P + epsilon, reduction="sum"))
        return kl_div_sum

    def association_discrepancy(self, P_list, S_list):
        discrepancies = torch.tensor([self.layer_association_discrepancy(P, S) for P, S in zip(P_list, S_list)])
        return discrepancies.mean()

    def loss_function(self, x_hat, x, P_list, S_list):
        frob_norm = torch.linalg.norm(x_hat - x, ord="fro")
        assoc_disc = self.association_discrepancy(P_list, S_list)
        return frob_norm + self.lambda_ * assoc_disc

    def min_loss(self, x_hat, x, P_list, S_list):
        # Detach P_list to prevent gradient computation for prior association
        p_list_detach = [P.detach() for P in P_list]
        return self.loss_function(x_hat, x, p_list_detach, S_list)

    def max_loss(self, x_hat, x, P_list, S_list):
        # Detach S_list to prevent gradient computation for series association
        s_list_detach = [S.detach() for S in S_list]
        return self.loss_function(x_hat, x, P_list, s_list_detach)

    def anomaly_score(self, x):
        x_hat, P_list, S_list = self(x)
        reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
        assoc_discrepancy = self.association_discrepancy(P_list, S_list)
        scores = F.softmax(-assoc_discrepancy, dim=0) * reconstruction_error
        return scores

    def compute_accuracy(self, anomaly_scores, labels, threshold=0.5):
        # Predict anomalies based on a threshold of anomaly scores
        predictions = (anomaly_scores > threshold).float()

        # Calculate accuracy
        correct_predictions = (predictions == labels).sum()
        accuracy = correct_predictions / len(labels)
        return accuracy


def train(model, data, labels, optimizer, epochs, threshold=0.5):
    model.train()
    losses = []
    min_loss_val = float('inf')  # Variable to track the minimum loss
    max_loss_val = float('-inf')  # Variable to track the maximum loss
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, P_list, S_list = model(data)

        # Compute min/max losses
        min_loss = model.min_loss(x_hat, data, P_list, S_list)
        max_loss = model.max_loss(x_hat, data, P_list, S_list)

        # You can use the min_loss or max_loss as the primary loss, depending on your need
        loss = min_loss  # or loss = max_loss

        # Check for NaN values in loss
        if torch.isnan(loss):
            print(f"NaN encountered at epoch {epoch + 1}, skipping...")
            continue

        loss.backward()
        optimizer.step()

        # Track min/max loss
        min_loss_val = min(min_loss_val, min_loss.item())
        max_loss_val = max(max_loss_val, max_loss.item())

        losses.append(loss.item())

        # Compute accuracy every 10 epochs (or any other interval)
        if (epoch + 1) % 10 == 0:
            anomaly_scores = model.anomaly_score(data)
            accuracy = model.compute_accuracy(anomaly_scores, labels, threshold)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Min Loss: {min_loss_val:.4f}, Max Loss: {max_loss_val:.4f}, Accuracy: {accuracy:.4f}")

    return losses, min_loss_val, max_loss_val


# Load and preprocess data
df = pd.read_csv('day.csv')  # Make sure this CSV file path is correct
df = df.drop(['Open time', 'Close time'], axis=1)
X = df.values

# For demonstration, we will randomly generate labels (1 for anomaly, 0 for normal)
# Replace this with your actual labels if you have them.
labels = np.random.randint(0, 2, len(X))  # Generate random labels for illustration
X_train = torch.FloatTensor(X)
labels = torch.FloatTensor(labels)  # Convert labels to torch tensor

# Model and training setup
N = X_train.shape[0]
d_model = X_train.shape[1]
hidden_dim = 64
lambda_ = 0.1

model = AnomalyTransformer(N, d_model, hidden_dim, lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10

losses, min_loss_val, max_loss_val = train(model, X_train, labels, optimizer, epochs)

# Save and reload model
torch.save(model.state_dict(), 'model.pth')
loaded_model = AnomalyTransformer(N, d_model, hidden_dim, lambda_)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

# Compute anomaly scores
anomaly_scores = loaded_model.anomaly_score(X_train)
norm_anomaly_scores = 0.1 + 0.8 * ((anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()))

# Display anomalies
for i, score in enumerate(norm_anomaly_scores):
    if score > 0.8:
        print(f"Index {i + 1}, Anomaly Score: {score.item():.4f}")

# Plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

print(f"Training finished. Min Loss: {min_loss_val:.4f}, Max Loss: {max_loss_val:.4f}")
