import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

# Load data
boston = load_boston()
data = boston['data'][:, 5].reshape(-1, 1)  # Select one feature for univariate forecasting
target = boston['target']

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
train_target = target[:train_size]
test_target = target[train_size:]

# Convert data to PyTorch tensors and create datasets and data loaders
train_tensor = torch.Tensor(train_data).view(-1, 1, 1)
test_tensor = torch.Tensor(test_data).view(-1, 1, 1)
train_target_tensor = torch.Tensor(train_target)
test_target_tensor = torch.Tensor(test_target)
train_dataset = TensorDataset(train_tensor, train_target_tensor)
test_dataset = TensorDataset(test_tensor, test_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1)

# Initialize model, loss function, and optimizer
model = RNN(1, 16, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 100
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    for feature, target in train_loader:
        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * feature.size(0)
    train_loss /= len(train_loader.dataset)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for feature, target in test_loader:
            output = model(feature)
            loss = criterion(output, target)
            test_loss += loss.item() * feature.size(0)
    test_loss /= len(test_loader.dataset)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Make predictions on test set
model.eval()
with torch.no_grad():
    test_predictions = []
    for feature, target in test_loader:
        output = model(feature)
        test_predictions += output.tolist()

# Inverse transform the predictions
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate root mean squared error (RMSE)
test_rmse = ((test_predictions - test_target) ** 2).mean() ** 0.5
print("Test RMSE:", test_rmse)

# Plot the predictions and actual values
plt.plot(test_target, label="True values")
plt.plot(test_predictions, label="Predictions")
plt.legend()
plt.show()