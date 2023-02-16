import torch
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
data = load_boston()
X = data['data']
y = data['target']

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Reshape data into 3D tensor
X = X.view(-1, 1, 13)

import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

# Create PyTorch dataloaders
batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# Instantiate model
input_size = 13
hidden_size = 32
num_layers = 2
output_size = 1
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
num_epochs = 100
for epoch in range(num_epochs):
    for xb, yb in train_dl:
        # Forward pass
        output = model(xb)
        loss = criterion(output, yb.unsqueeze(1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    test_loss = 0
    for xb, yb in test_dl:
        output = model(xb)
        test_loss += criterion(output, yb.unsqueeze(1)).item()
    test_loss /= len(test_dl)
print(f'Test loss: {test_loss:.4f}')

