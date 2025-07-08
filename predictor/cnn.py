import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv('patches_with_majority_pathology.csv')

# Features: columns 0-1535 (assuming these are named as strings '0', '1', ..., '1535')
feature_cols = [str(i) for i in range(1536)]
X = df[feature_cols].values.astype(np.float32)
y = df['Majority_Pathology'].values

# Encode labels if not numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y).astype(np.int64)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape for CNN: (samples, channels, length)
torch_X_train = torch.tensor(X_train).unsqueeze(1)  # shape: (N, 1, 1536)
torch_y_train = torch.tensor(y_train)
torch_X_test = torch.tensor(X_test).unsqueeze(1)
torch_y_test = torch.tensor(y_test)

# DataLoader
train_ds = TensorDataset(torch_X_train, torch_y_train)
test_ds = TensorDataset(torch_X_test, torch_y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define a simple 1D CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 384, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

num_classes = len(np.unique(y_encoded))
model = SimpleCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader.dataset):.4f}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc:.3f}")

train(model, train_loader, criterion, optimizer, epochs=30)
evaluate(model, test_loader)
