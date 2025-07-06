import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Data loading utility
def load_data(input_file, test_size=0.2, random_state=42):
    df = pd.read_csv(input_file)
    feature_cols = [str(i) for i in range(1536)]
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['label'].values).astype(np.int64)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model definitions
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.name = "DNN"
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 384, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.name = "CNN"
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

# Training utility
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{model.name}: Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in test_loader)
    return correct / len(test_loader.dataset)

# Create and save DNN model
def create_dnn_model(input_file, output_path, epochs=5):
    X_train, X_test, y_train, y_test = load_data(input_file)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32)
    model = SimpleDNN(1536, len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    print(f"DNN Accuracy: {acc:.3f}")
    # Re-fit label encoder to capture original class names
    df = pd.read_csv(input_file)
    le = LabelEncoder().fit(df['label'].values)
    # Save checkpoint with weights and label classes
    torch.save({
        'state_dict': model.state_dict(),
        'label_classes': le.classes_.tolist()
    }, output_path)
    print(f"DNN model + labels saved to {output_path}")

# Create and save CNN model
def create_cnn_model(input_file, output_path, epochs=5):
    X_train, X_test, y_train, y_test = load_data(input_file)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test)), batch_size=32)
    model = SimpleCNN(len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    print(f"CNN Accuracy: {acc:.3f}")
    # Re-fit label encoder to capture original class names
    df = pd.read_csv(input_file)
    le = LabelEncoder().fit(df['label'].values)
    # Save checkpoint with weights and label classes
    torch.save({
        'state_dict': model.state_dict(),
        'label_classes': le.classes_.tolist()
    }, output_path)
    print(f"CNN model + labels saved to {output_path}")
