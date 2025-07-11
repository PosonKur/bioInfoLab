import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



# Model definitions
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
        self.name = "DNN"
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_dim=1536, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_p),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_p)
        )
        # Calculate the output size after convolution
        conv_output_size = 32 * (input_dim // 4)  # After two MaxPool1d(2)
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
        self.name = "CNN"
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

class SimpleAttention(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_p=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
        self.name = "ATTN"
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)  # x: (batch, seq_len, 1)
        x2 = attn_out.squeeze(-1)
        x2 = self.norm(x2)
        x2 = self.dropout(x2)
        return self.fc(x2)

# Training utility
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            # Apply sample weights to the loss
            loss = criterion(outputs, yb)
            weighted_loss = (loss * wb).mean()
            weighted_loss.backward()
            total_loss += weighted_loss.item()
            optimizer.step()
        print(f"{model.name}: Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    correct = 0
    total = 0
    for xb, yb, _ in test_loader:  # Ignore weights during evaluation
        with torch.no_grad():
            outputs = model(xb)
            predicted = outputs.argmax(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return correct / total

# Create and save DNN model
def create_dnn_model(X_train, X_test, y_train, y_test, output_path, labelEncoder, inputDim=1536, epochs=5, dropout_p=0.5, train_weights=None, test_weights=None):
    # Require confidence scores
    if train_weights is None or test_weights is None:
        raise ValueError("train_weights and test_weights (confidence scores) are required!")
    
    print("Training DNN with confidence-based sample weights")
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(train_weights))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(test_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = SimpleDNN(inputDim, len(np.unique(y_train)), dropout_p=dropout_p)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Always use none for weighted loss
    optimizer = optim.Adam(model.parameters())
    acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    print(f"DNN Accuracy: {acc:.3f}")

    return model, acc

# Create and save CNN model
def create_cnn_model(X_train, X_test, y_train, y_test, output_path, labelEncoder, inputDim=1536, epochs=5, dropout_p=0.5, train_weights=None, test_weights=None):
    # Require confidence scores
    if train_weights is None or test_weights is None:
        raise ValueError("train_weights and test_weights (confidence scores) are required!")
    
    print("Training CNN with confidence-based sample weights")
    train_dataset = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train), torch.tensor(train_weights))
    test_dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test), torch.tensor(test_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = SimpleCNN(len(np.unique(y_train)), inputDim, dropout_p=dropout_p)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Always use none for weighted loss
    optimizer = optim.Adam(model.parameters())
    acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    print(f"CNN Accuracy: {acc:.3f}")

    return model, acc

# Create and save Attention model
def create_attention_model(X_train, X_test, y_train, y_test, output_path, labelEncoder, epochs=5, dropout_p=0.5):
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).unsqueeze(-1), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).unsqueeze(-1), torch.tensor(y_test)), batch_size=32)
    model = SimpleAttention(1536, len(np.unique(y_train)), dropout_p=dropout_p)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    print(f"ATTN Accuracy: {acc:.3f}")
    torch.save({
        'state_dict': model.state_dict(),
        'label_classes': labelEncoder.classes_.tolist()
    }, output_path)
    print(f"Attention model + labels saved to {output_path}")
