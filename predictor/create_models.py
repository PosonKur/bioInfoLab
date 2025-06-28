import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    df = pd.read_csv('patches_with_majority_pathology.csv')
    feature_cols = [str(i) for i in range(1536)]
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Majority_Pathology'].values).astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes)
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
            nn.Linear(32 * 384, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
        self.name = "CNN"
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer,epochs=30):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        print(f"{model.name}: Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in test_loader)
    return correct / len(test_loader.dataset)

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    train_loader_dnn = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader_dnn = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32)
    train_loader_cnn = DataLoader(TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader_cnn = DataLoader(TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test)), batch_size=32)

    dnn_model = SimpleDNN(1536, len(np.unique(y_train)))
    cnn_model = SimpleCNN(len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    print("Training DNN...")
    dnn_acc = train_and_evaluate(dnn_model, train_loader_dnn, test_loader_dnn, criterion, optim.Adam(dnn_model.parameters()))
    print("Training CNN...")
    cnn_acc = train_and_evaluate(cnn_model, train_loader_cnn, test_loader_cnn, criterion, optim.Adam(cnn_model.parameters()))

    print(f"DNN Accuracy: {dnn_acc:.3f}, CNN Accuracy: {cnn_acc:.3f}")
    print("Saving models...")
    torch.save(dnn_model.state_dict(), 'models/dnn_model.pth')
    torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
    print("Models saved as 'models/dnn_model.pth' and 'models/cnn_model.pth'")

if __name__ == "__main__":
    main()