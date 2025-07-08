import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [str(i) for i in range(1536)]
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['label'].values).astype(np.int64)
    
    # Load confidence scores if available
    if 'confidence' in df.columns:
        confidence = df['confidence'].values.astype(np.float32)
        print(f"Confidence scores loaded. Range: {confidence.min():.3f} - {confidence.max():.3f}")
    else:
        confidence = np.ones(len(y), dtype=np.float32)  # Default confidence = 1.0
        print("No confidence scores found, using default confidence = 1.0")
    
    # Print original label distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Original label distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Use stratified sampling to preserve label distribution
    X_train, X_test, y_train, y_test, conf_train, conf_test = train_test_split(
        X, y, confidence, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print train/test label distributions to verify
    print("\nTrain set label distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"  Label {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    print("\nTest set label distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_test, counts_test):
        print(f"  Label {label}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, conf_train, conf_test

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

class ConfidenceWeightedLoss(nn.Module):
    """Custom loss function that weights samples by their confidence scores"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs, targets, confidence_weights):
        # Calculate per-sample cross-entropy loss
        losses = self.ce_loss(outputs, targets)
        # Weight by confidence scores
        weighted_losses = losses * confidence_weights
        # Return mean weighted loss
        return weighted_losses.mean()

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (xb, yb, conf_b) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(xb)
            
            # Use confidence-weighted loss if available
            if isinstance(criterion, ConfidenceWeightedLoss):
                loss = criterion(outputs, yb, conf_b)
            else:
                loss = criterion(outputs, yb)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{model.name}: Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb, _ in test_loader:  # Ignore confidence during evaluation
            outputs = model(xb)
            predicted = outputs.argmax(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    
    return correct / total

def main(input, type):
    print("Loading data...")
    X_train, X_test, y_train, y_test, conf_train, conf_test = load_data(input)
    
    # Create data loaders with confidence scores
    train_ds_dnn = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(conf_train))
    test_ds_dnn = TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(conf_test))
    train_loader_dnn = DataLoader(train_ds_dnn, batch_size=32, shuffle=True)
    test_loader_dnn = DataLoader(test_ds_dnn, batch_size=32)
    
    train_ds_cnn = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train), torch.tensor(conf_train))
    test_ds_cnn = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test), torch.tensor(conf_test))
    train_loader_cnn = DataLoader(train_ds_cnn, batch_size=32, shuffle=True)
    test_loader_cnn = DataLoader(test_ds_cnn, batch_size=32)

    dnn_model = SimpleDNN(1536, len(np.unique(y_train)))
    cnn_model = SimpleCNN(len(np.unique(y_train)))
    
    # Use confidence-weighted loss
    criterion = ConfidenceWeightedLoss()
    
    print("Training DNN"+ str(type) +"...")
    dnn_acc = train_and_evaluate(dnn_model, train_loader_dnn, test_loader_dnn, criterion, optim.Adam(dnn_model.parameters()))
    print("Training CNN"+ str(type) +"...")
    cnn_acc = train_and_evaluate(cnn_model, train_loader_cnn, test_loader_cnn, criterion, optim.Adam(cnn_model.parameters()))

    print(f"DNN Accuracy: {dnn_acc:.3f}, CNN Accuracy: {cnn_acc:.3f}")
    print("Saving models...")
    torch.save(dnn_model.state_dict(), 'models/dnn_'+ str(type) + 'model.pth')
    torch.save(cnn_model.state_dict(), 'models/cnn_'+ str(type) + 'model.pth')
    #print("Models saved as dnn_model.pth' and 'cnn_model.pth'")

if __name__ == "__main__":
    #main("../preprocessing_clustering/patches_with_Majority_Cluster.csv","cluster")
    main("../preprocessing_pathology/patches_with_majority_pathology.csv","pathology")