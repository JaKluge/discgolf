import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import optuna

from classification import feature_extraction, read_csv_throws, create_ts_df

PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"
torch.manual_seed(42)


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, learning_rate=1e-3
    ):
        super(LSTMClassifier, self).__init__()
        self.name = "LSTM"
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(
            self.device
        )
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(
            self.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CNNClassifier(pl.LightningModule):
    def __init__(self, input_channels, output_dim, learning_rate=1e-3):
        super(CNNClassifier, self).__init__()
        self.name = "CNN"
        self.conv1 = nn.Conv2d(
            input_channels, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # transform into (num_samples, length time series, num_features)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def prepare_data(
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, batch_size=32
):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    X_train = torch.tensor(X_train_raw, dtype=torch.float32)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    X_train = X_train.view(X_train.size(0), -1, 7)
    X_test = X_test.view(X_test.size(0), -1, 7)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, le


def train_lstm(train_loader):
    input_dim = 7
    hidden_dim = 64
    output_dim = len(set(y_train_raw))
    n_layers = 3
    n_epochs = 10

    lstm = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers)
    trainer = Trainer(max_epochs=n_epochs)
    trainer.fit(lstm, train_loader)
    return lstm


def train_cnn(train_loader):
    input_channels = 1
    output_dim = len(set(y_train_raw))
    n_epochs = 10

    cnn = CNNClassifier(input_channels, output_dim)
    trainer = Trainer(max_epochs=n_epochs)
    trainer.fit(cnn, train_loader)
    return cnn


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            y_hat = torch.argmax(y_hat, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(y_hat.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy of {model.name}: {accuracy:.4f}\n")
    return accuracy


if __name__ == "__main__":

    throws_list = read_csv_throws()

    labels = [throw["Label"].iloc[0] for throw in throws_list]

    throw_features = feature_extraction(throws_list)

    throw_df = create_ts_df(
        throws_list,
        [
            "Euler_X",
            "Euler_Y",
            "Euler_Z",
            "Acc_Vector",
            "FreeAcc_X",
            "FreeAcc_Y",
            "FreeAcc_Z",
        ],
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        throw_df,
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42,
    )

    batch_size = 16
    train_loader, test_loader, label_encoder = prepare_data(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, batch_size
    )

    # # create optuna study
    # study = optuna.create_study(direction='maximize')
    # study.optimize(classify_lstm, n_trials=100)

    # classify using LSTM neural network
    lstm = train_lstm(train_loader)
    evaluate_model(lstm, test_loader)

    # classify using CNN neural network
    # cnn = train_cnn(train_loader)
    # evaluate_model(cnn, test_loader)
