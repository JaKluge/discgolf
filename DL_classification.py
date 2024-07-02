import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from classification import feature_extraction, read_csv_throws, create_ts_df

PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"
SEED = np.random.RandomState(0)


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, learning_rate=1e-3
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        print(x.size())
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
        print(x.size())
        print(y.size())
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

    # print(type(X_train_raw))
    # print(X_train_raw.shape)
    # print(type(y_train_raw))
    # print(np.array(y_train_raw).shape)

    X_train = torch.tensor(X_train_raw, dtype=torch.float32)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    # Reshape X_train and X_test to match LSTM input expectations (batch_size, sequence_length, input_dim)
    X_train = X_train.view(X_train.size(0), -1, 7)
    X_test = X_test.view(X_test.size(0), -1, 7)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, le


def classify_lstm(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    input_dim = 7
    hidden_dim = 64
    output_dim = len(set(y_train_raw))
    n_layers = 2
    batch_size = 32
    n_epochs = 10

    train_loader, test_loader, label_encoder = prepare_data(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, batch_size
    )

    model = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers)
    trainer = Trainer(max_epochs=n_epochs)
    trainer.fit(model, train_loader, test_loader)

    # Evaluation
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
    print(f"Test Accuracy: {accuracy:.4f}")


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
        random_state=SEED,
    )

    # classify using LSTM neural network
    classify_lstm(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
