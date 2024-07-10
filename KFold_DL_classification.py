import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from pytorch_lightning.loggers import WandbLogger
import wandb

from classification import feature_extraction, read_csv_throws, create_ts_df

PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"
USE_WANDB = False
torch.manual_seed(43)


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
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CNNClassifier(pl.LightningModule):
    def __init__(
        self, input_channels, input_channel_size, output_dim, learning_rate=1e-3
    ):
        super(CNNClassifier, self).__init__()
        self.name = "CNN"
        self.conv1 = nn.Conv1d(
            input_channels, 16, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, padding=0)
        self.fc1 = nn.Linear(
            int(16 * (input_channel_size / self.pool.kernel_size)), 32
        )
        self.fc2 = nn.Linear(32, output_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # transform into (num_samples, num_features, num_timesteps)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def prepare_data(X, y, batch_size=32):
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.reshape(X.shape[0], -1, 7)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, le


def train_lstm(train_loader, log_to_wandb=False):
    input_dim = 7
    hidden_dim = 64
    output_dim = len(set(train_loader.dataset.tensors[1].numpy()))
    print(len(set(train_loader.dataset.tensors[1].numpy())))
    n_layers = 3
    n_epochs = 10

    if log_to_wandb:
        run = wandb.init(reinit=True, entity="julianzabbarov")
        wandb_logger = WandbLogger(project="discgolf", log_model=False)
        lstm = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers)
        trainer = Trainer(
            max_epochs=n_epochs, logger=wandb_logger, log_every_n_steps=5
        )
        trainer.fit(lstm, train_loader)
        run.finish()
    else:
        lstm = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers)
        trainer = Trainer(max_epochs=n_epochs)
        trainer.fit(lstm, train_loader)
    return lstm


def train_cnn(train_loader, log_to_wandb=False):
    input_channels = 7
    output_dim = len(set(train_loader.dataset.tensors[1].numpy()))
    n_epochs = 10

    if log_to_wandb:
        run = wandb.init(reinit=True, entity="julianzabbarov")
        wandb_logger = WandbLogger(project="discgolf", log_model=False)
        cnn = CNNClassifier(input_channels, throws_list[0].shape[0], output_dim)
        trainer = Trainer(
            max_epochs=n_epochs, logger=wandb_logger, log_every_n_steps=5
        )
        trainer.fit(cnn, train_loader)
        run.finish()
    else:
        cnn = CNNClassifier(input_channels, throws_list[0].shape[0], output_dim)
        trainer = Trainer(max_epochs=n_epochs)
        trainer.fit(cnn, train_loader)
    return cnn


def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            y_hat = model(x)
            y_hat = torch.argmax(y_hat, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(y_hat.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, recall, precision, f1


if __name__ == "__main__":

    # read data
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

    # define cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = {
        "LSTM": {"accuracy": [], "recall": [], "precision": [], "f1": []},
        "CNN": {"accuracy": [], "recall": [], "precision": [], "f1": []},
    }

    # ensure that only training for one fold is logged to wandb
    logged_cnn = False
    logged_lstm = False

    for train_index, test_index in skf.split(throw_df, labels):
        # define train test splits
        X_train, X_test = throw_df[train_index], throw_df[test_index]
        y_train, y_test = (
            np.array(labels)[train_index],
            np.array(labels)[test_index],
        )

        # format data
        batch_size = 4
        train_loader, _ = prepare_data(X_train, y_train, batch_size)
        test_loader, _ = prepare_data(X_test, y_test, batch_size)

        # Train and evaluate LSTM
        if USE_WANDB and not logged_lstm:
            lstm_model = train_lstm(train_loader, log_to_wandb=True)
            logged_lstm = True
        else:
            lstm_model = train_lstm(train_loader, log_to_wandb=False)
        lstm_metrics = evaluate_model(lstm_model, test_loader)
        for metric, value in zip(all_metrics["LSTM"], lstm_metrics):
            all_metrics["LSTM"][metric].append(value)

        # Train and evaluate CNN
        input_channels = 7
        if USE_WANDB and not logged_cnn:
            cnn_model = train_cnn(train_loader, log_to_wandb=True)
            logged_cnn = True
        else:
            cnn_model = train_cnn(train_loader, log_to_wandb=False)
        cnn_metrics = evaluate_model(cnn_model, test_loader)
        for metric, value in zip(all_metrics["CNN"], cnn_metrics):
            all_metrics["CNN"][metric].append(value)

    # aggregate metrics over folds
    for model_name in all_metrics:
        print(f"\n{model_name} model average metrics:")
        for metric in all_metrics[model_name]:
            avg_metric = np.mean(all_metrics[model_name][metric])
            print(f"{metric.capitalize()}: {avg_metric:.4f}")
