import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import cycle
from classification import read_csv_throws, create_ts_df, feature_extraction


def create_df(list_throws):
    df = []
    # col_names = ['SampleTimeFine', 'Acc_Vector', 'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'Euler_X', 'Euler_Y', 'Euler_Z']
    col_names = [
        "SampleTimeFine",
        "FreeAcc_X",
        "FreeAcc_Y",
        "FreeAcc_Z",
        "Euler_X",
        "Euler_Y",
        "Euler_Z",
        "Acc_Vector",
    ]
    for throw in list_throws:
        ts = throw[col_names]
        if "SampleTimeFine" in col_names:
            timestamps = pd.to_datetime(ts["SampleTimeFine"], unit="us")
            time_diffs = timestamps - timestamps[0]
            time_diffs_microseconds = (
                time_diffs.dt.microseconds + time_diffs.dt.seconds * 10**6
            )
            ts["SampleTimeFine"] = time_diffs_microseconds
        df.append(ts)

    return df


def tsne(X, y):
    scaler = StandardScaler()
    X_normalised = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(X_normalised)

    unique_labels = np.unique(y)
    colours = cycle(["b", "g", "r"])
    plt.figure(figsize=(8, 6))
    for name, colour in zip(unique_labels, colours):
        mask = np.array(y) == name
        plt.scatter(
            tsne_data[mask, 0], tsne_data[mask, 1], alpha=0.5, label=name, color=colour
        )
    plt.title("t-SNE Visualisation of Metadata", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X = read_csv_throws()
    throw_features = feature_extraction(X)
    # X = [throw for throw in X if throw["Label"].iloc[0] != "PT"]
    y = [throw["Label"].iloc[0] for throw in X]

    df = create_df(X)
    print(throw_features)

    tsne(throw_features, y)
