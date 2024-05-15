import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import cycle

def create_combined_df(path, names):
    combined_dataframes = []
    col_names = ['SampleTimeFine', 'Acc_Vector', 'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'Euler_X', 'Euler_Y', 'Euler_Z']
    #col_names = ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']
    for name in names:
        directory = os.path.join(path, name)
        dfs = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                ts = pd.read_csv(filepath)
                ts = ts[col_names]
                ts['name'] = name
                dfs.append(ts)
        combined_df = pd.concat(dfs, axis=0)
        combined_dataframes.append(combined_df)

    return combined_dataframes

def tsne(combined_dataframes, names):
    combined_data = pd.concat(combined_dataframes, axis=0)
    X = combined_data.drop(columns=['name'])
    y = combined_data['name']

    scaler = StandardScaler()
    X_normalised = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(X_normalised)

    colours = cycle(['b', 'g', 'r', 'c'])
    plt.figure(figsize=(8, 6))
    for name, colour in zip(names, colours):
        mask = (y == name)
        plt.scatter(tsne_data[mask, 0], tsne_data[mask, 1], alpha=0.5, label=name, color=colour)
    plt.title('t-SNE Visualisation')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    input_path = 'data/20240430_splitted'
    names = ['Jannie', 'Julian', 'Kevin', 'Forehand']
    combined_dataframes = create_combined_df(input_path, names)
    tsne(combined_dataframes, names)
