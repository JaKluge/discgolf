import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from statistical_analysis import create_dfs

label_mapping = {
    'FreeAcc_X': 'Acceleration on X-axis', 
    'FreeAcc_Y': 'Acceleration on Y-axis', 
    'FreeAcc_Z': 'Acceleration on Z-axis', 
    'Acc_Vector': 'Acceleration magnitude',
    'Jannie': 'Janine, backhand',
    'Julian': 'Julian, backhand',
    'Kevin': 'Kevin, backhand',
    'Forehand': 'Julian, forehand'
}

def visualise_all(path, name, feature):
    directory = os.path.join(path, name)
    time = None

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df["SampleTimeFine"] -= df["SampleTimeFine"][0]
            df["SampleTimeFine"] /= 1000
            if time is None:
                time = df.SampleTimeFine
            plt.plot(time, df[feature], linewidth=1) 

    plt.title(f"{label_mapping[feature]} of all 40 throws by {label_mapping[name]}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (m/s²)')
    plt.ylim((0, 260) if feature == 'Acc_Vector' else (-250, 250))
    #plt.show()
    return plt

def create_all_line_plots(input_path):
    features = ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'Acc_Vector']
    names = ['Jannie', 'Julian', 'Kevin', 'Forehand']
    for name, feature in itertools.product(names, features):
        plt = visualise_all(input_path, name, feature)
        plt.savefig(f"./images/line_{name}_{feature}")
        plt.close()

def visualise_maxima(path, name):
    features = ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'Acc_Vector']
    directory = os.path.join(path, name)

    plt_index = 1
    for feature in features:
        times = []
        maxima = []

        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                max_index = df.idxmax()[feature]
                times.append(df["SampleTimeFine"][max_index])
                maxima.append(df[feature][max_index])

        ax = plt.subplot(2, 2, plt_index)
        plt_index += 1
        ax.scatter(x=times, y=maxima)
        ax.set_ylim(0, 250)
        ax.set_xlabel('Time')
        ax.set_title(feature)
    plt.tight_layout()
    plt.suptitle(name, fontsize='x-large')  
    #plt.show()
    return plt

def visualise_all_maxima(input_path):
    for name in ['Jannie', 'Kevin', 'Julian', 'Forehand']:
        plt = visualise_maxima(input_path, name)
        plt.savefig(f"images/maxima_{name}.png")
        plt.close()

def visualise_gyro_variances(input_path):
    for name in ['Jannie', 'Kevin', 'Julian', 'Forehand']:
        df = create_dfs(input_path, [name])[0]
        for feature in ['Euler_X','Euler_Y','Euler_Z']:
            plt.hist(df[feature], bins=20)
            plt.title(f"{name} {feature}")
            plt.savefig(f"images/hist_{name}_{feature}.png")
            plt.close()

if __name__ == "__main__":
    input_path = 'data/20240430_splitted'
    #create_all_line_plots(input_path)
    #visualise_all_maxima(input_path)
    visualise_gyro_variances(input_path)
