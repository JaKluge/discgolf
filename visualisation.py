import os
import pandas as pd
import matplotlib.pyplot as plt

def visualise_all(path, name, feature):
    directory = os.path.join(path, name)
    time = None

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            if time is None:
                time = df.SampleTimeFine
            plt.plot(time, df[feature], linewidth=1) 

    plt.title(name)
    plt.xlabel('Time')
    plt.ylabel(feature)  
    plt.show()

if __name__ == "__main__":
    input_path = 'data/20240430_splitted'
    feature = 'Acc_Vector'
    names = ['Jannie', 'Julian', 'Kevin', 'Forehand']
    for name in names:
        visualise_all(input_path, name, feature)
