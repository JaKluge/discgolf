import numpy as np
import pandas as pd
import os
from anomaly_detection import anomaly_detection

WINDOW_SIZE = 25
CUTTING_METHOD = "window_threshold"


def cut_throws(centers: np.array, game: pd.DataFrame):
    throws = []
    if CUTTING_METHOD == "window":
        for center in centers:
            start_time = max(0, center - WINDOW_SIZE)
            end_time = min(len(game), center + WINDOW_SIZE)
            throw = game.iloc[start_time:end_time]
            throws.append(throw)
    elif CUTTING_METHOD == "window_threshold":
        threshold = 15
        for center in centers:
            start_time_new = None
            end_time_new = None
            start_time = max(0, center - WINDOW_SIZE)
            end_time = min(len(game), center + WINDOW_SIZE)
            throw = game.iloc[start_time:end_time]
            # cut out only middle part where we exceeded the threshold first
            for i in range(start_time, end_time):
                if game["Acc_Vector"].iloc[i] > threshold and start_time_new is None:
                    start_time_new = i
                if (
                    game["Acc_Vector"].iloc[end_time - (i - start_time) - 1] > threshold
                    and end_time_new is None
                ):
                    end_time_new = end_time - (i - start_time) - 1
            throw = game.iloc[start_time_new:end_time_new]
            throws.append(throw)

    return throws


def pipeline(foldername: str):
    for file in os.listdir(os.path.join(os.getcwd(), foldername)):
        if file.endswith(".csv"):
            game = pd.read_csv(
                os.path.join(os.getcwd(), foldername, file),
                skiprows=11,
            )

    cluster_means, labels = anomaly_detection(game, foldername)
    if len(labels) == len(cluster_means):
        cut_throws(cluster_means, game)
    else:
        print("Throws were not identifyed correctly")

    # use classification now on individual throws
    print("Classifiy throws")


if __name__ == "__main__":
    foldername = "data/20240604/20240604_184206_JZ-BH-BH-PT_inside"
    pipeline(foldername)
