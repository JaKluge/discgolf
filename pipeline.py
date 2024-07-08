import numpy as np
import pandas as pd
import os
from anomaly_detection import anomaly_detection
from throw_cutter import cut_throws

CUTTING_METHOD = "window"


def pipeline(foldername: str):
    for file in os.listdir(os.path.join(os.getcwd(), foldername)):
        if file.endswith(".csv"):
            game = pd.read_csv(
                os.path.join(os.getcwd(), foldername, file),
                skiprows=11,
            )

    cluster_means, labels = anomaly_detection(game, foldername)
    if len(labels) == len(cluster_means):
        cut_throws(cluster_means, game, None, method=CUTTING_METHOD)
    else:
        print("Throws were not identifyed correctly")

    # use classification now on individual throws
    print("Classifiy throws")


if __name__ == "__main__":
    foldername = "data/20240604/20240604_184206_JZ-BH-BH-PT_inside"
    pipeline(foldername)
