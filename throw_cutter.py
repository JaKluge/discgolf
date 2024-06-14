import numpy as np
import pandas as pd
from dtw import align_time_series

WINDOW_SIZE = 20


def cut_throws(
    centers: np.array, game: pd.DataFrame, references: np.array, method="window"
):
    throws = []
    if method == "window":
        for center in centers:
            start_time = max(0, center - WINDOW_SIZE)
            end_time = min(len(game), center + WINDOW_SIZE)
            throw = game.iloc[start_time:end_time]
            throws.append(throw)

    elif method == "window_threshold":
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

    elif method == "dtw":
        if all(x is None for x in references):
            throws = cut_throws(centers, game, references)
        else:
            for i, center in enumerate(centers):
                start_time = max(0, center - WINDOW_SIZE)
                end_time = min(len(game), center + WINDOW_SIZE)
                throw = game.iloc[start_time:end_time]
                aligned_center = align_time_series(references[i], throw)
                aligned_center = range(start_time, end_time)[aligned_center]
                start_time = max(0, aligned_center - WINDOW_SIZE)
                end_time = min(len(game), aligned_center + WINDOW_SIZE)
                throw = game.iloc[start_time:end_time]
                throws.append(throw)

    return throws
