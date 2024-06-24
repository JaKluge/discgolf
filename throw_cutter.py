import numpy as np
import pandas as pd
from dtaidistance import dtw

WINDOW_SIZE = 20


# this returns the index of the position in the throw that aligns with the center of the reference
def align_time_series(time_series_1, time_series_2):
    reference = np.array(time_series_1["Acc_Vector"])
    throw = np.array(time_series_2["Acc_Vector"])
    # Calculate the DTW path
    path = dtw.warping_path(reference, throw)

    # Extract the aligned time series based on the DTW path
    aligned_series_1_indices = [i for i, _ in path]
    aligned_series_2_indices = [j for _, j in path]

    # Find the index in the throw that aligns with the reference center
    reference_center_index = len(reference) // 2
    aligned_index_in_throw = aligned_series_2_indices[
        aligned_series_1_indices.index(reference_center_index)
    ]

    return aligned_index_in_throw


def cut_throws(
    center: int,
    game: pd.DataFrame,
    label: str,
    references: dict,
    method="window",
):
    if method == "window":
        start_time = max(0, center - WINDOW_SIZE)
        end_time = min(len(game), center + WINDOW_SIZE)
        throw = game.iloc[start_time:end_time]

    elif method == "dtw":
        if references[label] is None:
            throw = cut_throws(center, game, label, references)
        else:
            start_time = max(0, center - WINDOW_SIZE)
            end_time = min(len(game), center + WINDOW_SIZE)
            throw = game.iloc[start_time:end_time]
            aligned_center_rel = align_time_series(references[label], throw)
            aligned_center = range(start_time, end_time)[aligned_center_rel]
            start_time = max(0, aligned_center - WINDOW_SIZE)
            end_time = min(len(game), aligned_center + WINDOW_SIZE)
            throw = game.iloc[start_time:end_time]

    return throw
