import numpy as np
import pandas as pd
import os
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from anomaly_detection import anomaly_detection
from pipeline import cut_throws


def collect_data(path):
    games = []
    forehand_throws = []
    backhand_throws = []
    putt_throws = []
    foldernames = []

    # sort subfolders by date
    subfolders = os.listdir(os.path.join(os.getcwd(), path))
    subfolders.sort(key=lambda x: x.split("_")[1])

    # get list of df for all measurements in defined folder
    for subfolder in subfolders:
        if not subfolder.startswith("2024"):
            continue
        foldernames.append(subfolder)
        for file in os.listdir(os.path.join(os.getcwd(), "data/20240604", subfolder)):
            if file.endswith(".csv"):
                games.append(
                    pd.read_csv(
                        os.path.join(os.getcwd(), "data/20240604", subfolder, file),
                        skiprows=11,
                    )
                )

    for game_idx, game in enumerate(games):
        cluster_means, labels = anomaly_detection(game, foldernames[game_idx])
        if len(labels) == len(cluster_means):
            throws = cut_throws(cluster_means, game)
            for throw_idx, throw in enumerate(throws):
                if labels[throw_idx] == "BH":
                    backhand_throws.append(throw)
                elif labels[throw_idx] == "FH":
                    forehand_throws.append(throw)
                elif labels[throw_idx] == "PT":
                    putt_throws.append(throw)
        else:
            print("Throws were not identifyed correctly\n")

    print(f"Number of identifyed forhand throws: {len(forehand_throws)}")
    print(f"Number of identifyed backhand throws: {len(backhand_throws)}")
    print(f"Number of identifyed putt throws: {len(putt_throws)}")

    return [backhand_throws, forehand_throws, putt_throws]


# this takes an array of arrays of dataframes correspnding to backhand, forehand and putts respectively
def feature_extraction(throw_sets: np.array):
    feature_settings = settings.MinimalFCParameters()
    # feature_settings = settings.EfficientFCParameters()

    throw_sets_concat = []
    for throws in throw_sets:
        data = []
        df_concat = pd.DataFrame(data)
        for idx, throw in enumerate(throws):
            throw.insert(0, "id", idx)
            df_concat = pd.concat([df_concat, throw])
        throw_sets_concat.append(df_concat)

    feature_sets = []
    for set in throw_sets_concat:
        throw_features = extract_features(
            set[
                [
                    "id",
                    "Euler_X",
                    "Euler_Y",
                    "Euler_Z",
                    "FreeAccMagnitude",
                    "FreeAcc_X",
                    "FreeAcc_Y",
                    "FreeAcc_Z",
                ]
            ],
            column_id="id",
            default_fc_parameters=feature_settings,
        )
        feature_sets.append(throw_features)
    return feature_sets


if __name__ == "__main__":
    path = "data/20240604"
    throw_sets = collect_data(path)
    feature_sets = feature_extraction(throw_sets)
