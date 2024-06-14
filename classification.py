import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from anomaly_detection import create_directories
from anomaly_detection import anomaly_detection
from pipeline import cut_throws
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

PLOT_DIR = "figures/anomaly_detection"


def create_df_old_data(path, name):
    dfs = []
    directory = os.path.join(path, name)
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)
    # extract features
    throw_features = feature_extraction(dfs)
    return throw_features


def vis_of_throw(throw: pd.DataFrame, foldername: str, idx: int):
    plt.figure(figsize=(8, 4))
    plt.plot(throw["Acc_Vector"], c="black")

    title_part = (
        foldername.split("_")[2] if len(foldername.split("_")) > 2 else foldername
    )
    plt.title(f"{title_part}_{idx}")
    plt.xlabel("Acc_Vector")
    plt.ylabel("Time")

    plot_dir = os.path.join(PLOT_DIR, str(foldername).replace(".csv", ""))
    create_directories(plot_dir)

    plot_path = os.path.join(plot_dir, f"throw_{idx}.png")
    plt.savefig(fname=plot_path)
    plt.close()


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
        for file in os.listdir(os.path.join(os.getcwd(), path, subfolder)):
            if file.endswith(".csv"):
                games.append(
                    pd.read_csv(
                        os.path.join(os.getcwd(), path, subfolder, file),
                        skiprows=11,
                    )
                )

    for game_idx, game in enumerate(games):
        # delete all images of individual throws
        for idx in range(3):
            path = os.path.join(
                os.path.join(PLOT_DIR, str(foldernames[game_idx]).replace(".csv", "")),
                f"throw_{idx}.png",
            )
            if os.path.isfile(path):
                os.remove(path)

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
                vis_of_throw(throw, foldernames[game_idx], throw_idx)
        else:
            print("Throws were not identifyed correctly\n")

    print(f"Number of identifyed forhand throws: {len(forehand_throws)}")
    print(f"Number of identifyed backhand throws: {len(backhand_throws)}")
    print(f"Number of identifyed putt throws: {len(putt_throws)}")

    return [backhand_throws, forehand_throws, putt_throws]


# this takes an array of dataframes (backhand, forehand, putt) and extracts features for every throw
def feature_extraction(throw_set: np.array):
    feature_settings = settings.MinimalFCParameters()
    # feature_settings = {"mean": None, "standard_deviation": None, "length": None}

    data = []
    df_concat = pd.DataFrame(data)
    for idx, throw in enumerate(throw_set):
        throw.insert(0, "id", idx)
        df_concat = pd.concat([df_concat, throw])

    throw_features = extract_features(
        df_concat[
            [
                "id",
                "Euler_X",
                "Euler_Y",
                "Euler_Z",
                "Acc_Vector",
                "FreeAcc_X",
                "FreeAcc_Y",
                "FreeAcc_Z",
            ]
        ],
        column_id="id",
        default_fc_parameters=feature_settings,
    )
    return throw_features


def classify(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)

    print(cross_val_score(clf, X_train, y_train, cv=5, scoring="recall_macro"))
    print("predicted labels: " + str(clf.predict(X_test)))
    print("true labels: " + str(y_test))


if __name__ == "__main__":
    # get throws from games and extract features
    path = "data/20240612"
    throw_sets = collect_data(path)
    feature_sets = []
    for throw_set in throw_sets:
        throw_features = feature_extraction(throw_set)
        feature_sets.append(throw_features)
    # combine FH and BH throws (not PT yet)
    df_train = pd.concat(feature_sets[0:2])
    y_train = np.concatenate(
        (
            ["BH"] * len(feature_sets[0]),
            ["FH"] * len(feature_sets[1]),
        ),
        axis=None,
    )

    # take old backhand and forehand throws by julian and create features
    path = "data/20240430_splitted"
    df_backhand = create_df_old_data(path, "Julian")
    df_forehand = create_df_old_data(path, "Forehand")

    df_backhand.insert(len(df_backhand.columns), "Label", "BH")
    df_forehand.insert(len(df_forehand.columns), "Label", "FH")

    df_concat = pd.concat([df_backhand, df_forehand])

    # Version 1: take all data from julian for training and testing
    X = df_concat.loc[:, df_concat.columns != "Label"]
    y = df_concat["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # Version 2: train and test on extracted throws
    X_train, X_test, y_train, y_test = train_test_split(
        df_train, y_train, stratify=y_train, test_size=0.2
    )

    classify(X_train, X_test, y_train, y_test)
