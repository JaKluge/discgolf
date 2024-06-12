import numpy as np
import pandas as pd
import os
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from anomaly_detection import anomaly_detection
from pipeline import cut_throws
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn import svm
from sklearn.model_selection import cross_val_score


def create_df_old_data(path, name):
    dfs = []
    directory = os.path.join(path, name)
    # iterate over all throws in the directory and compute maximum or variance for all features; add result as one row to the dataframe
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)
    throw_features = feature_extraction(dfs)
    return throw_features


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


# this takes an array of dataframes
def feature_extraction(throw_set: np.array):
    feature_settings = settings.MinimalFCParameters()
    # feature_settings = settings.EfficientFCParameters()

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

    # print(cross_val_score(clf, X_train, y_train, cv=5, scoring="recall_macro"))
    print(clf.predict(X_test))
    print(y_test)


if __name__ == "__main__":
    # path = "data/20240604"
    # throw_sets = collect_data(path)
    # feature_sets = []
    # for throw_set in throw_sets:
    #   throw_features = feature_extraction(throw_set)
    #  feature_sets.append(throw_features)

    path = "data/20240430_splitted"
    df_backhand = create_df_old_data(path, "Julian")
    df_forehand = create_df_old_data(path, "Forehand")

    df_backhand.insert(len(df_backhand.columns), "Label", "BH")
    df_forehand.insert(len(df_forehand.columns), "Label", "FH")

    df_concat = pd.concat([df_backhand, df_forehand])
    X = df_concat.loc[:, df_concat.columns != "Label"]
    y = df_concat["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    classify(X_train, X_test, y_train, y_test)
