import numpy as np
import pandas as pd
import glob
import os
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"


# this takes a list of dataframes/throws and extracts features for every throw
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
    clf.fit(X_train, y_train)

    print(
        "\nAccuracy of cross validation: "
        + str(cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy"))
    )
    print("Predicted labels: " + str(clf.predict(X_test)))
    print("True labels: " + str(y_test))


def read_csv_throws():
    csv_files = glob.glob(os.path.join(DF_DIR, "*.csv"))
    throws_list = [pd.read_csv(file) for file in csv_files]
    return throws_list


if __name__ == "__main__":

    throws_list = read_csv_throws()
    labels = [throw["Label"].iloc[0] for throw in throws_list]

    throw_features = feature_extraction(throws_list)

    X_train, X_test, y_train, y_test = train_test_split(
        throw_features,
        labels,
        stratify=labels,
        test_size=0.2,
    )
    classify(X_train, X_test, y_train, y_test)
