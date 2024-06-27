import numpy as np
import pandas as pd
import glob
import os
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sktime.classification.ensemble import BaggingClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_unit_test
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.shapelet_based import ShapeletTransformClassifier

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# from sktime.classification.ensemble import TimeSeriesForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"
SEED = np.random.RandomState(5)


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


def create_ts_df(throw_list, columns):
    n_samples = len(throw_list)
    n_timestamps = throw_list[0].shape[0]
    n_features = len(columns)

    ts_data = np.empty((n_samples, n_timestamps, n_features))
    for i, throw in enumerate(throw_list):
        ts_data[i] = throw[columns].values

    return ts_data


def classify_ts(X_train, X_test, y_train, y_test):
    # scaler = TimeSeriesScalerMeanVariance()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    clf = BaggingClassifier(
        RocketClassifier(num_kernels=100),
        n_estimators=10,
    )
    clf.fit(X_train, y_train)
    print("Predicted labels: " + str(clf.predict(X_test)))
    print("True labels: " + str(y_test))

    # knn = KNeighborsTimeSeriesClassifier(n_neighbors=2)
    # p_grid = {"n_neighbors": [1, 5]}
    # cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    # clf = GridSearchCV(estimator=knn, param_grid=p_grid, cv=cv)

    # cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    # print("\nAccuracy of cross-validation (Time Series): ", cv_scores)
    # print("Mean cross-validation accuracy: ", np.mean(cv_scores))

    # clf.fit(X_train, y_train)
    # print("Predicted labels: " + str(clf.predict(X_test)))
    # print("True labels: " + str(y_test))


def evaluate_classifiers(X, y):
    classifiers = {
        "Rocket + Bagging": BaggingClassifier(
            RocketClassifier(num_kernels=100), n_estimators=10
        ),
        "TimeSeriesForest": TimeSeriesForestClassifier(n_estimators=100),
        # Shapelet Transform": ShapeletTransformClassifier(),
    }

    results = []

    for name, clf in classifiers.items():
        print(name)
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        accuracy = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")
        precision = cross_val_score(clf, X, y, cv=kf, scoring="precision_weighted")
        recall = cross_val_score(clf, X, y, cv=kf, scoring="recall_weighted")
        f1 = cross_val_score(clf, X, y, cv=kf, scoring="f1_weighted")

        results.append(
            {
                "Classifier": name,
                "Accuracy": np.mean(accuracy),
                "Precision": np.mean(precision),
                "Recall": np.mean(recall),
                "F1 Score": np.mean(f1),
            }
        )

    return pd.DataFrame(results)


def classify_features(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=SEED)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")

    print("\nAccuracy of cross-validation (Features): ", cv_scores)
    print("Mean cross-validation accuracy: ", np.mean(cv_scores))

    clf.fit(X_train, y_train)
    print("Predicted labels: " + str(clf.predict(X_test)))
    print("True labels: " + str(y_test))


def read_csv_throws():
    csv_files = glob.glob(os.path.join(DF_DIR, "*.csv"))
    throws_list = [pd.read_csv(file) for file in csv_files]
    return throws_list


if __name__ == "__main__":

    throws_list = read_csv_throws()
    labels = [throw["Label"].iloc[0] for throw in throws_list]

    # throw_features = feature_extraction(throws_list)

    # X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(
    #     throw_features,
    #     labels,
    #     stratify=labels,
    #     test_size=0.2,
    #     random_state=SEED,
    # )

    # classify_features(X_train_feat, X_test_feat, y_train_feat, y_test_feat)

    throw_df = create_ts_df(
        throws_list,
        [
            "Euler_X",
            "Euler_Y",
            "Euler_Z",
            "Acc_Vector",
            "FreeAcc_X",
            "FreeAcc_Y",
            "FreeAcc_Z",
        ],
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        throw_df,
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=SEED,
    )

    # classify_ts(
    #     np.array(X_train_raw),
    #     np.array(X_test_raw),
    #     np.array(y_train_raw),
    #     np.array(y_test_raw),
    # )

    df = evaluate_classifiers(np.array(X_train_raw), np.array(y_train_raw))
    print(df)
