import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from sktime.classification.ensemble import BaggingClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"
SEED = np.random.RandomState(5)


# this takes a list of dataframes/throws and extracts features for every throw
def feature_extraction(throw_set: np.array):
    feature_settings = settings.MinimalFCParameters()
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
        if ts_data[i].shape[0] == throw[columns].values.shape[0]:
            ts_data[i] = throw[columns].values
    return ts_data


def classify_ts(X, y):
    classifiers = {
        "kNN": KNeighborsTimeSeriesClassifier(n_neighbors=2),
        "Rocket + Bagging": BaggingClassifier(
            RocketClassifier(num_kernels=100), n_estimators=10
        ),
        "Shapelet Transform": ShapeletTransformClassifier(),
    }

    results = []

    for name, clf in classifiers.items():
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


def classify_features(X_train, y_train):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Naive Bayes",
    ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        GaussianNB(),
    ]
    results = []
    for name, clf in zip(names, classifiers):
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        accuracy = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
        precision = cross_val_score(
            clf, X_train, y_train, cv=kf, scoring="precision_weighted"
        )
        recall = cross_val_score(
            clf, X_train, y_train, cv=kf, scoring="recall_weighted"
        )
        f1 = cross_val_score(clf, X_train, y_train, cv=kf, scoring="f1_weighted")

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


def read_csv_throws():
    csv_files = glob.glob(os.path.join(DF_DIR, "*.csv"))
    throws_list = [pd.read_csv(file) for file in csv_files]
    return throws_list


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def evaluate_naive_bayes(X_train, X_test, y_train, y_test):
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)

    # Calculate and print the confusion matrix
    plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix for Naive Bayes")


if __name__ == "__main__":

    throws_list = read_csv_throws()
    labels = [throw["Label"].iloc[0] for throw in throws_list]

    throw_features = feature_extraction(throws_list)

    X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(
        throw_features,
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=SEED,
    )

    results_features = classify_features(X_train_feat, y_train_feat)
    print(results_features)

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

    evaluate_naive_bayes(X_train_feat, X_test_feat, y_train_feat, y_test_feat)
    # X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    #     throw_df,
    #     labels,
    #     stratify=labels,
    #     test_size=0.2,
    #     random_state=SEED,
    # )

    # results_ts = classify_ts(np.array(X_train_raw), np.array(y_train_raw))
    # print(results_ts)
