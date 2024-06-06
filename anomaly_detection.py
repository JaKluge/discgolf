import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


PLOT_DIR = "figures/anomaly_detection"
METHOD = "isolation_forest"

dfs = []
foldernames = []


def determine_num_throws_from_filename(filename: str):
    description = filename.split("_")[2]
    return (
        description.count("BH")
        + description.count("FH")
        + description.count("PT")
    )


def create_directories(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_acceleration(df: pd.DataFrame, df_idx: int):
    df["FreeAccMagnitude"] = (
        df["FreeAcc_X"] ** 2 + df["FreeAcc_Y"] ** 2 + df["FreeAcc_Z"] ** 2
    ) ** 0.5
    df["FreeAccMagnitude"].plot(figsize=(8, 4), c="black")
    plt.title(foldernames[df_idx].split("_")[2])
    create_directories(
        path=os.path.join(
            PLOT_DIR, str(foldernames[df_idx]).replace(".csv", "")
        )
    )
    plt.close()


def get_anomalies(df: pd.DataFrame, contamination: float, method: str):
    if method == "isolation_forest":
        clf = IsolationForest(contamination=contamination)
        clf.fit(df[["FreeAccMagnitude"]])
        return clf.predict(df[["FreeAccMagnitude"]])
    if method == "lof":
        lof = LocalOutlierFactor(n_neighbors=50)
        pred = lof.fit_predict(df[["FreeAccMagnitude"]])
        return pred


def plot_anomalies(df: pd.DataFrame, df_idx: int, name_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(8, 4))
    anomalies = df.loc[df["Anomaly"] == -1, ["FreeAccMagnitude"]]  # Anomaly
    ax.plot(df.index, df["FreeAccMagnitude"], color="black", label="Normal")
    ax.scatter(
        anomalies.index,
        anomalies["FreeAccMagnitude"],
        color="red",
        label="Anomaly",
    )
    plt.title(foldernames[df_idx].split("_")[2])
    plt.legend()
    plt.savefig(
        fname=os.path.join(
            PLOT_DIR,
            str(foldernames[df_idx]).replace(".csv", ""),
            "anomalies"+name_suffix+".png",
        )
    )
    plt.close()


def determine_num_clusters(
    anomalies: np.array,
    df_idx: int,
    num_max_cluster: int,
    plot_knee: bool = False,
):
    wcss = []
    for i in range(2, min(11, num_max_cluster)):
        kmeans = KMeans(
            n_clusters=i,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=0,
        )
        kmeans.fit(anomalies)
        wcss.append(kmeans.inertia_)

    if plot_knee:
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, min(11, num_max_cluster)), wcss)
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.savefig(
            fname=os.path.join(
                PLOT_DIR,
                str(foldernames[df_idx]).replace(".csv", ""),
                "knee.png",
            )
        )
        plt.close()

    kl = KneeLocator(
        range(2, min(11, num_max_cluster)),
        wcss,
        curve="convex",
        direction="decreasing",
        S=5,
    )
    if kl.elbow is None:
        print("Warning: No elbow detected. Opting for default = 2.")
        return 2
    else:
        return kl.elbow


def get_cluster_means(n_clusters: int, anomalies: np.array):
    # k means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(anomalies)
    cluster_representatives = (
        kmeans.cluster_centers_.round()
        .astype(int)
        .reshape(
            -1,
        )
    )
    return cluster_representatives


def plot_anomaly_groups(df: pd.DataFrame, df_idx: int):
    # plot acceleration
    fig, ax = plt.subplots(figsize=(8, 4))
    anomalies = df.loc[df["AnomalyGroup"] == 1, ["FreeAccMagnitude"]]  # Anomaly
    ax.plot(df.index, df["FreeAccMagnitude"], color="black", label="Normal")

    # plot intervals to indicate anomaly groups
    for index, _ in anomalies.iterrows():
        start_index = max(0, index - 100)  # Start index of the color interval
        end_index = min(
            len(df.index) - 1, index + 100
        )  # End index of the color interval
        ax.axvspan(
            df.index[start_index],
            df.index[end_index],
            color="orange",
            alpha=0.5,
        )

    # add legend for the color intervals
    ax.plot(
        [], [], color="orange", alpha=0.5, label="Anomaly Interval", linewidth=3
    )
    plt.title(foldernames[df_idx].split("_")[2])

    # # add pointers to the anomalies indicating that first intervals shows backhand, seocond interval shows forehand and last a putt
    # ax.annotate('Backhand', xy=(anomalies.index[0], anomalies['FreeAccMagnitude'].values[0]), xytext=(anomalies.index[0] + 100, anomalies['FreeAccMagnitude'].values[0] + 0.5),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'))
    # ax.annotate('Forehand', xy=(anomalies.index[1], anomalies['FreeAccMagnitude'].values[1]), xytext=(anomalies.index[1] + 100, anomalies['FreeAccMagnitude'].values[1] + 0.5),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'))
    # ax.annotate('Putt', xy=(anomalies.index[2], anomalies['FreeAccMagnitude'].values[2]), xytext=(anomalies.index[2] + 100, anomalies['FreeAccMagnitude'].values[2] + 0.5),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.legend()
    plt.savefig(
        fname=os.path.join(
            PLOT_DIR,
            str(foldernames[df_idx]).replace(".csv", ""),
            "anomalies_grouped.png",
        )
    )
    plt.close()


if __name__ == "__main__":

    # sort subfolders by date
    subfolders = os.listdir(os.path.join(os.getcwd(), "data/20240604"))
    subfolders.sort(key=lambda x: x.split("_")[1])

    for subfolder in subfolders:
        foldernames.append(subfolder)
        for file in os.listdir(
            os.path.join(os.getcwd(), "data/20240604", subfolder)
        ):
            if file.endswith(".csv"):
                dfs.append(
                    pd.read_csv(
                        os.path.join(
                            os.getcwd(), "data/20240604", subfolder, file
                        ),
                        skiprows=11,
                    )
                )

    for df_idx, df in enumerate(dfs):
        print("{foldername}:".format(foldername=foldernames[df_idx]))
        num_throws = determine_num_throws_from_filename(
            filename=foldernames[df_idx]
        )
        print("Number of throws: ", num_throws)

        # plot acceleration for inspection
        plot_acceleration(df, df_idx)

        # determine anomalies using Isolation Forest
        df["Anomaly"] = get_anomalies(df, contamination=0.01, method=METHOD)
        # print(df["Anomaly"].value_counts())
        plot_anomalies(df, df_idx, name_suffix="_raw")

        # get list of timestamps of anomalies
        anomalies = np.array(
            df.loc[df["Anomaly"] == -1].index.tolist()
        ).reshape(-1, 1)

        anomalies = anomalies.squeeze()
        # filter anomaly indices for those were at least 50 consecutive data points are included
        delete_indices = []
        for idx, value in enumerate(anomalies.tolist()[:-10]):
            current_range = anomalies[idx:idx+10]
            # print(current_range)
            ideal_range = np.array(range(anomalies[idx], anomalies[idx]+10))
            # print(ideal_range)
            if not np.equal(current_range, ideal_range).all():
                delete_indices.append(idx)
        anomalies = np.delete(arr=anomalies, obj=delete_indices, axis=0)

        df["Anomaly"] = 1
        df.loc[anomalies.tolist(), "Anomaly"] = -1

        plot_anomalies(df, df_idx, name_suffix="_filtered")

        anomalies = anomalies.reshape(-1, 1)


        # determine number of clusters
        n_clusters = determine_num_clusters(
            anomalies=anomalies,
            num_max_cluster=len(anomalies),
            plot_knee=True,
            df_idx=df_idx,
        )
        print("Predicted numbers of throws", n_clusters, "\n")

        # get cluster means
        cluster_means = get_cluster_means(n_clusters, anomalies)

        # mark cluster means in df
        df["AnomalyGroup"] = 0
        df.loc[cluster_means, "AnomalyGroup"] = 1

        # plot anomalies groups
        plot_anomaly_groups(df=df, df_idx=df_idx)
