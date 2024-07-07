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
SEED = np.random.RandomState(0)


def determine_throws_from_filename(filename: str):
    description = filename.split("_")[2]
    labels = description.split("-")[1:]
    return (
        description.count("BH") + description.count("FH") + description.count("PT"),
        labels,
    )


def create_directories(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_acceleration(df: pd.DataFrame, foldername: str):
    df["Acc_Vector"].plot(figsize=(8, 4), c="black")
    plt.title(foldername.split("_")[2])
    create_directories(path=os.path.join(PLOT_DIR, str(foldername).replace(".csv", "")))
    plt.close()


def get_anomalies(df: pd.DataFrame, contamination: float, method: str):
    if method == "isolation_forest":
        clf = IsolationForest(contamination=contamination, random_state=SEED)
        clf.fit(df[["Acc_Vector"]])
        return clf.predict(df[["Acc_Vector"]])
    if method == "lof":
        lof = LocalOutlierFactor(n_neighbors=50)
        pred = lof.fit_predict(df[["Acc_Vector"]])
        return pred


def plot_anomalies(
    df: pd.DataFrame,
    foldername: str,
    column_name: str,
    name_prefix: str = "",
):
    _, ax = plt.subplots(figsize=(8, 4))
    anomalies = df.loc[df[column_name] == -1, ["Acc_Vector"]]  # Anomaly
    ax.plot(df.index, df["Acc_Vector"], color="black", label="Normal")
    ax.scatter(
        anomalies.index,
        anomalies["Acc_Vector"],
        color="red",
        label="Anomaly",
    )
    plt.title(foldername.split("_")[2])
    plt.legend()
    plt.savefig(
        fname=os.path.join(
            PLOT_DIR,
            str(foldername).replace(".csv", ""),
            "anomalies" + name_prefix + ".png",
        )
    )
    plt.close()


def determine_num_clusters(
    anomalies: np.array,
    foldername: str,
    num_max_cluster: int,
    plot_knee: bool = False,
    method: str = "kmeans",
):
    if method == "kmeans":
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
                    str(foldername).replace(".csv", ""),
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
    elif method == "heuristic":
        print(anomalies.squeeze())
        cluster_counter = 0
        for num in anomalies:
            if num + 1 not in anomalies:
                print(num)
                cluster_counter += 1
        return cluster_counter + 1


def clean_anomalies(anomalies: np.array):
    # filter anomaly indices for those were at least filter_num consecutive data points are included
    delete_indices = []
    filter_num = 5
    for idx, value in enumerate(anomalies.tolist()[:-filter_num]):
        current_range = anomalies[idx : idx + filter_num]
        # print(current_range)
        ideal_range = np.array(range(anomalies[idx], anomalies[idx] + filter_num))
        # print(ideal_range)
        if not np.equal(current_range, ideal_range).all():
            delete_indices.append(idx)

    return np.delete(arr=anomalies, obj=delete_indices, axis=0)


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


def plot_anomaly_groups(df: pd.DataFrame, foldername: str):
    # plot acceleration
    _, ax = plt.subplots(figsize=(8, 4))
    anomalies = df.loc[df["AnomalyGroup"] == 1, ["Acc_Vector"]]  # Anomaly
    ax.plot(df.index, df["Acc_Vector"], color="black", label="Normal")

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
    ax.plot([], [], color="orange", alpha=0.5, label="Anomaly Interval", linewidth=3)
    plt.title(foldername.split("_")[2])

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
            str(foldername).replace(".csv", ""),
            "anomalies_grouped.png",
        )
    )
    plt.close()


def remove_overlapping_cluster_means(centers):
    problematic_distance = 200
    centers = np.sort(centers)
    non_overlapping_centers = [centers[0]]

    for i in range(1, len(centers)):
        # If the current center does not overlap with the last non-overlapping center, add it to the list
        if centers[i] > non_overlapping_centers[-1] + problematic_distance:
            non_overlapping_centers.append(centers[i])

    return np.array(non_overlapping_centers)


def anomaly_detection(df: pd.DataFrame, foldername: str, anomaly_contamination: float):
    # get ground thruth about number of throws from filename
    print("{foldername}:".format(foldername=foldername))
    num_throws, labels = determine_throws_from_filename(filename=foldername)
    print("Number of throws: ", num_throws)
    df["Acc_Vector"] = (
        df["FreeAcc_X"] ** 2 + df["FreeAcc_Y"] ** 2 + df["FreeAcc_Z"] ** 2
    ) ** 0.5
    # df["FreeAccMagnitude"] = df["FreeAccMagnitude"].rolling(window=10, center=False).mean()
    # df = df.dropna().reset_index()

    # plot acceleration for inspection
    plot_acceleration(df, foldername)

    # determine anomalies using Isolation Forest
    df["Anomaly"] = get_anomalies(
        df, contamination=anomaly_contamination, method=METHOD
    )
    # print(df["Anomaly"].value_counts())
    plot_anomalies(df, foldername, name_prefix="_raw", column_name="Anomaly")

    # get and clean anomalies
    anomalies = np.array(df.loc[df["Anomaly"] == -1].index.tolist())
    anomalies = clean_anomalies(anomalies)
    df["CleanedAnomaly"] = 1
    # print(df.index)
    df.loc[anomalies.tolist(), "CleanedAnomaly"] = -1
    anomalies = anomalies.reshape(-1, 1)

    # plot cleaned anomalies
    plot_anomalies(
        df, foldername, name_prefix="_filtered", column_name="CleanedAnomaly"
    )

    # determine number of clusters
    if len(anomalies) >= 2:
        n_clusters = determine_num_clusters(
            anomalies=anomalies,
            num_max_cluster=len(anomalies),
            foldername=foldername,
            plot_knee=True,
            method="kmeans",
        )
        # get cluster means
        cluster_means_with_overlaps = get_cluster_means(n_clusters, anomalies)
        cluster_means = remove_overlapping_cluster_means(cluster_means_with_overlaps)

        print("Predicted numbers of throws", len(cluster_means), "\n")

        # mark cluster means in df
        df = pd.concat(
            [
                df,
                pd.DataFrame(np.zeros(shape=(len(df), 1)), columns=["AnomalyGroup"]),
            ],
            axis=1,
        )
        df.loc[cluster_means, "AnomalyGroup"] = 1

        # plot anomalies groups
        plot_anomaly_groups(df=df, foldername=foldername)

    else:
        print("Too few anomalies found!")
        cluster_means, labels = None, None

    return cluster_means, labels


if __name__ == "__main__":
    # sort subfolders by date
    path = "data/20240626"
    dfs = []
    foldernames = []
    subfolders = os.listdir(os.path.join(os.getcwd(), path))
    subfolders.sort(key=lambda x: x.split("_")[1])

    # get list of df for all measurements in defined folder
    for subfolder in subfolders:
        if not subfolder.startswith("2024"):
            continue
        foldernames.append(subfolder)
        for file in os.listdir(os.path.join(os.getcwd(), path, subfolder)):
            if file.endswith(".csv"):
                dfs.append(
                    pd.read_csv(
                        os.path.join(os.getcwd(), path, subfolder, file),
                        skiprows=11,
                    )
                )

    for df_idx, df in enumerate(dfs):
        _, _ = anomaly_detection(df, foldernames[df_idx], 0.03)
