import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from anomaly_detection import create_directories
from anomaly_detection import anomaly_detection
from throw_cutter import cut_throws


PLOT_DIR = "figures/anomaly_detection"
DF_DIR = "data/splitted_throws"


# visualise individual throws and save them in the corresponding folder
def vis_of_throw(throw: pd.DataFrame, foldername: str, idx: int):
    plt.figure(figsize=(8, 4))
    plt.plot(throw["Acc_Vector"], c="black")

    title_part = (
        foldername.split("_")[2] if len(foldername.split("_")) > 2 else foldername
    )
    plt.title(f"{title_part}_{idx}")
    plt.xlabel("Time")
    plt.ylabel("Acc_Vector")

    plot_dir = os.path.join(PLOT_DIR, str(foldername).replace(".csv", ""))
    create_directories(plot_dir)

    plot_path = os.path.join(plot_dir, f"throw_{idx}.png")
    plt.savefig(fname=plot_path)
    plt.close()


def get_games(paths, anomaly_contanimations):
    games = []
    foldernames = []
    anomalies_list = []
    # sort subfolders by date
    for path, anomaly_contanimation in zip(paths, anomaly_contanimations):
        subfolders = os.listdir(os.path.join(os.getcwd(), path))
        subfolders.sort(key=lambda x: x.split("_")[1])

        # get csv files for all games
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
                    anomalies_list.append(anomaly_contanimation)

    return games, foldernames, anomalies_list


# go through all games and extract BH, FH and PT
def collect_data(paths, anomaly_contanimations):
    throws_list = []
    # references for cutting alignment:
    references = {"BH": None, "FH": None, "PT": None}

    games, foldernames, anomalies_list = get_games(paths, anomaly_contanimations)

    for game_idx, (game, anomaly_contanimation) in enumerate(
        zip(games, anomalies_list)
    ):
        # delete all images of individual throws
        for idx in range(len(games)):
            path = os.path.join(
                os.path.join(PLOT_DIR, str(foldernames[game_idx]).replace(".csv", "")),
                f"throw_{idx}.png",
            )
            if os.path.isfile(path):
                os.remove(path)

        # anomaly detection in game
        cluster_means, labels = anomaly_detection(
            game, foldernames[game_idx], anomaly_contanimation
        )

        # if correct number of anomalies (=throws) were found, extract them
        if len(labels) == len(cluster_means):
            throws = [
                cut_throws(cluster_mean, game, label, references, "dtw")
                for cluster_mean, label in zip(cluster_means, labels)
            ]

            for throw_idx, throw in enumerate(throws):
                throw_copy = throw.copy()
                throw_copy["Label"] = labels[throw_idx]
                throws_list.append(throw_copy)
                label = labels[throw_idx]
                if references[label] is None:
                    references[label] = throw_copy
                vis_of_throw(throw, foldernames[game_idx], throw_idx)

        else:
            print("Throws were not identifyed correctly\n")

    print(
        f"Number of identified forhand throws: {sum(throw['Label'].iloc[0] == 'FH' for throw in throws_list)}"
    )
    print(
        f"Number of identified backhand throws: {sum(throw['Label'].iloc[0] == 'BH' for throw in throws_list)}"
    )
    print(
        f"Number of identified putt throws: {sum(throw['Label'].iloc[0] == 'PT' for throw in throws_list)}\n"
    )

    return throws_list


# visualise all identified throws of one category (BH, FH, PT): find in figures/overview
def visualise_all(throw_set, name):
    time = None
    for throw in throw_set:
        throw["Acc_Vector"]
        throw["SampleTimeFine"] /= 1000
        if time is None:
            time = throw.SampleTimeFine
        plt.plot(time, throw["Acc_Vector"], linewidth=1)

    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("All " + name + " throws")

    plot_dir = os.path.join(PLOT_DIR, "overview")
    create_directories(path=plot_dir)
    plot_path = os.path.join(plot_dir, f"all_throws_{name}.png")
    plt.savefig(fname=plot_path)
    plt.close()

    return


def remove_files_in_directory(directory):
    if os.path.exists(directory):
        file_list = os.listdir(directory)
        for file in file_list:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
    else:
        print(f"Directory {directory} does not exist.")


if __name__ == "__main__":
    # get throws from games and extract features
    # paths = ["data/manually_cutted_throws"]
    anomaly_contaminations = [0.01, 0.03]
    # paths = ["data/20240604"]
    paths = ["data/20240604", "data/20240608"]
    throws_list = collect_data(paths, anomaly_contaminations)

    os.makedirs(DF_DIR, exist_ok=True)
    remove_files_in_directory(DF_DIR)
    for i, throw in enumerate(throws_list):
        file_path = os.path.join(DF_DIR, f"timeseries_{i}.csv")
        throw.to_csv(file_path, index=False)

    visualise_all([df for df in throws_list if df["Label"].iloc[0] == "FH"], "FH")
    visualise_all([df for df in throws_list if df["Label"].iloc[0] == "BH"], "BH")
    visualise_all([df for df in throws_list if df["Label"].iloc[0] == "PT"], "PT")
