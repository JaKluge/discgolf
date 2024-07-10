import numpy as np
import pandas as pd
import os


MANUAL_DIR = "data/manually_cutted_throws"


# difference of start and end indices should be larger than 40
def manual_cutter(foldername, start_idx, end_idx, label):
    path = os.path.join("data", foldername.split("_")[0])

    for file in os.listdir(os.path.join(os.getcwd(), path, foldername)):
        if file.endswith(".csv"):
            # Read the first 11 rows to get the header and the rest of the data for the game
            with open(os.path.join(os.getcwd(), path, foldername, file), "r") as f:
                header_lines = [next(f) for _ in range(11)]
            game = pd.read_csv(
                os.path.join(os.getcwd(), path, foldername, file),
                skiprows=11,
            )

        # cut throw
        print(foldername)
        throw = game.iloc[start_idx:end_idx]
        throw_foldername = (
            foldername.split("_")[0]
            + "_"
            + foldername.split("_")[1]
            + "_"
            + (foldername.split("_")[2]).split("-")[0]
            + "-"
            + label
            + "_outside"
        )
        print(throw_foldername)
        throw_folder_path = os.path.join(MANUAL_DIR, throw_foldername)
        os.makedirs(MANUAL_DIR, exist_ok=True)
        os.makedirs(throw_folder_path, exist_ok=True)
        file_path = os.path.join(throw_folder_path, f"{throw_foldername}.csv")

        with open(file_path, "w") as f:
            for line in header_lines:
                f.write(line)
            throw.to_csv(f, index=False)


if __name__ == "__main__":
    manual_cutter("20240612_202444_U-BH-FH-PT_outside", 100, 300, "BH")
    manual_cutter("20240612_202444_U-BH-FH-PT_outside", 900, 1100, "FH")
    manual_cutter("20240612_202444_U-BH-FH-PT_outside", 1250, 1400, "PT")

    manual_cutter("20240612_202806_U-BH-FH-PT_outside", 80, 200, "BH")
    manual_cutter("20240612_202806_U-BH-FH-PT_outside", 500, 650, "FH")
    manual_cutter("20240612_202806_U-BH-FH-PT_outside", 900, 1050, "PT")

    manual_cutter("20240612_201038_U-BH-FH-PT_outside", 250, 500, "BH")
    manual_cutter("20240612_201038_U-BH-FH-PT_outside", 1000, 1250, "FH")
    manual_cutter("20240612_201038_U-BH-FH-PT_outside", 1500, 1750, "PT")

    manual_cutter("20240623_141340_am-BH-FH_outside", 1550, 1750, "BH")
    manual_cutter("20240623_141340_am-BH-FH_outside", 2700, 2900, "FH")

    manual_cutter("20240623_141805_am-BH-FH_outside", 1220, 1420, "BH")
    manual_cutter("20240623_141805_am-BH-FH_outside", 2950, 3150, "FH")
