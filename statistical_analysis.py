import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power, tt_solve_power
import rpy2.robjects.numpy2ri
from anomaly_detection import PLOT_DIR

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

stats_r = importr("stats")


def create_dfs(path, compare, feature, mode="max_acceleration"):
    dfs_to_compare = []

    # names of the datasets to compare: at least on of ['Jannie', 'Julian', 'Kevin', 'Forehand']
    for name in compare:
        values_list = []
        directory = os.path.join(path, name)
        # iterate over all throws in the directory and compute maximum or variance for all features; add result as one row to the dataframe
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                if mode == "max_acceleration":
                    values = df.max()
                elif mode == "variance_gyroscope":
                    values = df.var()
                elif mode == "above_threshold_duration":
                    values = get_above_threshold_duration(df[feature], 150)
                values_list.append(values)

        values_df = pd.DataFrame(values_list)
        dfs_to_compare.append(values_df)

    return dfs_to_compare


def get_above_threshold_duration(acceleration_data, threshold):
    above_threshold_duration = 0
    for acceleration in acceleration_data:
        if acceleration > threshold:
            above_threshold_duration += 1
    return above_threshold_duration


def t_test(sample_1, sample_2, paired, alpha=0.05, equal_var=True):
    reject = False
    if paired:
        _, p_value = stats.ttest_rel(sample_1, sample_2)
    else:
        _, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=equal_var)
    if p_value < alpha:
        reject = True
    return reject, p_value


def non_parametric_test(
    sample_1, sample_2, paired, alpha=0.05, alternative="two-sided"
):
    reject = False
    if paired:
        _, p_value = stats.wilcoxon(sample_1, sample_2, alternative=alternative)
    else:
        _, p_value = stats.mannwhitneyu(sample_1, sample_2, alternative=alternative)
    if p_value < alpha:
        reject = True
    return reject, p_value


def verify_normal_dist(data):
    # Use Q-Q plot to visually verify that data follows a normal distribution
    z = (data - np.mean(data)) / np.std(data)
    stats.probplot(z, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.show()
    return


def equal_variance_test(sample_1, sample_2, alpha=0.05):
    # Levene's test H0: both samples come from population with equal variance
    reject = False
    _, p_value = stats.levene(sample_1, sample_2)
    if p_value < alpha:
        reject = True
    _, ax = plt.subplots()
    ax.boxplot([sample_1, sample_2])
    ax.set_ylabel("Values")
    ax.set_title("Boxplots of Sample 1 and Sample 2")
    plt.show()
    return reject, p_value


def compare_all(path, mode, parametric=False):
    regarded_features = ["Acc_Vector", "FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z"]
    # regarded_features = ["Euler_X", "Euler_Y", "Euler_Z"]
    for feature in regarded_features:
        paired = True
        print(f"{feature}\n")

        BH, FH, PT = create_sample_groups(path, mode, feature)

        sample_pairs = [
            (BH, FH),
            (BH, PT),
            (PT, FH),
        ]

        dict_names = {
            "BH": BH,
            "FH": FH,
            "PT": PT,
        }

        for sample_1, sample_2 in sample_pairs:
            if parametric:
                reject_levene, _ = equal_variance_test(sample_1, sample_2)
                if reject_levene and paired:
                    print(f"No t-Test possible for {sample_1} and {sample_2}:")
                    print("Significant difference in variances detected")
                    continue
                reject_t_test, p_value = t_test(
                    sample_1,
                    sample_2,
                    paired,
                    equal_var=not reject_levene,
                )
                test_type = "Welch's Test" if reject_levene else "t-Test"
            else:
                reject_test, p_value = non_parametric_test(sample_1, sample_2, paired)
                test_type = "Wilcoxon" if paired else "Mann-Whitney"

            rejection_status = (
                "rejected"
                if (parametric and reject_t_test) or (not parametric and reject_test)
                else "not rejected"
            )
            print(
                f"{test_type} of {find_key_by_value(dict_names, sample_1)} and {find_key_by_value(dict_names, sample_2)}:"
            )
            print(f"H0 is {rejection_status} with p-value = {p_value}\n")
        print("-" * 50)


def fisher_exact_test(samples, threshold, alternative="two-sided", alpha=0.05):

    # Proportion test comparing two sample groups
    reject = False
    df1, df2 = samples[0], samples[1]
    sum1 = (df1[0] > threshold).sum()
    sum2 = (df2[0] > threshold).sum()
    contingency_table = [
        [sum1, sum2],
        [len(df1) - sum1, len(df2) - sum2],
    ]
    _, p_value = stats.fisher_exact(contingency_table, alternative=alternative)
    if p_value < alpha:
        reject = True
    prop_1 = sum1 / len(df1)
    prop_2 = sum2 / len(df2)
    rejection_status = "rejected" if reject else "not rejected"
    print(f"Fisher exact test: prop_1 = {prop_1}, prop_2 = {prop_2}, alpha = {alpha}")
    print(f"H0 is {rejection_status} with p-value = {p_value}\n")
    return reject, p_value


def multi_proportions(samples, threshold):
    # Multi-proportion test
    df1, df2, df3 = samples[0], samples[1], samples[2]
    successes = [(df[0] > threshold).sum() for df in [df1, df2, df3]]
    total_samples = [len(df[0]) for df in [df1, df2, df3]]
    prop_test = stats_r.prop_test(np.array(successes), np.array(total_samples))
    print(prop_test)
    return


def get_above_threshold_duration(acceleration_data, threshold):
    above_threshold_duration = 0
    for acceleration in acceleration_data:
        if acceleration > threshold:
            above_threshold_duration += 1
    return above_threshold_duration


def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val.equals(value):
            return key
    return None


def calculate_sample_size(samples, paired=False, power=0.9):
    print(
        f"Sample size calculation for {samples}, {'paired' if paired else 'unpaired'}"
    )

    df1, df2 = create_dfs(input_path, samples, mode="max_acceleration")
    std1 = df1["Acc_Vector"].std()
    std2 = df2["Acc_Vector"].std()
    mean1 = df1["Acc_Vector"].mean()
    mean2 = df2["Acc_Vector"].mean()
    delta = abs(mean1 - mean2)

    if paired:
        n1 = tt_ind_solve_power(effect_size=delta / std1, alpha=0.01, power=power)
        n5 = tt_ind_solve_power(effect_size=delta / std1, alpha=0.05, power=power)
    else:
        n1 = tt_solve_power(effect_size=delta / std1, alpha=0.01, power=power)
        n5 = tt_solve_power(effect_size=delta / std1, alpha=0.05, power=power)

    print(
        f"Debug: std1={std1}, std2={std2}, mean1={mean1}, mean2={mean2}, delta={delta}"
    )
    print("For alpha=0.01: ", n1)
    print("For alpha=0.05: ", n5)
    print("--------------------------------------------------------------")


def create_sample_groups(path, mode, feature):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    values_dict = {"BH": [], "FH": [], "PT": []}

    for file in csv_files:
        throw = pd.read_csv(file)
        label = throw["Label"][0]
        if mode == "max_acceleration":
            values = throw[feature].max()
        elif mode == "above_threshold_duration":
            values = get_above_threshold_duration(throw[feature], 150)
        values_dict[label].append(values)

    return (
        pd.DataFrame(values_dict["BH"]),
        pd.DataFrame(values_dict["FH"]),
        pd.DataFrame(values_dict["PT"]),
    )


if __name__ == "__main__":
    input_path = "data/splitted_throws"
    # above_threshold_duration, max_acceleration
    BH, FH, PT = create_sample_groups(
        input_path, "above_threshold_duration", "Acc_Vector"
    )
    reject, p_value = fisher_exact_test([BH, FH], 7, alternative="greater")
