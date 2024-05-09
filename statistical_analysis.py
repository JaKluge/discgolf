import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def create_dfs(path, compare, mode='variance_gyroscope'):
    dfs_to_compare = []

    for name in compare:
        values_list = []
        directory = os.path.join(path, name)
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                if mode == 'max_acceleration':
                    values = df.max()
                elif mode == 'variance_gyroscope':
                    values = df.var()
                values_list.append(values)

        values_df = pd.DataFrame(values_list)
        dfs_to_compare.append(values_df)

    return dfs_to_compare

def t_test(sample_1, sample_2, paired, alpha=0.05, equal_var=True):
    reject = False
    if paired:
        _, p_value = stats.ttest_rel(sample_1, sample_2)   
    else:
        _, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=equal_var)
    if p_value < alpha:
        reject = True

    return reject, p_value

def non_parametric_test(sample_1, sample_2, paired, alpha=0.05, alternative = 'two-sided'):
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
    z = (data-np.mean(data))/np.std(data)
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
    ax.set_ylabel('Values')
    ax.set_title('Boxplots of Sample 1 and Sample 2')
    plt.show()
    return reject, p_value

def compare_all(path, parametric=False):
    #regarded_features = ['Acc_Vector', 'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']
    regarded_features = ['Euler_X', 'Euler_Y', 'Euler_Z']
    sample_pairs = [('Jannie', 'Julian'), ('Julian', 'Kevin'), ('Jannie', 'Kevin'), ('Julian', 'Forehand')]

    for feature in regarded_features:
        print(f'{feature}\n')
        for sample_1, sample_2 in sample_pairs:
            compare = [sample_1, sample_2]
            dfs_to_compare = create_dfs(path, compare, mode='variance_gyroscope')
            paired = (sample_2 == 'Forehand')

            if parametric:
                reject_levene, _ = equal_variance_test(dfs_to_compare[0][feature], dfs_to_compare[1][feature])
                if reject_levene and paired:
                    print(f'No t-Test possible for {sample_1} and {sample_2}:')
                    print('Significant difference in variances detected')
                    continue
                reject_t_test, p_value = t_test(dfs_to_compare[0][feature], dfs_to_compare[1][feature], paired, equal_var=not reject_levene)
                test_type = "Welch's Test" if reject_levene else 't-Test'
            else:
                reject_test, p_value = non_parametric_test(dfs_to_compare[0][feature], dfs_to_compare[1][feature], paired)
                test_type = "Wilcoxon" if paired else 'Mann-Whitney'

            rejection_status = 'rejected' if (parametric and reject_t_test) or (not parametric and reject_test) else 'not rejected'
            print(f'{test_type} of {sample_1} and {sample_2}:')
            print(f'H0 is {rejection_status} with p-value = {p_value}\n')
        print('-' * 50)


if __name__ == "__main__":
    input_path = 'data/20240430_splitted'
    compare_all(input_path)





