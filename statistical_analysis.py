import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def create_dfs(path, compare):
    dfs_to_compare = []

    for name in compare:
        max_values_list = []
        directory = os.path.join(path, name)
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                max_values = df.max()
                max_values_list.append(max_values)

        max_values_df = pd.DataFrame(max_values_list)
        dfs_to_compare.append(max_values_df)

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

def verify_normal_dist(data):
    # Use Q-Q plot to visually verify that data follows a normal distribution
    print(data)
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

    fig, ax = plt.subplots()
    ax.boxplot([sample_1, sample_2])
    ax.set_xticklabels(['Data 1', 'Data 2'])
    ax.set_ylabel('Values')
    ax.set_title('Boxplots of Sample 1 and Sample 2')
    #plt.show()
    return reject, p_value

def compare_all(path):
    regarded_features = ['Acc_Vector', 'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']
    sample_pairs = [('Julian', 'Jannie'), ('Julian', 'Kevin'), ('Jannie', 'Kevin'), ('Julian', 'Forehand')]
    
    for feature in regarded_features:
        print(f'{feature}\n')
        for sample_1, sample_2 in sample_pairs:
            compare = [sample_1, sample_2]
            dfs_to_compare = create_dfs(path, compare)
            
            # Determine if the t-test should be paired
            paired = (sample_2 == 'Forehand')
            
            # Perform Levene's test to check for equal variances
            reject_levene, _ = equal_variance_test(dfs_to_compare[0][feature], dfs_to_compare[1][feature])
            
            # Perform t-test
            reject_t_test, p_value = t_test(dfs_to_compare[0][feature], dfs_to_compare[1][feature], 
                                            paired, equal_var=not reject_levene)
            
            test_type = "Welch's Test" if reject_levene and not paired else 't-Test'
            rejection_status = 'rejected' if reject_t_test else 'not rejected'
            print(f'{test_type} of {sample_1} and {sample_2}:')
            print(f'H0 is {rejection_status} with p-value = {p_value}\n')
        print('-' * 50)

if __name__ == "__main__":
    input_path = 'data/20240430_splitted'
    compare_all(input_path)
