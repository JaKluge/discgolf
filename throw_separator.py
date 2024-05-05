import os
import pandas as pd
import numpy as np
import shutil

def find_maxima_of_throws(data, window_size=30, backhand=True):
    if backhand:
        acc_x_threshold = 120
    else:
        acc_x_threshold = 40
    
    above_threshold_positions = data[data['FreeAcc_X'] > acc_x_threshold].index
    max_indices = []
    
    for position in above_threshold_positions:
        start_index = max(0, position - window_size)
        end_index = min(len(data), position + window_size)
        window_data = data.iloc[start_index:end_index]
        max_index = window_data['FreeAcc_X'].idxmax()
        max_indices.append(max_index)
    
    max_indices = np.array(list(set(max_indices)))
    print(f'This file contained {len(max_indices)} throws\n')
    return sorted(max_indices)

def split_and_save_throws(input_path, output_path, time_window=25):
    files = os.listdir(input_path)
    num_metadata_lines = 11
    folder_names = ['Jannie', 'Julian', 'Kevin', 'Forehand']
    folder_name = None
    
    for file_name in files:
        backhand = True
        
        if not file_name.startswith('2024'):
            continue
        
        print(file_name)
        input_folder = os.path.join(input_path, file_name)
        ts_file = os.listdir(input_folder)[0]
        
        if not ts_file.endswith(".csv"):
            continue
        
        input_file = os.path.join(input_folder, ts_file)
        
        for name in folder_names:
            if name in input_folder:
                output_folder = os.path.join(output_path, name)
                folder_name = name
                break
      
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        df = pd.read_csv(input_file, sep=',', skiprows=num_metadata_lines)
        
        if folder_name == "Forehand":
            backhand = False
            
        throw_indices = find_maxima_of_throws(df, backhand=backhand)
        
        for i, throw_index in enumerate(throw_indices):
             start_time = max(0, throw_index - time_window)
             end_time = min(len(df), throw_index + time_window)
             throw_data = df.iloc[start_time:end_time]
             throw_filename = os.path.join(output_folder, ts_file[:-4] + f"_{i+1}.csv")
             throw_data.to_csv(throw_filename, index=False)

if __name__ == "__main__":
    input_path = "data/20240430"
    output_path = "data/20240430_splitted"
    
    # Remove folder
    # shutil.rmtree(output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    split_and_save_throws(input_path, output_path)
