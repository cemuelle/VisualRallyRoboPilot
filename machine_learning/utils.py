import os
import pickle
import lzma
import numpy as np

def load_data(folder_path):
    data_per_subfolder = {}
    # Use os.walk to traverse through all subfolders and files
    for root, dirs, files in os.walk(folder_path):
        # Create a list to store inputs and targets for the current subfolder
        inputs = []
        targets = []
        number_of_samples = 0
        subfolder_name = os.path.relpath(root, folder_path)  # Get name of the current subfolder
 
        # Iterate over each file in the subfolder
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                try:
                    with lzma.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        number_of_samples += 2 * len(data)
                        # Process the original data
                        inputs.extend([x.image for x in data])
                        targets.extend([list(x.current_controls) for x in data])

                        # Add symmetric data
                        inputs.extend([np.fliplr(x.image) for x in data])
                        targets.extend([[x.current_controls[0], x.current_controls[1], x.current_controls[3], x.current_controls[2]] for x in data])

                except Exception as e:
                    print(f"Error loading data from {file_path} due to {e}")
                
        # After processing all files in the current subfolder, store inputs and targets
        if number_of_samples > 0:
            data_per_subfolder[subfolder_name] = (inputs, targets)
 
    print(f"Loaded data from {len(data_per_subfolder)} subfolders.")
    #print the number of samples in each subfolder
    for subfolder, (inputs, targets) in data_per_subfolder.items():
        print(f"Subfolder {subfolder} has {len(inputs)} samples.")
    print(f"Total number of samples: {sum(len(inputs) for inputs, _ in data_per_subfolder.values())}")
    return data_per_subfolder
 
# used to load data for test
def load_data_single_folder(folder_path):
    inputs = []
    targets = []
    number_of_samples = 0
    files_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_path in files_paths:
        try:
            with lzma.open(file_path, 'rb') as f:
                data = pickle.load(f)
                number_of_samples += 2*len(data)
                inputs.extend([x.image for x in data])
                targets.extend([list(x.current_controls) for x in data])
 
                # Add symmetric data
                inputs.extend([np.fliplr(x.image) for x in data])
                targets.extend([[x.current_controls[0], x.current_controls[1], x.current_controls[3], x.current_controls[2]] for x in data])
 
        except Exception as e:
            print(f"Error loading data from {file_path} due to {e}")
    print(f"Number of samples: {number_of_samples}")
    return inputs, targets


#load_data_single_folder("./data/blue")