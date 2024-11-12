import os
import pickle
import lzma
import numpy as np

def load_data(folder_path):
    inputs = [] 
    targets = []
    number_of_samples = 0
    # Use os.walk to traverse through all subfolders and files
    for root, dirs, files in os.walk(folder_path):
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
                        inputs.extend([[x.image, colToRgb(subfolder_name)] for x in data])
                        targets.extend([list(x.current_controls) for x in data])

                        # Add symmetric data
                        inputs.extend([[np.fliplr(x.image), colToRgb(subfolder_name)] for x in data])
                        targets.extend([[x.current_controls[0], x.current_controls[1], x.current_controls[3], x.current_controls[2]] for x in data])

                except Exception as e:
                    print(f"Error loading data from {file_path} due to {e}")
                


    print(f"Total number of samples: {number_of_samples}.")
    return inputs,targets
 
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


def colToRgb(colorName):
    match colorName:
        case "red":
            return [255,0,0]
        case "cyan":
            return [0,255,255]
        case "blue":
            return [0,0,255]
        case default:
            return [0,0,0]