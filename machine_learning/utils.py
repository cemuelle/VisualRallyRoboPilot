import os
import pickle
import lzma
import numpy as np
from machine_learning.preprocessing import preprocess
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.inputs_image = []
        self.inputs_color = []
        self.inputs_speed = []
        self.targets = []
        self.transform = transform
        self.number_of_samples = 0
        self.load_data(folder_path)

    def load_data(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            subfolder_name = os.path.relpath(root, folder_path)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    try:
                        with lzma.open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            self.number_of_samples += 2 * len(data)
                            for x in data:
                                image = Image.fromarray(x.image)
                                if self.transform:
                                    image = self.transform(image)
                                self.inputs_speed.append(x.car_speed)
                                self.inputs_image.append(image)
                                self.inputs_color.append(colToRgb(subfolder_name))
                                self.targets.append(list(x.current_controls))
                                
                                # Add symmetric data
                                flipped_image = Image.fromarray(np.fliplr(x.image))
                                if self.transform:
                                    flipped_image = self.transform(flipped_image)
                                self.inputs_speed.append(x.car_speed)
                                self.inputs_image.append(flipped_image)
                                self.inputs_color.append(colToRgb(subfolder_name))
                                self.targets.append([x.current_controls[0], x.current_controls[1], x.current_controls[3], x.current_controls[2]])
                    except Exception as e:
                        print(f"Error loading data from {file_path} due to {e}")

        print(f"Number of samples: {self.number_of_samples}")

    def __len__(self):
        return len(self.inputs_image)

    def __getitem__(self, idx):
        return self.inputs_image[idx], torch.tensor(self.inputs_color[idx], dtype=torch.float32), torch.tensor(self.inputs_speed[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
    
    def get_distribution(self):
        # get how many samples of each class
        return np.sum(np.array(self.targets), axis=0)

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
        
def get_most_recent_model(models_folder):
    # Get all files in the models folder
    model_files = [
        f for f in os.listdir(models_folder)
        if os.path.isfile(os.path.join(models_folder, f)) and f.startswith("model_") and f.endswith(".pth")
    ]
    
    # Extract timestamp and epoch from filenames
    def parse_model_file(filename):
        try:
            # Remove "model_" prefix and ".pth" suffix
            timestamp_epoch = filename.replace("model_", "").replace(".pth", "")
            timestamp, epoch = timestamp_epoch.rsplit("_", 1)
            return timestamp, int(epoch)
        except ValueError:
            return None

    # Sort files by timestamp and epoch
    model_files = [
        (filename, *parse_model_file(filename))
        for filename in model_files
        if parse_model_file(filename) is not None
    ]
    model_files.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by timestamp, then epoch, descending

    if model_files:
        return os.path.join(models_folder, model_files[0][0])
    else:
        return None