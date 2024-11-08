import os
import pickle
import lzma

def load_data(folder_path):
    inputs = []
    targets = []
    number_of_samples = 0
    files_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_path in files_paths:
        try:
            with lzma.open(file_path, 'rb') as f:
                data = pickle.load(f)
                number_of_samples += 2*len(data)
                inputs.extend([list(x.raycast_distances) + [x.car_speed] for x in data])
                targets.extend([list(x.current_controls) for x in data])

                # Add symmetric data
                inputs.extend([list(reversed(x.raycast_distances)) + [x.car_speed] for x in data])
                targets.extend([[x.current_controls[0], x.current_controls[1], x.current_controls[3], x.current_controls[2]] for x in data])

        except Exception as e:
            print(f"Error loading data from {file_path} due to {e}")
    print(f"Number of samples: {number_of_samples}")
    return inputs, targets