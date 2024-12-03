import torchvision.transforms as transforms
import torch
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

greyscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class ColorThresholdTransform:
    def __init__(self, target_color, margin=0.01):
        self.target_color = torch.tensor(target_color, dtype=torch.float32).view(3, 1, 1)
        self.target_color /= 255.0
        self.margin = margin

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)

        if img.shape[0] != 3:
            raise ValueError("Input image must be RGB")
        
        distance_per_channel = torch.abs(img - self.target_color)

        total_distance = distance_per_channel.sum(dim=0)

        return total_distance < self.margin
        

if __name__ == "__main__":

    import pickle
    import lzma
    import matplotlib.pyplot as plt

    with lzma.open("./data/cyan/record_cyan_0.npz", "rb") as file:
        data = pickle.load(file)

        if data[0].image is not None:
            # plt.imshow(data[0].image)
            # plt.show()
            target_color = torch.tensor([255.0, 0.0, 0.0])
            transform = ColorThresholdTransform(target_color, margin=0.01)
            image = Image.fromarray(data[0].image)
            mask = transform(image)
            plt.imshow(mask)
            plt.show()