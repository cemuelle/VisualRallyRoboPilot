import torchvision.transforms as transforms
import torch
from PIL import Image

import matplotlib.pyplot as plt
resize = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

preprocess = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    resize,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

greyscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    resize,
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class ColorThresholdTransform:
    def __init__(self, target_color=None, margin=0.01):
        """
        Args:
            target_color (tuple): RGB color to threshold
            margin (float): Margin of error for the threshold
        """
        self.setColor(target_color)
        self.margin = margin

    def setColor(self, target_color):
        """
        Args:
            target_color (tuple): RGB color to threshold
        """
        if target_color is None or (isinstance(target_color, (list, tuple)) and target_color == -1):
            self.target_color = None
        else:
            self.target_color = torch.tensor(target_color, dtype=torch.float32).view(3, 1, 1)
            self.target_color /= 255.0

    def __call__(self, img):
        """
        Args:
            img (PIL.Image.Image): Image to threshold
        Returns:
            torch.Tensor: Binary mask of the image
        """
        if isinstance(img, Image.Image):
            img = resize(img)

        if img.shape[0] != 3:
            raise ValueError("Input image must be RGB")
        
        if self.target_color is None:
            return torch.zeros(img.shape[1], img.shape[2], dtype=torch.bool)

        return torch.abs(img - self.target_color).sum(dim=0) < self.margin

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
            transform = ColorThresholdTransform(-1, margin=0.01)
            # transform = ColorThresholdTransform(target_color, margin=0.01)
            image = Image.fromarray(data[0].image)
            mask = transform(image)
            plt.imshow(mask)
            plt.show()