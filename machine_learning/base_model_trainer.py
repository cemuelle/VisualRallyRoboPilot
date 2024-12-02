import torch
from utils import *
import torch.nn as nn
from datetime import datetime
from models import AlexNetPerso
from preprocessing import preprocess
from torch.utils.data import DataLoader, random_split

import os

def trainer(model, training_dataloader, validation_dataloader, num_epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    best_loss = float('inf')

    last_saved_model_path = None

    training_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        avg_loss = train_one_epoch(training_dataloader)

        running_loss = 0.0

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                inputs_image, inputs_color, inputs_speed, labels = data
                vinputs_image, vinputs_color, vinputs_speed, vlabels = inputs_image.to(device), inputs_color.to(device), inputs_speed.to(device), labels.to(device)
                vinputs_speed = vinputs_speed.unsqueeze(1)

                voutputs = model(vinputs_image, vinputs_color, vinputs_speed)
        
                vloss = criterion(voutputs, vlabels)

                running_loss += vloss.item()
        
        avg_vloss = running_loss / (i+1)

        print(f"Validation Loss: {avg_vloss}, Training Loss: {avg_loss}")
        # print(f"predicted: {voutputs}, sigmoid : {torch.sigmoid(voutputs)}, actual: {vlabels}")
        training_loss.append(avg_loss)
        validation_loss.append(avg_vloss)

        if avg_vloss < best_loss:
            best_loss = avg_vloss
            model_path = 'models/model_{}_{}.pth'.format(timestamp, epoch+1)
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)

            # Remove the last saved model if it exists
            if last_saved_model_path is not None:
                if os.path.exists(last_saved_model_path):
                    os.remove(last_saved_model_path)
                    print(f"Removed previous model {last_saved_model_path}")

            last_saved_model_path = model_path

    # make a graph of the training and validation loss across epochs

    import matplotlib.pyplot as plt

    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel(f"Loss ({criterion.__class__.__name__})")
    plt.legend()
    plt.title("Training the CNN")
    plt.show()
        

def train_one_epoch(training_dataloader):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_dataloader):
        inputs_image, inputs_color, inputs_speed, labels = data
        vinputs_image, vinputs_color, vinputs_speed, vlabels = inputs_image.to(device), inputs_color.to(device), inputs_speed.to(device), labels.to(device)
        vinputs_speed = vinputs_speed.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(vinputs_image, vinputs_color, vinputs_speed)
        loss = criterion(outputs, vlabels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if i % 100 == 99:
            last_loss = running_loss / 100
            running_loss = 0.0
            print(f" Batch {i + 1}, Loss: {last_loss}")

    return last_loss

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = AlexNetPerso(4)
    model.to(device)

    print("Loading data...")
    dataset = CustomDataset("./data", transform=preprocess)
    training_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - training_size
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    training_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())

    pos_weight = dataset.get_distribution().sum() / dataset.get_distribution()
    pos_weight[1] = pos_weight[1] * 0.3

    print(pos_weight)
    
    class_weights = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    trainer(model, training_dataloader, validation_dataloader, 100)