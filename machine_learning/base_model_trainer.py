import torch
from utils import *
import torch.nn as nn
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from models import AlexNetPerso
from preprocessing import preprocess
from torch.utils.data import DataLoader, TensorDataset, random_split

def trainer(model, training_dataloader, validation_dataloader, num_epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/pilot_trainer_{}'.format(timestamp))
    writer = None
    epoch_number = 0

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        avg_loss = train_one_epoch(epoch, training_dataloader, writer)

        running_loss = 0.0

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                inputs_image, inputs_color, labels = data
                vinputs_image, vinputs_color, vlabels = inputs_image.to(device), inputs_color.to(device), labels.to(device)

                voutputs = model(vinputs_image, vinputs_color)
                vloss = criterion(voutputs, vlabels)

                running_loss += vloss.item()
        
        avg_vloss = running_loss / (i+1)

        print(f"Validation Loss: {avg_vloss}, Training Loss: {avg_loss}")

        if writer is not None:
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch)
            writer.flush()

        if avg_vloss < best_loss:
            best_loss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)
        

def train_one_epoch(epoch_index, training_dataloader, tb_writer=None):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_dataloader, 0):
        inputs_image, inputs_color, labels = data
        vinputs_image, vinputs_color, vlabels = inputs_image.to(device), inputs_color.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(vinputs_image, vinputs_color)
        loss = criterion(outputs, vlabels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if i % 100 == 99:
            last_loss = running_loss / 100
            running_loss = 0.0
            print(f" Batch {i + 1}, Loss: {last_loss}")
            if tb_writer is not None:
                tb_x = epoch_index * len(training_dataloader) + i
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            
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

    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())

    class_weights = torch.tensor([0.6, 2.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainer(model, training_dataloader, validation_dataloader, 100)