import torch
import torch.nn as nn
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from models import AlexNetPerso

def trainer(model, training_dataloader, validation_dataloader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetPerso(4, 0)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = torch.tensor([0.6, 2.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/pilot_trainer_{}'.format(timestamp))
    writer = None
    epoch_number = 0

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        avg_loss = train_one_epoch(epoch, writer)

        running_loss = 0.0

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                vinputs, vlabels = data
                vinputs, vlabels = inputs.to(device), labels.to(device)

                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)

                running_loss += vloss
        
        avg_vloss = running_loss / len(i+1)

        print(f"Validation Loss: {avg_vloss}, Training Loss: {avg_loss}")

        if writer is not None:
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch)
            writer.flush()

        if avg_vloss < best_loss:
            best_loss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
        

def train_one_epoch(epoch_index, tb_writer=None):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            running_loss = 0.0
            print(f" Batch {i + 1}, Loss: {last_loss}")
            if tb_writer is not None:
                tb_x = epoch_index * len(training_dataloader) + i
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            
    return last_loss

def load_data():
    # Load the training and validation datasets
    # Replace with your own data loading code
    pass

if __name__ == "__main__":
    model = AlexNetPerso()
    training_data, training_labels, validation_data, validation_labels = load_data()
    training_dataset = data.TensorDataset(training_data, training_labels)
    validation_dataset = data.TensorDataset(validation_data, validation_labels)
    training_dataloader = data.DataLoader(training_dataset, batch_size=32, shuffle=True)
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=32, shuffle=False)

    trainer(model, training_dataloader, validation_dataloader, 100, 0.001)