import torch
from utils import *
import torch.nn as nn
from datetime import datetime
from models import AlexNetPerso
from preprocessing import preprocess, greyscale
from torch.utils.data import DataLoader, random_split

import os

def trainer(model, training_dataloader, validation_dataloader, num_epochs, criterion, optimizer, scheduler=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_loss = float('inf')

    last_saved_model_path = None

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        
        avg_loss, avg_acc = train_one_epoch(model, device, training_dataloader, criterion, optimizer)
        avg_vloss, avg_vacc = validation_one_epoch(model, device, validation_dataloader, criterion)

        if scheduler is not None:
            scheduler.step(avg_loss)

        print(f"Validation Loss: {avg_vloss:.4f}, Training Loss: {avg_loss:.4f}, Validation Accuracy: {avg_vacc:.4f}, Training Accuracy: {avg_acc:.4f}")
        training_loss.append(avg_loss)
        validation_loss.append(avg_vloss)
        training_accuracy.append(avg_acc)
        validation_accuracy.append(avg_vacc)

        best_loss, last_saved_model_path = save_model(model, timestamp, epoch, avg_vloss, best_loss, last_saved_model_path)

    plot_learning_curve(training_loss, validation_loss, training_accuracy, validation_accuracy)
        

def train_one_epoch(model, device, training_dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for i, data in enumerate(training_dataloader):
        inputs_image, inputs_speed, labels = data
        vinputs_image, vinputs_speed, vlabels = inputs_image.to(device), inputs_speed.to(device), labels.to(device)
        vinputs_speed = vinputs_speed.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(vinputs_image, vinputs_speed)

        loss = criterion(outputs, vlabels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        prediction = torch.sigmoid(outputs) > 0.5
        correct_preds += (prediction == vlabels).sum().item()
        total_preds += vlabels.numel()
    
    avg_loss = running_loss / len(training_dataloader)  
    avg_acc = correct_preds / total_preds

    return avg_loss, avg_acc

def validation_one_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total_preds = 0
    with torch.no_grad():
        for image, speed, target in test_loader:
            image, speed, target = image.to(device), speed.to(device), target.to(device)
            output = model(image, speed)

            test_loss += criterion(output, target)

            pred = torch.sigmoid(output) > 0.5
            correct += (pred == target).sum().item()
            total_preds += target.numel()
    test_loss /= len(test_loader.dataset)

    return test_loss, correct / total_preds

def save_model(model, timestamp, epoch, avg_vloss, best_loss, last_saved_model_path):
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

    return best_loss, last_saved_model_path

def plot_learning_curve(training_loss, validation_loss, training_accuracy, validation_accuracy):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(training_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = AlexNetPerso(4)
    model.to(device)

    print("Loading data...")

    if model.use_grayscale:
        transform_image = greyscale
    else:
        transform_image = preprocess

    dataset = CustomDataset("./data", transform_image=transform_image)
    training_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - training_size
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    training_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    pos_weight = dataset.get_distribution().sum() / dataset.get_distribution()
    pos_weight[1] = pos_weight[1] * 0.3

    print(pos_weight)
    
    class_weights = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    trainer(model=model, training_dataloader=training_dataloader, validation_dataloader=validation_dataloader, num_epochs=100, criterion=criterion, optimizer=optimizer, scheduler=scheduler)