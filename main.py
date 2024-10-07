import torch
from torch.utils.data import DataLoader
from DisasterModel import DisasterModel
from dataset_loader import DisasterDataset
from train_utils import train_model
from torchvision import transforms
import matplotlib.pyplot as plt

# Define data transforms
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset and dataloader
# dataset = DisasterDataset('augmented_data/augmented_images', 'augmented_data/augmented_labels', 'severity_labels.csv', data_transform)
dataset = DisasterDataset('Data/images', 'Data/labels', 'severity_labels.csv', data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model
model = DisasterModel(num_classes=1, severity_classes=4)

# Train the model
try:
    epoch_losses, epoch_accuracies = train_model(model, dataloader, num_epochs=10, target_severity=None)
    
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(epoch_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    torch.save(model.state_dict(), "models/interrupted_model.pth")