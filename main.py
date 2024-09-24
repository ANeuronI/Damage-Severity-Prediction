import torch
from torch.utils.data import DataLoader
from DisasterModel import DisasterModel
from dataset_loader import DisasterDataset
from train_utils import train_model
from torchvision import transforms

# Define data transforms
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset and dataloader
dataset = DisasterDataset('augmented_data/augmented_images', 'augmented_data/augmented_labels', 'severity_labels.csv', data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model
model = DisasterModel(num_classes=1, severity_classes=1)

# Train the model
train_model(model, dataloader, num_epochs=10, target_severity=None)
