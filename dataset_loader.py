import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import pandas as pd

class DisasterDataset(Dataset):
    def __init__(self, image_dir, label_dir, severity_file, transform):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # Load severity levels from a CSV or JSON file
        if severity_file.endswith('.csv'):
            self.severity_data = pd.read_csv(severity_file, index_col=0)  # Assuming image names are in the first column
        elif severity_file.endswith('.json'):
            with open(severity_file, 'r') as f:
                self.severity_data = json.load(f)

        # List of pre-disaster images (assuming filenames start with 'pre')
        self.images = [file for file in os.listdir(image_dir) if file.startswith('pre')]

    def __len__(self):
        return len(self.images)

    def wkt_to_keypoints(self, polygon_wkt):
        """Convert WKT polygon to a list of keypoints (x, y)."""
        polygon_coords = polygon_wkt.replace('POLYGON ((', '').replace('))', '').split(', ')
        keypoints = [(float(coord.split()[0]), float(coord.split()[1])) for coord in polygon_coords]
        return keypoints
    
    def keypoints_to_wkt(self, keypoints):
        """
        Convert keypoints back to WKT polygon format.
        """
        polygon_wkt = 'POLYGON ((' + ', '.join([f'{kp[0]} {kp[1]}' for kp in keypoints]) + '))'
        return polygon_wkt

    def __getitem__(self, index):
        # Load pre and post-disaster image pairs
        pre_image_name = self.images[index]
        post_image_name = pre_image_name.replace('pre', 'post')

        # Load labels for post-disaster images
        post_label_name = pre_image_name.replace('pre', 'post').replace('.png', '').replace('image', 'label') + '.json'
        pre_image_path = os.path.join(self.image_dir, pre_image_name)
        post_image_path = os.path.join(self.image_dir, post_image_name)
        post_label_path = os.path.join(self.label_dir, post_label_name)

        # Open pre and post-disaster images
        pre_image = Image.open(pre_image_path)
        post_image = Image.open(post_image_path)

        # Load and convert label to binary mask
        with open(post_label_path, 'r') as f:
            post_label = json.load(f)

        mask = np.zeros((256, 256), dtype=np.uint8)
        for feature in post_label['features']['xy']:
            polygon_wkt = feature['wkt']
            keypoints = self.wkt_to_keypoints(polygon_wkt)
            polygon = np.array(keypoints, dtype=np.int32)
            cv2.drawContours(mask, [polygon], 0, 255, -1)

        # Load severity level
        image_id = pre_image_name.replace('pre_', '').replace('.png', '')
        if isinstance(self.severity_data, dict):
            severity = torch.tensor(self.severity_data[image_id], dtype=torch.float32)
        else:  # Assuming it's a DataFrame for CSV data
            severity = torch.tensor(self.severity_data.loc[image_id].values[0], dtype=torch.float32)

        # Apply transformations to pre and post images
        pre_image = self.transform(pre_image)
        post_image = self.transform(post_image)
        mask = torch.tensor(mask, dtype=torch.long)

        # Concatenate pre-disaster and post-disaster images along the channel dimension
        image = torch.cat((pre_image, post_image), dim=0)

        return image, mask, severity

# # Define data transforms
# data_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create dataset and data loader
# dataset = DisasterDataset('augmented_data/augmented_images', 'augmented_data/augmented_labels', 'augmented_data/severity.json', data_transform)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
