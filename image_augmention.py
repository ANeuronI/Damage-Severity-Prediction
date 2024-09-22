import os
import albumentations as A
from PIL import Image
import numpy as np
import json

# Define the augmentations

# Post-processing augmentations
post_augmentations = A.Compose([
    A.Flip(p=0.5),  # Randomly flip the image with a probability of 50%
    A.Rotate(p=0.5),  # Randomly rotate the image with a probability of 50%
    A.RandomScale(p=0.5, scale_limit=0.5),  # Randomly scale the image with a probability of 50% and a scale limit of 50%
    A.RandomBrightnessContrast(p=0.4),  # Randomly adjust brightness and contrast with a probability of 40%
], p=1)  # Apply all augmentations with a probability of 100%

# Both pre-processing and post-processing augmentations
both_augmentations = A.Compose([
    A.Flip(p=0.5),  # Randomly flip the image with a probability of 50%
    A.Rotate(p=0.5),  # Randomly rotate the image with a probability of 50%
    A.RandomScale(p=0.5, scale_limit=0.5),  # Randomly scale the image with a probability of 50% and a scale limit of 50%
    A.RandomBrightnessContrast(p=0.4),  # Randomly adjust brightness and contrast with a probability of 40%
], p=1)  # Apply all augmentations with a probability of 100%

# Color augmentations
color_augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.4),  # Randomly adjust brightness and contrast with a probability of 40%
    A.HueSaturationValue(p=0.4),  # Randomly adjust hue, saturation, and value with a probability of 40%
], p=1)  # Apply all augmentations with a probability of 100%


# Corrected paths
data_dir = 'Data/images'
label_dir = 'Data/labels'  # Directory containing the label JSONs
augmented_images_dir = 'augmented_data/augmented_images'
augmented_labels_dir = 'augmented_data/augmented_labels'

os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# Load the images and labels
images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
labels = [os.path.join(label_dir, f.replace('.png', '.json')) for f in os.listdir(data_dir) if f.endswith('.png')]

def wkt_to_keypoints(polygon_wkt):
    """
    Convert WKT polygon to a list of keypoints (x, y).
    Returns:
        list of tuple: A list of tuples where each tuple represents a keypoint (x, y).
    """
    polygon_coords = polygon_wkt.replace('POLYGON ((', '').replace('))', '').split(', ')
    keypoints = [(float(coord.split()[0]), float(coord.split()[1])) for coord in polygon_coords]
    return keypoints


def keypoints_to_wkt(keypoints):
    """
    Convert keypoints back to WKT polygon format.

    Args:
        keypoints (list of tuple): A list of tuples where each tuple contains 
                                   the x and y coordinates of a keypoint.

    Returns:
        str: A string representing the keypoints in WKT polygon format.
    """
    
    polygon_wkt = 'POLYGON ((' + ', '.join([f'{kp[0]} {kp[1]}' for kp in keypoints]) + '))'
    return polygon_wkt



def apply_augmentation_to_label(label_data, augmentation_fn, image_array):
    """
    Applies the augmentation to the label's polygon coordinates.

    Args:
        label_data (dict): The label data containing polygon coordinates in [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) format.
        augmentation_fn (callable): The augmentation function to apply to the keypoints.
        image_array (np.ndarray): The image array used to create a dummy image for augmentation.

    Returns:
        dict: The updated label data with augmented polygon coordinates.
    """
    
    for feature in label_data['features']['xy']:
        polygon_wkt = feature['wkt']
        keypoints = wkt_to_keypoints(polygon_wkt)
        
        # Create a dummy image with the same size as the keypoints
        dummy_image = np.zeros_like(image_array)
        
        # Add dummy angle and scale values to keypoints
        augmented_keypoints = []
        for keypoint in keypoints:
            x, y = keypoint
            augmented_keypoints.append((x, y, 0, 1))  # angle=0, scale=1
        
        # Apply the augmentation to the keypoints
        augmented = augmentation_fn(image=dummy_image, keypoints=augmented_keypoints)
        augmented_keypoints = augmented['keypoints']
        
        # Remove dummy angle and scale values from augmented keypoints
        updated_keypoints = [(x, y) for x, y, _, _ in augmented_keypoints]
        
        # Update the label with the augmented keypoints
        feature['wkt'] = keypoints_to_wkt(updated_keypoints)

    return label_data


for i, (image_path, label_path) in enumerate(zip(images, labels)):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    
    # Apply post augmentations
    post_image = post_augmentations(image=image_array)['image']
    
    # Apply both augmentations
    pre_image = both_augmentations(image=image_array)['image']
    post_image = both_augmentations(image=image_array)['image']
    
    # Apply color augmentations
    pre_image = color_augmentations(image=pre_image)['image']
    post_image = color_augmentations(image=post_image)['image']
    
    # Apply augmentations to labels
    pre_label_data = apply_augmentation_to_label(label_data.copy(), both_augmentations, image_array)
    post_label_data = apply_augmentation_to_label(label_data.copy(), post_augmentations, image_array)
    
    # Save the augmented images
    Image.fromarray(pre_image).save(os.path.join(augmented_images_dir, f'pre_augmented_image_{i}.png'))
    Image.fromarray(post_image).save(os.path.join(augmented_images_dir, f'post_augmented_image_{i}.png'))
    
    # Save the augmented labels
    with open(os.path.join(augmented_labels_dir, f'pre_augmented_label_{i}.json'), 'w') as f:
        json.dump(pre_label_data, f)
    with open(os.path.join(augmented_labels_dir, f'post_augmented_label_{i}.json'), 'w') as f:
        json.dump(post_label_data, f)

print(f"Processed and saved {len(images)} images and their corresponding labels.")