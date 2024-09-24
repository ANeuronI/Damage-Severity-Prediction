import os
import json
import pandas as pd

def determine_severity(label):
    label_data = json.loads(label)
    features = label_data['features']['xy']
    
    # Initialize damage counters
    destroyed = 0
    minor_damage = 0
    
    for feature in features:
        properties = feature['properties']
        
        # Check if 'subtype' key exists
        if 'subtype' in properties:
            if properties['subtype'] == 'destroyed':
                destroyed += 1
            elif properties['subtype'] == 'minor-damage':
                minor_damage += 1
    
    # Determine severity level based on damage counters
    if destroyed > 5 or minor_damage > 10:
        severity = 4  # High Damage
    elif destroyed > 0 or minor_damage > 5:
        severity = 3  # Moderate Damage
    elif minor_damage > 0:
        severity = 2  # Low Damage
    else:
        severity = 1  # No Damage
    
    return severity

def generate_severity_csv(label_dir, output_file):
    # Initialize severity data dictionary
    severity_data = {}
    
    # Iterate through label files
    for file in os.listdir(label_dir):
        if file.endswith('.json'):
            # Extract image ID from filename (assuming filename format: label_post_image_0.json)
            image_id = file.replace('label_post_', '').replace('.json', '')
            
            # Load label data
            with open(os.path.join(label_dir, file), 'r') as f:
                label = f.read()
            
            # Determine severity level
            try:
                severity = determine_severity(label)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                severity = 0  # Default severity level for error cases
            
            # Store severity level in dictionary
            severity_data[image_id] = severity
    
    # Create DataFrame from severity data
    severity_df = pd.DataFrame(list(severity_data.items()), columns=['image_id', 'severity_level'])
    
    # Save DataFrame to CSV file
    severity_df.to_csv(output_file, index=False)

# Define label directory and output CSV file
label_dir = 'augmented_data/augmented_labels'
output_file = 'severity.csv'

# Generate severity.csv file
generate_severity_csv(label_dir, output_file)