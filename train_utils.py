import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_optimizer(model, lr=1e-5):
    return optim.Adam(model.parameters(), lr=lr)

def get_loss_functions():
    segmentation_loss_fn = nn.BCEWithLogitsLoss()
    severity_loss_fn = nn.MSELoss()
    return segmentation_loss_fn, severity_loss_fn

def train_model(model, dataloader, num_epochs, target_severity, device=device):
    model = model.to(device)
    optimizer = get_optimizer(model)
    
    segmentation_loss_fn, severity_loss_fn = get_loss_functions()
    
    epoch_losses = []
    epoch_accuracies = []
    checkpoint_dir = 'models/'
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
          
        for pre_post_images, mask, severity_label in dataloader:
            pre_post_images = pre_post_images.to(device)
            mask = mask.to(device)
            # mask = nn.functional.avg_pool2d(mask, kernel_size=16).to(device)
            severity_label = severity_label.to(device)

            optimizer.zero_grad()

            # Forward pass
            seg_output, severity_output = model(pre_post_images)
            
            seg_output = nn.functional.interpolate(seg_output, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)
            
            # Compute the losses
            seg_loss = segmentation_loss_fn(seg_output, mask.unsqueeze(1).float())
            severity_label = severity_label.unsqueeze(1)
            severity_loss = severity_loss_fn(severity_output, severity_label.float())
            
            print(severity_output.shape)
            print(severity_label.shape)
            
            # Total loss
            loss = seg_loss + 0.1 * severity_loss

            # Backward pass and optimization
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            # _, preds = torch.max(seg_output, 1)
            preds = torch.sigmoid(seg_output)>0.5
            
            running_corrects += (preds == mask.unsqueeze(1)).sum().item()
            
            if len(dataloader) > 0: 
                epoch_loss = running_loss / len(dataloader)
                epoch_accuracy = running_corrects / (len(dataloader.dataset)*mask.size(1) * mask.size(2))
                
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy} ')
            else:
                print("Dataloader is empty.")
        
        if epoch % 5 == 0:  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
    return epoch_losses, epoch_accuracies