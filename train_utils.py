import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)

def get_loss_functions():
    segmentation_loss_fn = nn.BCEWithLogitsLoss()  # For binary segmentation
    severity_loss_fn = nn.MSELoss()  # For regression severity level
    return segmentation_loss_fn, severity_loss_fn

def train_model(model, dataloader, num_epochs, target_severity, device=device):
    model = model.to(device)
    optimizer = get_optimizer(model)
    segmentation_loss_fn, severity_loss_fn = get_loss_functions()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for pre_post_images, mask, severity_label in dataloader:
            pre_post_images = pre_post_images.to(device)
            mask = mask.to(device)
            severity_label = severity_label.to(device)

            optimizer.zero_grad()

            # Forward pass
            seg_output, severity_output = model(pre_post_images)

            # Compute the losses
            seg_loss = segmentation_loss_fn(seg_output, mask.unsqueeze(1).float())
            severity_loss = severity_loss_fn(severity_output, severity_label.float())

            # Total loss
            loss = seg_loss + severity_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
