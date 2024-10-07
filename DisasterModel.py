import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DisasterModel(nn.Module):
    def __init__(self, num_classes=1, severity_classes=4):
        super(DisasterModel, self).__init__()
        
        # Shared ResNet Encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 6
        self.encoder.fc = nn.Identity()  # Remove fully connected layer
        
        # UNet Decoder for Segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1)
        )
        
        # Fully connected layers for Severity Prediction
        self.severity_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, severity_classes)  # Output severity level
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        # Forward pass through the shared ResNet encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x_flat = torch.flatten(x, 1)
        
        # Use the shared feature vector for segmentation and severity prediction
        seg_output = self.decoder(x)  # Segmentation mask
        seg_output = nn.functional.interpolate(seg_output, size=(256, 256), mode='bilinear', align_corners=True)
        severity_output = self.severity_fc(x_flat)  # Severity level
        
        return seg_output, severity_output
    
    def _initialize_weights(self):
      for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  
    
# import torchinfo

# model = DisasterModel()
# # torchinfo.summary(model)

# torchinfo.summary(model)
