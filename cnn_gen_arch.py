import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class ECGCNN(nn.Module):
    def __init__(self, output_labels = 10, num_channels=2, features = False):
        super(ECGCNN, self).__init__()
        self.output_labels = output_labels
        self.num_channels = num_channels
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.features = features
        
        # Pooling layer
        if(self.features):
            self.pool = nn.AdaptiveAvgPool2d((3, 3))
        else:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Adaptive pooling to make it size-independent
        self.adap_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        # After adaptive pooling, the output size is [batch_size, 256, 6, 6]
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.output_labels)  # Output layer for labels
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutions, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Apply adaptive pooling
        x = self.adap_pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 6 * 6)
        
        # Fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x



class CustomCNN(nn.Module):
    def __init__(self, map_width, map_height, output_labels=10, num_channels=2):
        super(CustomCNN, self).__init__()
        self.map_width = map_width
        self.map_height = map_height
        self.output_labels = output_labels
        self.num_channels = num_channels

        # Vertical features extraction path
        self.conv1 = nn.Conv2d(num_channels, 16, (3, self.map_width))  # Kernel size (3, map_width)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AdaptiveAvgPool2d((10, 1))  # Assuming reduction to 10x1

        # Horizontal features extraction path
        self.conv2 = nn.Conv2d(self.num_channels, 16, (self.map_height, 2))  # Kernel size (map_height, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 4))  # Assuming reduction to 1x4

        # Fully connected layers
        # Adjust the input dimensions of fc1 according to the output sizes from the pooling layers
        self.fc1 = nn.Linear(16 * 10 * 1 + 16 * 1 * 4, 120)  # Combine features from both paths
        self.fc2 = nn.Linear(120, 60)  # Second fully connected layer
        self.fc3 = nn.Linear(60, self.output_labels)  # Final output layer

    def forward(self, x):
        # Process vertical features
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.pool1(x1)
        x1 = x1.view(x1.size(0), -1)

        # Process horizontal features
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.pool2(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate features from both paths
        x = torch.cat((x1, x2), dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final layer does not necessarily need activation if used for classification with CrossEntropyLoss
        return x

      


        
    