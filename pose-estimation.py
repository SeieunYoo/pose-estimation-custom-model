import torch.nn as nn
import torch.optim as optim

class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseEstimationModel, self).__init__()
        
        # CNN Layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Adaptive Pooling to automatically handle different input sizes
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Flattened size of feature maps after pooling
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Adjusted based on pooling output size
        self.fc2 = nn.Linear(1024, num_keypoints * 2)  # Output for x, y coordinates for each keypoint
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Adaptive Pooling to reduce feature map size
        x = self.pool(x)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        keypoints = self.fc2(x)
        
        return keypoints.view(x.size(0), -1, 2)  # Reshape to (batch_size, num_keypoints, 2)

# 모델 생성 및 학습 준비
model = PoseEstimationModel(num_keypoints=17)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
