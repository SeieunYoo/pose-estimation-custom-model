import torch
import torch.nn as nn
import torch.optim as optim

class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=17, num_paf=38, num_stages=6):
        super(PoseEstimationModel, self).__init__()
        
        # 입력받은 num_keypoints와 num_paf를 클래스 변수로 저장
        self.num_keypoints = num_keypoints
        self.num_paf = num_paf

        # Initial convolution layers (VGG-like backbone)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        # 1x1 convolution to reduce the channels of the residual
        self.residual_reduce = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)

        # CPM stages for keypoints and PAF
        self.stages = nn.ModuleList()
        first_stage_channels = 512 + 128  # first stage gets the output from conv4_2 + residual
        self.stages.append(self._make_stage(first_stage_channels))

        # Following stages get the output of the previous stage + residual
        next_stage_channels = num_keypoints + num_paf + 128
        for _ in range(1, num_stages):
            self.stages.append(self._make_stage(next_stage_channels))

    def _make_stage(self, input_channels):
        layers = [
            nn.Conv2d(input_channels, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_keypoints + self.num_paf, kernel_size=1, stride=1, padding=0)  # Output layer
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and pooling layers (Backbone)
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))

        # Adding residual connections, using 1x1 convolution to reduce channels
        residual = self.residual_reduce(x)

        # Apply CPM stages
        for stage in self.stages:
            concat_input = torch.cat([x, residual], dim=1)
            x = stage(concat_input)

        # Final output layers
        keypoint_output = x[:, :self.num_keypoints, :, :]  # keypoints
        paf_output = x[:, self.num_keypoints:, :, :]  # PAFs

        return keypoint_output, paf_output


# 모델 생성, 손실 함수, 옵티마이저 설정
model = PoseEstimationModel(num_keypoints=17, num_paf=38, num_stages=6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
