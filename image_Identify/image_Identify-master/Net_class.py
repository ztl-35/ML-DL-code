import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            # 8 * (h * w = 308 * 513)
            nn.Conv2d(1, 8, kernel_size=5, stride=[3, 1], padding=2),
            nn.BatchNorm2d(8),
            nn.Dropout(),
            nn.ReLU(True),
            # 8 * 154 * 256
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            # 16 * 52 * 86
            nn.Conv2d(8, 16, kernel_size=4, stride=[3, 3], padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout(),
            nn.ReLU(True),
            # 16 * 26 * 43
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(17200, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
            nn.ReLU(True),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        input_fc = x.view(x.size(0), -1)
        output = self.fc(input_fc)
        return output
