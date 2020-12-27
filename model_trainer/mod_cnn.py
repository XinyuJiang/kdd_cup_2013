import torch
import torch.nn as nn
import config


class CNN(nn.Module):
    def __init__(self, num_classes=config.lable_num):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # 1
            # nn.Conv1d(in_channels=19, out_channels=17, kernel_size=1),
            # nn.BatchNorm1d(17),
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3),
            nn.BatchNorm1d(10),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            # 2
            # nn.Linear(17, 17),
            nn.Linear(170, 170),
            nn.ReLU(True),
            nn.Dropout(),
            # 3
            nn.Linear(170, num_classes),
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out