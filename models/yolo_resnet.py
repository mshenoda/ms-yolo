from torch import nn
from torchvision.models import resnet34

class YoloResNet(nn.Module):
    def __init__(self, S, B, num_classes):
        super(YoloResNet, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        # self.resnet = resnet18()
        self.resnet = resnet34()
        # print(self.resnet.fc.in_features)
        # print(*list(self.resnet.children())[-2:])  # show last two layers

        # backbone part, (cut resnet's last two layers)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        # conv part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # nn.Conv2d(1024, 1024, 3, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.ReLU()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out
