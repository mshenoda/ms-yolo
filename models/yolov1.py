from torch import nn
from torchvision.models import resnet34


class YoloTiny(nn.Module):
    def __init__(self, S, B, num_classes):
        super(YoloTiny, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # conv part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(9216, 1470),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1470, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out


class YOLOResNet(nn.Module):
    """YOLO-Resnet model structure
    yolo-v1 resnet = resnet(backbone) + conv + fc
    """

    def __init__(self, S, B, num_classes):
        super(YOLOResNet, self).__init__()
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
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out
