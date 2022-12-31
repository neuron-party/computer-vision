class PANetPathAggregation(nn.Module):
    '''
    Path Aggregation portion of PANet to be used as the neck for object detection models
    Adaptive Feature Pooling, Box branch, and Fully-connected fusion left out
    Paper: https://arxiv.org/pdf/1803.01534.pdf
    '''
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        ResNet50 extracted feature maps are of channel shape 256, 512, 1024, 2048
        '''
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        P2 = self.backbone.layer1(x)
        P3 = self.backbone.layer2(P2)
        P4 = self.backbone.layer3(P3)
        P5 = self.backbone.layer4(P4)

        N2 = P2 # [b, 256, h, w]
        x = self.conv1(N2)
        x = x + P3
        N3 = self.relu(self.conv2(x)) # [b, 256, h/2, w/2]

        x = self.conv3(N3)
        x = x + P4
        N4 = self.relu(self.conv4(x)) # [b, 256, h/4, w/4]

        x = self.conv5(N4)
        x = x + P5
        N5 = self.relu(self.conv6(x)) # [b, 256, h/8, w/8]

        return N2, N3, N4, N5