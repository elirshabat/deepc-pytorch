import torch
from torchvision import models
import yaml
import os.path
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
import torch.nn.init as init
import numpy as np


def initialize_weights(method='kaiming', *models):
    for model in models:
        for module in model.modules():

            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d) or isinstance(module, torch.nn.Linear):
                if method == 'kaiming':
                    init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'xavier':
                    init.xavier_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'orthogonal':
                    init.orthogonal(module.weight.data, np.sqrt(2.0))
                elif method == 'normal':
                    init.normal(module.weight.data,mean=0, std=0.02)
                if module.bias is not None:
                    init.constant_(module.bias.data,0)


class GlobalConvolutionBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        super(GlobalConvolutionBlock, self).__init__()

        self.left = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(k[0],1), padding=(k[0]//2,0)),
                                  torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1,k[1]), padding=(0,k[1]//2)))

        self.right = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1,k[1]), padding=(0,k[1]//2)),
                                   torch.nn.Conv2d(out_channels, out_channels, kernel_size=(k[0],1), padding=(k[0]//2,0)))

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return left + right


class BoundaryRefine(torch.nn.Module):
    def __init__(self, in_channels):
        super(BoundaryRefine, self).__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                         torch.nn.BatchNorm2d(in_channels),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                         torch.nn.BatchNorm2d(in_channels))

    def forward(self, x):
        convs = self.layer(x)
        return x.expand_as(convs)+convs


class ResnetMIS(torch.nn.Module):

    def __init__(self, pretrained_resnet=True, out_channels=3):
        super().__init__()

        resent = models.resnet101(pretrained=pretrained_resnet)
        self.layer0 = torch.nn.Sequential(resent.conv1, resent.bn1, resent.relu, resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4

        # Assuming input of size 240x320
        self.gcn256 = GlobalConvolutionBlock(256, out_channels, (59, 79))
        self.br256 = BoundaryRefine(out_channels)
        self.gcn512 = GlobalConvolutionBlock(512, out_channels, (29, 39))
        self.br512 = BoundaryRefine(out_channels)
        self.gcn1024 = GlobalConvolutionBlock(1024, out_channels, (13, 19))
        self.br1024 = BoundaryRefine(out_channels)
        self.gcn2048 = GlobalConvolutionBlock(2048, out_channels, (7, 9))
        self.br2048 = BoundaryRefine(out_channels)

        self.br1 = BoundaryRefine(out_channels)
        self.br2 = BoundaryRefine(out_channels)
        self.br3 = BoundaryRefine(out_channels)
        self.br4 = BoundaryRefine(out_channels)
        self.br5 = BoundaryRefine(out_channels)

        self.activation = torch.nn.Sigmoid()

        self.deconv1 = torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.deconv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)

        initialize_weights(self.gcn256, self.gcn512, self.gcn1024, self.gcn2048,
                           self.br5, self.br4, self.br3, self.br2, self.br1,
                           self.br256, self.br512, self.br1024, self.br2048,
                           self.deconv1, self.deconv2)

    def forward(self, x):
        x = self.layer0(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        enc1 = self.br256(self.gcn256(layer1))
        enc2 = self.br512(self.gcn512(layer2))
        enc3 = self.br1024(self.gcn1024(layer3))
        enc4 = self.br2048(self.gcn2048(layer4))

        dec1 = self.br1(torch.nn.functional.upsample(enc4, size=enc3.size()[2:], mode='bilinear') + enc3)
        dec2 = self.br2(torch.nn.functional.upsample(dec1, enc2.size()[2:], mode='bilinear') + enc2)
        dec3 = self.br3(torch.nn.functional.upsample(dec2, enc1.size()[2:], mode='bilinear') + enc1)
        dec4 = self.br4(self.deconv1(dec3))

        score_map = self.br5(self.deconv2(dec4))

        return self.activation(score_map)


if __name__ == "__main__":
    paths_config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "Local", "paths.yaml")
    with open(paths_config_file_path, 'r') as f:
        paths = yaml.load(f)
        img_dir = paths['coco_train2014_images']
        anns_file = paths['coco_train2014_annotations']

    model = ResnetMIS(pretrained_resnet=True)

    dataset = CocoDataset(anns_file, img_dir, augmentations.Resize(240, 320))

    sample = dataset[0]
    img = sample['image']

    embedding = model(img.permute([2, 0, 1]).unsqueeze(0).float())
    print(embedding.shape)
