# # from https://github.com/wkentaro/pytorch-fcn
import torch.nn as nn
from torch.nn import functional as F

class FCN8s(nn.Module):
    def __init__(self, n_classes=4, n_frames=5):
        super(FCN8s, self).__init__()
        # lt@20240406 reduce resolution
        self.pool0 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv1
        # The 100 pixel input padding guarantees that the network output can be aligned to the input for any input size
        # since the kernel size of fc6 is 7 and its padding equals to 0.
        self.conv1_1 = nn.Conv2d(n_frames, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)

    def forward(self, x):
        # lt@20240406 reduce resolution
        x = self.pool0(x)
        # lt@20240408 add resolution
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        # crop for alignment
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        # crop
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        # crop
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        # lt@20240406 recover resolution
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        # lt@20240408 add resolution
        # h = F.interpolate(h, scale_factor=0.5, mode='bilinear', align_corners=True)
        return h
    
# from https://github.com/wkentaro/pytorch-fcn
# import torch.nn as nn

# class FCN8s(nn.Module):
#     def __init__(self, n_classes=4, n_frames=5):
#         super(FCN8s, self).__init__()
#         # conv1
#         # The 100 pixel input padding guarantees that the network output can be aligned to the input for any input size
#         # since the kernel size of fc6 is 7 and its padding equals to 0.
#         self.conv1_1 = nn.Conv2d(n_frames, 64, 3, padding=100)
#         self.bn1_1 = nn.BatchNorm2d(64)
#         self.relu1_1 = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.bn1_2 = nn.BatchNorm2d(64)
#         self.relu1_2 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

#         # conv2
#         self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn2_1 = nn.BatchNorm2d(128)
#         self.relu2_1 = nn.ReLU(inplace=True)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
#         self.bn2_2 = nn.BatchNorm2d(128)
#         self.relu2_2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

#         # conv3
#         self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn3_1 = nn.BatchNorm2d(256)
#         self.relu3_1 = nn.ReLU(inplace=True)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#         self.bn3_2 = nn.BatchNorm2d(256)
#         self.relu3_2 = nn.ReLU(inplace=True)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
#         self.bn3_3 = nn.BatchNorm2d(256)
#         self.relu3_3 = nn.ReLU(inplace=True)
#         self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

#         # conv4
#         self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#         self.bn4_1 = nn.BatchNorm2d(512)
#         self.relu4_1 = nn.ReLU(inplace=True)
#         self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bn4_2 = nn.BatchNorm2d(512)
#         self.relu4_2 = nn.ReLU(inplace=True)
#         self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bn4_3 = nn.BatchNorm2d(512)
#         self.relu4_3 = nn.ReLU(inplace=True)
#         self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

#         # conv5
#         self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bn5_1 = nn.BatchNorm2d(512)
#         self.relu5_1 = nn.ReLU(inplace=True)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bn5_2 = nn.BatchNorm2d(512)
#         self.relu5_2 = nn.ReLU(inplace=True)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bn5_3 = nn.BatchNorm2d(512)
#         self.relu5_3 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

#         # fc6
#         self.fc6 = nn.Conv2d(512, 4096, 7)
#         self.bn6 = nn.BatchNorm2d(4096)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.drop6 = nn.Dropout2d()

#         # fc7
#         self.fc7 = nn.Conv2d(4096, 4096, 1)
#         self.bn7 = nn.BatchNorm2d(4096)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.drop7 = nn.Dropout2d()

#         self.score_fr = nn.Conv2d(4096, n_classes, 1)
#         self.score_pool3 = nn.Conv2d(256, n_classes, 1)
#         self.score_pool4 = nn.Conv2d(512, n_classes, 1)

#         self.upscore2 = nn.ConvTranspose2d(
#             n_classes, n_classes, 4, stride=2, bias=False)
#         self.upscore8 = nn.ConvTranspose2d(
#             n_classes, n_classes, 16, stride=8, bias=False)
#         self.upscore_pool4 = nn.ConvTranspose2d(
#             n_classes, n_classes, 4, stride=2, bias=False)

#     def forward(self, x):
#         h = x
#         h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
#         h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
#         h = self.pool1(h)

#         h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
#         h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
#         h = self.pool2(h)

#         h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
#         h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
#         h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
#         h = self.pool3(h)
#         pool3 = h  # 1/8

#         h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
#         h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
#         h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
#         h = self.pool4(h)
#         pool4 = h  # 1/16

#         h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
#         h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
#         h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
#         h = self.pool5(h)

#         h = self.relu6(self.bn6(self.fc6(h)))
#         h = self.drop6(h)

#         h = self.relu7(self.bn7(self.fc7(h)))
#         h = self.drop7(h)

#         h = self.score_fr(h)
#         h = self.upscore2(h)
#         upscore2 = h  # 1/16

#         h = self.score_pool4(pool4)
#         # crop for alignment
#         h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
#         score_pool4c = h  # 1/16

#         h = upscore2 + score_pool4c  # 1/16
#         h = self.upscore_pool4(h)
#         upscore_pool4 = h  # 1/8

#         h = self.score_pool3(pool3)
#         # crop
#         h = h[:, :,
#               9:9 + upscore_pool4.size()[2],
#               9:9 + upscore_pool4.size()[3]]
#         score_pool3c = h  # 1/8

#         h = upscore_pool4 + score_pool3c  # 1/8

#         h = self.upscore8(h)
#         # crop
#         h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

#         return h