
from torchvision import models
import torch
import cv2
import numpy as np

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)


items = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
         'googlenet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
         'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18',
         'resnet34', 'resnet50', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
         'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0',
         'squeezenet1_1', 'wide_resnet101_2', 'wide_resnet50_2']

for item in items:
    print(item)
    try:
        model = getattr(models, item)(pretrained=True)
    except (ValueError, NotImplementedError):
        print("No checkpoint is available for model type {}".format(item))
        continue

# for model in [resnet18, alexnet, squeezenet, vgg16, densenet, inception, googlenet, shufflenet, mobilenet, resnext50_32x4d, wide_resnet50_2, mnasnet]:
    name = model._get_name()
    layer = list(model.state_dict().keys())[0]
    weights = model.state_dict()[layer]
    print(name, layer, weights.shape)
    size = int(weights.shape[2])
    num_channels = int(weights.shape[0])
    s1 = 8
    s2 = num_channels//s1
    print(size, s1, s2)

    A = torch.empty((3,s1*size,s2*size), dtype=torch.float32)
    for i in range(s1):
        for j in range(s2):
            # print(name, i, j)
            # print(weights[s2*i+j].shape)
            A[:, size*i:size*(i+1), size*j:size*(j+1)] = weights[s2*i+j]

    B = A.numpy().T
    img = 255*(B-B.min())/(B.max()-B.min())
    img = img.astype(np.uint8)

    for i in range(1,s1):
        img = cv2.line(img, (size*i,0), (size*i,s2*size), (0,0,0), 1)
    for i in range(1,s2):
        img = cv2.line(img, (0,size*i), (s1*size,size*i), (0,0,0), 1)

    # cv2.imwrite("vis2/{}_layer1.jpg".format(name), img)
    cv2.imwrite("vis3/{}_layer1.jpg".format(item), img)



# models = [models.resnet18(pretrained=True),
#           models.resnext101_32x8d(pretrained=True),
#           models.densenet121(pretrained=True),
#           models.alexnet(pretrained=True),
#          ]
#
# layers = ["conv1.weight",
#           "conv1.weight",
#           "features.conv0.weight",
#           "features.0.weight",
#          ]
#
# names = ["vis/resnet18_layer1_test.jpg",
#          "vis/resnet101_layer1.jpg",
#          "vis/resnet121_layer1.jpg",
#          "vis/alexnet_layer1.jpg",
#         ]
#
# sizes = [7,7,7,11]
#
# for model, layer, size, name in zip(models, layers, sizes, names):
#     weights = model.state_dict()[layer]
#     A = torch.empty((3,8*size,8*size), dtype=torch.float32)
#     for i in range(8):
#         for j in range(8):
#             A[:, size*i:size*(i+1), size*j:size*(j+1)] = weights[8*i+j]
#
#     B = A.numpy().T
#     img = 255*(B-B.min())/(B.max()-B.min())
#     img = img.astype(np.uint8)
#
#     for i in range(1,8):
#         img = cv2.line(img, (size*i,0), (size*i,8*size), (0,0,0), 1)
#         img = cv2.line(img, (0,size*i), (8*size,size*i), (0,0,0), 1)
#
#     cv2.imwrite(name, img)
