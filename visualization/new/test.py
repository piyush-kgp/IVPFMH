
from torchvision import models
import torch
import cv2
import numpy as np

items = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
         'googlenet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
         'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
         'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
         'squeezenet1_0', 'squeezenet1_1', 'wide_resnet101_2', 'wide_resnet50_2']

for item in items:
    print(item)
    try:
        model = getattr(models, item)(pretrained=True)
    except (ValueError, NotImplementedError):
        print("No checkpoint is available for model type {}".format(item))
        continue
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

    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    cv2.imwrite("{}_layer1.jpg".format(item), img)

    img = np.full((512,512,3), 255, dtype=np.uint8)
    cv2.putText(img, item, (5,256), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imwrite("{}_layer0.jpg".format(item), img)
