import torch.nn as nn
from torchvision import models

def Net(num_classes):
    print("Loading in pretrained network ...")
    model = models.resnet18(pretrained=True)    # Get pretrained network
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    print("Number of inputs to final fully connected layer: %d" % num_features)

    # Change final layer.
    # Parameters of newly constructed modules have requires_grad=True by default.
    model.fc = nn.Linear(num_features, num_classes)

    return model
