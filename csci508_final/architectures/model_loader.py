import sys

def model_loader(model_dict, num_classes):
    if model_dict['resnet18']:
        from .resnet18 import Net
    elif model_dict['resnet34']:
        from .resnet34 import Net
    elif model_dict['resnet50']:
        from .resnet50 import Net
    elif model_dict['resnet50-modnet']:
        from .resnet50modnet import Net
    else:
        print("No model has been selected")
        sys.exit()

    model = Net(num_classes)

    return model