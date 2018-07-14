import torchvision.models as models


def resnet():
    """
    Get pre-trained ResNet module.
    :return: Pre-trained ResNet module
    """
    return models.resnet152(pretrained=True)
