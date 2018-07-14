# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import torch
#
#
# def resize(sample, h, w):
#
#     # transformations = transforms.Compose([transforms.ToPILImage(),
#     #                                       transforms.Resize((h, w), Image.NEAREST),
#     #                                       transforms.ToTensor()])
#
#     image_transformations = transforms.Compose([transforms.ToPILImage(),
#                                                 transforms.Resize((h, w)),
#                                                 np.array,
#                                                 torch.from_numpy])
#
#
#
#     labels_transformations = transforms.Compose([transforms.ToPILImage(),
#                                                  transforms.Resize((h, w), Image.NEAREST),
#                                                  np.array,
#                                                  torch.from_numpy])
#
#     sample = {'image': image_transformations(sample['image']),
#               'labels': labels_transformations(np.expand_dims(sample['labels'], 2))}
#
#     return sample
