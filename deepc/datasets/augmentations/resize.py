import torch
import numpy as np
from PIL import Image


class Resize:

    def __init__(self, h, w):
        """
        Construct an instance.
        :param h: new height
        :param w: new width
        """
        self._h = h
        self._w = w

    def __call__(self, sample):
        """
        Resize the images in the given sample.
        :param sample: Sample returned from dataset.
        :return: The resized sample.
        """
        return {'image': self.resize_tensor_image(sample['image']),
                'labels': self.resize_tensor_image(sample['labels']),
                'cluster_ids': sample['cluster_ids']}

    def resize_tensor_image(self, tensor):
        """
        Resize a tensor that represent an image.
        """
        original_np_img = np.array(tensor)
        img = Image.fromarray(original_np_img).resize((self._w, self._h), Image.NEAREST)
        np_img = np.array(img)
        torch_img = torch.tensor(np_img, dtype=torch.uint8)
        return torch_img
