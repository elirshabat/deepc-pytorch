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
        img = self.resize_tensor_image(sample['image'].type(torch.uint8)).type(sample['image'].dtype)
        labels = self.resize_tensor_image(sample['labels'])

        return {'image': img,
                'labels': labels,
                'cluster_ids': labels.unique().type(sample['cluster_ids'].dtype)}

    def resize_tensor_image(self, tensor):
        """
        Resize a uint8 tensor that represent an image.
        """
        original_np_img = np.array(tensor)
        img = Image.fromarray(original_np_img).resize((self._w, self._h), Image.NEAREST)
        np_img = np.array(img)
        return torch.tensor(np_img, dtype=torch.uint8)
