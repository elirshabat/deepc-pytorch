import cv2
import torch


class Resize:

    def __init__(self, h, w):
        self._h = h
        self._w = w

    def __call__(self, sample):
        np_image = cv2.resize(sample['image'], dsize=(self._h, self._w), interpolation=cv2.INTER_NEAREST)
        np_labels = cv2.resize(sample['labels'], dsize=(self._h, self._w), interpolation=cv2.INTER_NEAREST)

        return {'image': torch.from_numpy(np_image),
                'labels': torch.from_numpy(np_labels)}
