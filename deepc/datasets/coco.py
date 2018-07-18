from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from augmentations.resize import Resize
import yaml


def show_sample(sample, ignore_image=False, ignore_labels=False):
    """
    Show the given sample.
    :param sample: sample from the CocoDataset
    """
    plt.figure()
    if not ignore_image:
        plt.imshow(sample['image'])
    if not ignore_labels:
        plt.imshow(sample['labels'], alpha=0.5)


class CocoDataset(Dataset):

    def __init__(self, anns_file, images_dir, transform=None):
        self._anns_file = anns_file
        self._images_dir = images_dir
        self._transform = transform

        self._coco = COCO(anns_file)
        self._img_ids = self._coco.getImgIds()

    def __len__(self):
        """
        Get the number of samples in the dataset.
        :return: the number of samples in the dataset.
        """
        return len(self._img_ids)

    def __getitem__(self, item):
        """
        Get the sample that correspond to the given item.
        :param item: index of sample to return
        :return: sample of type dictionary with keys: 'image' and 'labels'
        """
        img_id = self._img_ids[item]
        coco_img = self._coco.imgs[img_id]
        coco_anns = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))

        image_path = os.path.join(self._images_dir, coco_img['file_name'])
        image = self.load_image(image_path)

        labels = self._anns_to_tensor(coco_img, coco_anns)

        sample = {'image': image, 'labels': labels}
        if self._transform:
            sample = self._transform(sample)

        return sample

    @staticmethod
    def load_image(file_path):
        """
        Load image from file to Tensor of type uint8.
        :param file_path: path to image file
        :return: Tensor that contains the image.
        """
        original_img = Image.open(file_path)
        original_np_img = np.array(original_img)
        original_torch_img = torch.tensor(original_np_img, dtype=torch.uint8)
        return original_torch_img

    def _anns_to_tensor(self, img, anns):
        """
        Convert coco annotations to numpy array that represent labels.
        :param img: Image dictionary in coco format.
        :param anns: Annotations of the given image.
        :return: Tensor.
        """
        image_size = (img['height'], img['width'])
        labels = np.zeros(image_size)
        for i in range(len(anns)):
            ann = anns[i]
            label_mask = self._coco.annToMask(ann) == 1
            new_label = i + 1
            labels[label_mask] = new_label
        return torch.tensor(labels.astype('uint8'), dtype=torch.uint8)


if __name__ == '__main__':

    paths_config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "Local", "paths.yaml")
    with open(paths_config_file_path, 'r') as f:
        paths = yaml.load(f)
        img_dir = paths['coco_train2014_images']
        anns_file = paths['coco_train2014_annotations']

    normal_dataset = CocoDataset(anns_file, img_dir)

    resized_dataset = CocoDataset(anns_file, img_dir, Resize(200, 300))

    items = [0, 3, 17]

    for i in items:
        show_sample(normal_dataset[i])
        show_sample(resized_dataset[i])

    plt.show()
