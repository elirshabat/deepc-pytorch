from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import os.path
import PIL.Image
import matplotlib.pyplot as plt


def show_sample(sample):
    plt.figure()
    plt.imshow(sample['image'])
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
        labels = self._anns_to_ndarray(coco_img, coco_anns)
        image_path = os.path.join(self._images_dir, coco_img['file_name'])
        image = np.array(PIL.Image.open(image_path, 'r'))

        sample = {'image': image, 'labels': labels}
        if self._transform:
            sample = self._transform(sample)

        return sample

    def _anns_to_ndarray(self, img, anns):
        """
        Convert coco annotations to numpy array that represent labels.
        :param img: Image dictionary in coco format.
        :param anns: Annotations of the given image.
        :return: ndarray.
        """
        image_size = (img['height'], img['width'])
        labels = np.zeros(image_size)
        for i in range(len(anns)):
            ann = anns[i]
            label_mask = self._coco.annToMask(ann) == 1
            new_label = i + 1
            labels[label_mask] = new_label
        return labels.astype('uint8')
