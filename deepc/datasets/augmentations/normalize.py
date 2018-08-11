class Normalize:

    def __init__(self, reverse=False):
        self._reverse = reverse

    def __call__(self, sample):
        img = sample['image']*255 if self._reverse else sample['image']/255
        return {'image': img, 'labels': sample['labels']}
