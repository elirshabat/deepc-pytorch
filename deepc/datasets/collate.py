import torch


def collate_fn(samples):
    pass
    # images, captions = zip(*data)
    #
    # # Merge images (from tuple of 3D tensor to 4D tensor).
    # images = torch.stack(images, 0)
    #
    # # Merge captions (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]
    # return images, targets, lengths