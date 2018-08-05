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



# stacked_images = torch.stack([s['image'] for s in samples])
# stacked_labels = torch.stack([s['labels'] for s in samples])
# max_clusters = max([len(s['cluster_ids']) for s in samples])
# stacked_cluster_ids = torch.stack([torch.cat([s['cluster_ids'], -1*torch.ones(max_clusters - len(s['cluster_ids']), device=s['cluster_ids'].device, dtype=s['cluster_ids'].dtype)]) for s in samples])
# out_sample = {'image': stacked_images, 'labels': stacked_labels, 'cluster_ids': stacked_cluster_ids}