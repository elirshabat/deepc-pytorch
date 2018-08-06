# import torch
#
#
# def collate_fn(samples):
#     stacked_images = torch.stack([s['image'] for s in samples])
#     stacked_labels = torch.stack([s['labels'] for s in samples])
#     max_clusters = max([len(s['cluster_ids']) for s in samples])
#     stacked_cluster_ids = torch.stack([torch.cat([s['cluster_ids'],
#                                                   -1*torch.ones(max_clusters - len(s['cluster_ids']),
#                                                                 device=s['cluster_ids'].device,
#                                                                 dtype=s['cluster_ids'].dtype)])
#                                        for s in samples])
#     return {'image': stacked_images, 'labels': stacked_labels, 'cluster_ids': stacked_cluster_ids}
