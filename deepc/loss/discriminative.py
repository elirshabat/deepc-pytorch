import torch
import os.path
import yaml
from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
import numpy as np


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, dims, var_weight=1.0, dist_weight=1.0, reg_weight=1.0,
                 delta_var=None, delta_dist=None, delta_reg=None):
        """
        Initialize the loss function.
        :param delta_var: hinge value form the variance term
        :param delta_dist: hinge value form the distance term
        :param var_weight: wight of the variance term
        :param dist_weight: wight of the distance term
        :param reg_weight: wight of the regularization term
        """
        super().__init__()

        auto_delta_dist = 2
        auto_delta_var = 1
        auto_delta_reg = np.sqrt(dims)

        self._delta_var = delta_var if delta_var else auto_delta_var
        self._delta_dist = delta_dist if delta_dist else auto_delta_dist
        self._delta_reg = delta_reg if delta_reg else auto_delta_reg

        self._var_weight = var_weight
        self._dist_weight = dist_weight
        self._reg_weight = reg_weight

    def forward(self, data, labels, cluster_ids):
        """
        Compute the loss of the given data with respect to the labels.
        :param data: data points
        :param labels: ground truth of the clusters
        :return: the loss (scalar value)
        """
        loss_params = self._calc_loss_params(data, labels, cluster_ids)

        if loss_params['num_clusters'] <= 1:
            return 0

        var_term = self._calc_var_term(loss_params)
        dist_term = self._calc_dist_term(loss_params)
        reg_term = self._calc_reg_term(loss_params)

        return self._var_weight*var_term + self._dist_weight*dist_term + self._reg_weight*reg_term

    @staticmethod
    def _calc_loss_params(data, labels, cluster_ids):
        """
        Calculate parameters needed to compute the loss.
        :param data: input data points
        :param labels: input labels
        :return: loss parameters
        """
        out_params = dict()
        out_params['cluster_ids'] = cluster_ids
        out_params['num_clusters'] = len(out_params['cluster_ids'])
        out_params['cluster_params'] = dict()

        for c_id in out_params['cluster_ids']:
            c_params = dict()
            c_params['mask'] = labels == c_id
            c_params['size'] = c_params['mask'].sum().float()
            rows = c_params['mask'].nonzero()[:, 0]
            columns = c_params['mask'].nonzero()[:, 1]
            c_params['points'] = data[:, rows, columns]
            c_params['center'] = c_params['points'].sum(1) / c_params['size']

            out_params['cluster_params'][c_id] = c_params

        return out_params

    def _calc_var_term(self, loss_params):
        """
        Calculate the variance term of the loss.
        :param loss_params: parameters needed to compute the loss
        :return: scalar value that is the variance term of the loss
        """
        c_vars = []
        for c_id in loss_params['cluster_params']:
            c_params = loss_params['cluster_params'][c_id]
            center = c_params['center'].unsqueeze(1).repeat([1, c_params['points'].shape[1]])
            c_var = (((c_params['points'] - center).norm(dim=0) - self._delta_var).clamp(0)**2).sum()
            c_vars.append(c_var)

        return sum(c_vars)/loss_params['num_clusters']

    def _calc_dist_term(self, loss_params):
        """
        Calculate the distance term of the loss.
        :param loss_params: parameters needed to compute the loss
        :return: scalar value that is the distance term of the loss
        """
        centers = [loss_params['cluster_params'][c_id]['center'] for c_id in loss_params['cluster_params']]

        sum_dist = sum([(2.0*self._delta_dist - (centers[i] - centers[j]).norm()).clamp(0)**2
                        for i in range(len(centers))
                        for j in range(len(centers))
                        if i != j])

        return sum_dist/(loss_params['num_clusters']*(loss_params['num_clusters'] - 1))

    def _calc_reg_term(self, loss_params):
        """
        Calculate the regularization term of the loss.
        :param loss_params: parameters needed to compute the loss
        :return: scalar value that is the regularization term of the loss
        """
        sum_term = sum([(loss_params['cluster_params'][c_id]['center'].norm() - self._delta_reg).clamp(0)
                        for c_id in loss_params['cluster_params']])
        return sum_term/loss_params['num_clusters']


if __name__ == "__main__":
    paths_config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "Local", "paths.yaml")
    with open(paths_config_file_path, 'r') as f:
        paths = yaml.load(f)
        img_dir = paths['coco_train2014_images']
        anns_file = paths['coco_train2014_annotations']

    model = ResnetMIS(pretrained_resnet=True)
    loss_func = DiscriminativeLoss(1, 1, 2, 2, 1)

    dataset = CocoDataset(anns_file, img_dir, augmentations.Resize(240//2, 320//2))

    num_samples = 10

    for i in range(num_samples):
        sample = dataset[i]
        img = sample['image']
        labels = sample['labels']

        embedding = model(img.permute([2, 0, 1]).unsqueeze(0).float()).squeeze(0)
        loss = loss_func(embedding, labels)
        print(f"loss={loss}")
        model.zero_grad()
        loss.backward()

    print("Done")