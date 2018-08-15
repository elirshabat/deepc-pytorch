import torch
import numpy as np


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, var_weight=1.0, dist_weight=1.0, reg_weight=1.0,
                 delta_var=None, delta_dist=None, cuda=True):
        """
        Initialize the loss function.
        :param delta_var: hinge value form the variance term
        :param delta_dist: hinge value form the distance term
        :param var_weight: wight of the variance term
        :param dist_weight: wight of the distance term
        :param reg_weight: wight of the regularization term
        """
        super().__init__()

        auto_delta_var = 1
        delta_var = delta_var if delta_var else auto_delta_var
        self._delta_var = torch.tensor(delta_var, dtype=torch.float)

        auto_delta_dist = 2
        delta_dist = delta_dist if delta_dist else auto_delta_dist
        self._delta_dist = torch.tensor(delta_dist, dtype=torch.float)

        self._var_weight = torch.tensor(var_weight, dtype=torch.float)
        self._dist_weight = torch.tensor(dist_weight, dtype=torch.float)
        self._reg_weight = torch.tensor(reg_weight, dtype=torch.float)

        if cuda:
            self._delta_var, self._delta_dist = self._delta_var.cuda(), self._delta_dist.cuda()
            self._var_weight = self._var_weight.cuda()
            self._dist_weight = self._dist_weight.cuda()
            self._reg_weight = self._reg_weight.cuda()

    def forward(self, data, labels):
        """
        Compute the loss of the given data with respect to the labels.
        :param data: data points
        :param labels: ground truth of the clusters
        :return: the loss (scalar value)
        """
        batch_size, d, h, w = data.shape
        var_terms, dist_terms, reg_terms = [], [], []

        for batch_index in range(batch_size):
            X, L = data[batch_index, :, :, :].view(d, -1), labels[batch_index, :, :].view(-1)
            cluster_ids = L.unique()
            k = len(cluster_ids)
            centers = []

            if k <= 1:
                var_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                dist_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                reg_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
            else:
                for c_index in range(k):
                    c_id = cluster_ids[c_index]
                    labels_mask = (L == c_id)
                    c_size = labels_mask.sum().float()
                    points_mask = labels_mask.unsqueeze(0).repeat([d, 1]).float()
                    c_points = X*points_mask
                    c_center = c_points.sum(1)/c_size
                    centers.append(c_center)
                    c_var = (((c_points - c_center.unsqueeze(1)*points_mask).norm(dim=0) - self._delta_var).clamp(0)**2).sum()/c_size
                    var_terms.append(c_var)

                centers_tensor = torch.stack(centers)
                for c_index in range(k):
                    dist_term = ((self._delta_dist - (centers[c_index] - centers_tensor).norm(dim=1)).clamp(0)**2/(k*(k - 1))).sum()/2
                    dist_terms.append(dist_term)

                reg_term = centers_tensor.norm(dim=1).mean()
                reg_terms.append(reg_term)

        loss = (self._var_weight*torch.stack(var_terms).sum()
                + self._dist_weight*torch.stack(dist_terms).sum()
                + self._reg_weight*torch.stack(reg_terms).sum())/batch_size

        return loss
