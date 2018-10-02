import torch
import numpy as np


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, var_weight=1.0, dist_weight=1.0, reg_weight=1.0,
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

        auto_delta_var = 1
        self._delta_var = delta_var if delta_var else auto_delta_var

        auto_delta_dist = 2
        self._delta_dist = delta_dist if delta_dist else auto_delta_dist

        self._delta_reg = delta_reg

        # self._var_weight = torch.tensor(var_weight, dtype=torch.float)
        # self._dist_weight = torch.tensor(dist_weight, dtype=torch.float)
        # self._reg_weight = torch.tensor(reg_weight, dtype=torch.float)
        self._var_weight = var_weight
        self._dist_weight = dist_weight
        self._reg_weight = reg_weight

    def forward(self, data, labels):
        """
        Compute the loss of the given data with respect to the labels.
        :param data: data points
        :param labels: ground truth of the clusters
        :return: the loss (scalar value)
        """
        batch_size, d, h, w = data.shape
        var_terms, dist_terms, reg_terms = [], [], []

        delta_reg = self._delta_reg if self._delta_reg is not None else np.sqrt(d)

        for batch_index in range(batch_size):
            X, L = data[batch_index, :, :, :].view(d, -1), labels[batch_index, :, :].view(-1)
            cluster_ids = L.unique()
            n_clusters = len(cluster_ids)
            centers = []

            if n_clusters <= 1:
                var_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                dist_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                reg_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
            else:
                batch_var_terms = []
                for c_index in range(n_clusters):
                    labels_mask = (L == cluster_ids[c_index])
                    pts = X[:, labels_mask]
                    c_center = pts.mean(1)
                    c_var = (((pts - c_center.unsqueeze(1)).norm(dim=0) - self._delta_var).clamp(0) ** 2).mean()
                    centers.append(c_center)
                    batch_var_terms.append(c_var)
                var_term = torch.stack(batch_var_terms).mean()

                centers_tensor = torch.stack(centers)
                dist_matrix = (centers_tensor.unsqueeze(0) - centers_tensor.unsqueeze(1)).norm(dim=2)
                dist_cost_matrix = ((self._delta_dist - (dist_matrix + torch.eye(n_clusters, device=data.device)*self._delta_dist)).clamp(0)**2)
                dist_term = dist_cost_matrix.sum()/(n_clusters*(n_clusters - 1))

                reg_term = ((centers_tensor.norm(dim=1) - delta_reg).clamp(0)**2).mean()

                var_terms.append(var_term)
                dist_terms.append(dist_term)
                reg_terms.append(reg_term)

        loss = (self._var_weight*torch.stack(var_terms).mean()
                + self._dist_weight*torch.stack(dist_terms).mean()
                + self._reg_weight*torch.stack(reg_terms).mean())

        return loss
