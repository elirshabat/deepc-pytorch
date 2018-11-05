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
        # Data info
        batch_size, d, h, w = data.shape
        device = data.device
        delta_reg = self._delta_reg if self._delta_reg is not None else np.sqrt(d)

        # Tensor of size (batch_size x d x n_points)
        data = data.view(data.shape[0], data.shape[1], -1)

        # Tensor of size (batch_size x n_points)
        labels = labels.view(labels.shape[0], -1)

        # Create cluster indicators
        c_indicators_list = []
        for batch_index in range(batch_size):
            b_labels = labels[batch_index, :]
            b_cluster_ids = b_labels.unique()
            b_n_clusters = len(b_cluster_ids)
            b_c_indicators = torch.stack([b_labels == b_cluster_ids[c_index] for c_index in range(b_n_clusters)])
            c_indicators_list.append(b_c_indicators)
        clusters_dim = max([ci.shape[0] for ci in c_indicators_list])
        c_indicators_list_padded = [
            torch.cat([ci, torch.zeros([clusters_dim - ci.shape[0], ci.shape[1]], dtype=torch.uint8, device=device)],
                      dim=0) for ci in c_indicators_list]

        # Tensor of size (batch_size x clusters_dim x n_points)
        c_indicators = torch.stack(c_indicators_list_padded).type(torch.float32)
        c_sizes = c_indicators.sum(2)
        real_cluster = (c_sizes != 0).type(torch.float32)
        dummy_cluster = (c_sizes == 0).type(torch.float32)
        n_clusters = real_cluster.sum(1)
        cluster_pairs = (n_clusters * (n_clusters - 1)) / 2
        one_over_c_sizes = (c_sizes != 0).type(torch.float32) / (c_sizes + (c_sizes == 0).type(torch.float32))

        # Compute the centers tensor
        centers = (data.unsqueeze(1) * c_indicators.unsqueeze(2)).sum(3) * one_over_c_sizes.unsqueeze(2)

        # Variance term
        distance_from_centers = (data.permute([0, 2, 1]).unsqueeze(1) - centers.unsqueeze(2)).norm(dim=3)
        distance_from_center_errors = (distance_from_centers - self._delta_var).clamp(0) ** 2
        var_terms = (c_indicators * distance_from_center_errors).sum(2) * one_over_c_sizes
        var_term = (var_terms.sum(1) / n_clusters).mean()

        # Distance term
        dist_matrix = (centers.unsqueeze(1) - centers.unsqueeze(2)).norm(dim=3)

        dist_cost_matrix = (self._delta_dist - (dist_matrix
                                                + torch.eye(clusters_dim, device=data.device) * self._delta_dist
                                                + dummy_cluster.unsqueeze(1) * self._delta_dist
                                                + dummy_cluster.unsqueeze(2) * self._delta_dist)).clamp(0) ** 2
        # dist_term = (dist_cost_matrix.sum() / 2) / cluster_pairs
        dist_term = ((dist_cost_matrix.sum(2).sum(1) / 2) / cluster_pairs).mean()

        # Regularization term
        reg_term = (((centers.norm(dim=2) - delta_reg).clamp(0) ** 2).sum(1) / n_clusters).mean()

        # Final value
        return self._var_weight * var_term + self._dist_weight * dist_term + self._reg_weight * reg_term


def _main():
    cuda_available = (torch.cuda.device_count() > 0)
    device = torch.device('cuda') if cuda_available else torch.device('cpu')
    x = torch.rand([64, 5, 60, 80], dtype=torch.float32).to(device)
    y = torch.randint(0, 10, [64, 60, 80], dtype=torch.uint8).to(device)
    criterion = DiscriminativeLoss()
    loss_value = criterion(x, y)
    print(loss_value)


if __name__ == '__main__':
    _main()
