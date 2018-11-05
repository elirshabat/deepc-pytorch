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
            # Compute batch information:
            X, L = data[batch_index, :, :, :].view(d, -1).permute(1, 0), labels[batch_index, :, :].view(-1)
            cluster_ids = L.unique()
            n_clusters = len(cluster_ids)

            # Handle the case of single instance (loss should be 0)
            if n_clusters <= 1:
                var_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                dist_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
                reg_terms.append(torch.tensor(0, device=data.device, dtype=torch.float))
            # Handle the case of more than one instance
            else:
                # Compute the point-cluster indicator (n_clusters x n_points)
                c_indicators = torch.stack([L == cluster_ids[c_index]
                                            for c_index in range(n_clusters)]).type(torch.float32)
                c_sizes = c_indicators.sum(1)

                # Compute the centers of the clusters (n_clusters x n_dims)
                centers_tensor = torch.matmul(c_indicators, X) / c_sizes.unsqueeze(1)

                # Compute the variance term of single example out of batch of examples
                distance_from_centers = (X.unsqueeze(0) - centers_tensor.unsqueeze(1)).norm(dim=2)
                distance_from_center_errors = (distance_from_centers - self._delta_var).clamp(0) ** 2
                batch_var_terms = (c_indicators * distance_from_center_errors).sum(1) / c_sizes
                var_term = batch_var_terms.mean()

                # Compute the distance term of single example out of batch of examples
                dist_matrix = (centers_tensor.unsqueeze(0) - centers_tensor.unsqueeze(1)).norm(dim=2)
                dist_cost_matrix = ((self._delta_dist - (
                            dist_matrix + torch.eye(n_clusters, device=data.device) * self._delta_dist)).clamp(0) ** 2)
                dist_term = dist_cost_matrix.sum() / (n_clusters * (n_clusters - 1))

                # Compute the regularization term of single example out of batch of examples
                reg_term = ((centers_tensor.norm(dim=1) - delta_reg).clamp(0) ** 2).mean()

                # Save the computed terms for later use
                var_terms.append(var_term)
                dist_terms.append(dist_term)
                reg_terms.append(reg_term)

        loss = (self._var_weight * torch.stack(var_terms).mean()
                + self._dist_weight * torch.stack(dist_terms).mean()
                + self._reg_weight * torch.stack(reg_terms).mean())

        return loss


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
