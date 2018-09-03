from torch.utils.data import Dataset


class GradualDataset(Dataset):

    def __init__(self, origin, start_len):
        super().__init__()
        self._origin = origin
        self.len = start_len

    def __getitem__(self, index):
        return self._origin.__getitem__(index)

    def __len__(self):
        if len(self._origin) > self.len:
            return self.len
        else:
            return len(self._origin)
