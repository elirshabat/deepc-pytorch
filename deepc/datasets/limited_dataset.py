from torch.utils.data import Dataset


class LimitedDataset(Dataset):

    def __init__(self, origin, length):
        super().__init__()
        self._origin = origin
        self._length = min(length, len(origin))

    def __getitem__(self, index):
        return self._origin.__getitem__(index)

    def __len__(self):
        return self._length
