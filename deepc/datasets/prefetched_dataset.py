from torch.utils.data import Dataset


class PrefetchedDataset(Dataset):

    def __init__(self, origin):
        super().__init__()
        self._origin = origin
        self._data = [origin.__getitem__(i) for i in range(len(origin))]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)
