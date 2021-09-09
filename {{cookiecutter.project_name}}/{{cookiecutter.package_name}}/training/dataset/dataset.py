from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        pass
