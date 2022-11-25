import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")


class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should
    override ``__len__``, that provides the size of the dataset,
    and ``__getitem__``, supporting integer indexing in range
    from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TOSDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.data1 = X
        self.data2 = Y
        self.transform = transform

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x = self.data1[index]
        y = self.data2[index]

        if self.transform is not None:
            x = torch.tensor(x)

        return torch.squeeze(x, dim=1), torch.tensor(y)


class TOSPredictDataset(Dataset):
    def __init__(self, X, transform=None):
        self.data1 = X
        self.transform = transform

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x = self.data1[index]

        if self.transform is not None:
            x = torch.tensor(x)

        return torch.squeeze(x, dim=1)


def create_dataloader(embeddings, BATCH_SIZE, NUM_WORKERS):
    test_data = TOSPredictDataset(embeddings, transform=transforms.ToTensor())
    return DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
