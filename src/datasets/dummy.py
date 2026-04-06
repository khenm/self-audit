import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Random tensor dataset for smoke-testing the training loop.

    Instantiated via Hydra ``_target_: src.datasets.dummy.DummyDataset``.
    """

    def __init__(self, num_samples: int = 1000, img_size=(3, 224, 224), num_classes: int = 10):
        self.num_samples = num_samples
        self.img_size = tuple(img_size)
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(*self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {"image": image, "label": label}
