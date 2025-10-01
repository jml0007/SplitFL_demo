from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: Dataset, transform_simclr):
        self.base_dataset = base_dataset
        self.transform_simclr = transform_simclr
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return self.transform_simclr(img), self.transform_simclr(img)

def get_transforms(image_size: int):
    base = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    simclr = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(), T.Normalize((0.1307,), (0.3081,)),
    ])
    return base, simclr

def make_dataloaders(data_root: str, image_size: int, batch_size: int, num_workers: int, iid: bool, num_clients: int):
    base_tf, simclr_tf = get_transforms(image_size)
    # Public dataset for SimCLR (MNIST)
    public_base = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=None)
    public_simclr = SimCLRDataset(public_base, transform_simclr=simclr_tf)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=base_tf)

    full = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=base_tf)
    n = len(full)
    idxs = torch.randperm(n).tolist()
    shards = [idxs[i::num_clients] for i in range(num_clients)]
    client_sets = [Subset(full, s) for s in shards]
    client_loaders = [DataLoader(cs, batch_size=batch_size, shuffle=True, num_workers=num_workers) for cs in client_sets]

    public_loader = DataLoader(public_simclr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return client_sets, client_loaders, public_loader, test_loader
