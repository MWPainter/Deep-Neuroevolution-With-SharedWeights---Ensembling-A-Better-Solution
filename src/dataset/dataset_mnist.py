import torch as t
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os

from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """
    A wrapper around PyTorch's MNIST dataset implementation, making it useful for us
    """
    def __init__(self, train=True, normalize=True, subtract_mean=True):
        """
        Make the PyTorch MNIST dataset instance and

        :param train: If this is a training dataset (else this is a test set)
        :param normalize: If samples should be normalized
        :param subtract_mean: If samples should be mean subtrracted (so that mean sample from dataset is zero).
        """
        Dataset.__init__(self)

        # Files for the data, and make sure that the path exists
        this_files_dir = os.path.dirname(__file__)
        mnist_dir = os.path.join(this_files_dir, "data", "mnist")
        train_file = os.path.join(mnist_dir, "training.pt")
        val_file = os.path.join(mnist_dir, "test.pt")

        if not os.path.exists(mnist_dir):
            os.makedirs(mnist_dir)

        # If files dont exists, download them
        self.data_file = train_file if train else val_file
        download = not os.path.isdir(train_file) or not os.path.isdir(val_file)

        # Apply torchvision transforms appropriately to be able to normalize and/or subtract means
        # Also apply a transform to pad to (32,32) images, rather than (28,28), because power of two
        transs = [transforms.Pad(2), transforms.ToTensor()]
        if normalize:
            mean = 0.0 if subtract_mean else 0.5
            transs.append(transforms.Normalize((mean,), (1.0,)))
        trans = transforms.Compose(transs)

        # Finally make the MNIST instance
        self.dataset = dset.MNIST(root=self.data_file, train=train, transform=trans, download=download)



    def __getitem__(self, index):
        """
        Gets item at 'index' from the dataset
        """
        x, y = self.dataset[index]
        y_onehot = t.zeros((10,))
        y_onehot.scatter_(0, y, 1)
        return (x, y_onehot)



    def __len__(self):
        """
        Gets length of the dataset.
        """
        return 320
        return len(self.dataset)
