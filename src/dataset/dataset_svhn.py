import torch as t
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os

from torch.utils.data import Dataset


class SvhnDataset(Dataset):
    """
    A wrapper around PyTorch's SVHN dataset implementation, making it useful for us
    """
    def __init__(self, train=True, normalize=True, subtract_mean=True, labels_as_logits=True, use_extra_train=False):
        """
        Make the PyTorch MNIST dataset instance and

        :param train: If this is a training dataset (else this is a test set)
        :param normalize: If samples should be normalized
        :param subtract_mean: If samples should be mean subtrracted (so that mean sample from dataset is zero).
        :param labels_as_logiits: If we should return lab
        :param use_extra_train: If we should combine the extra and training sets together
        """
        Dataset.__init__(self)
        self.train = train
        self.use_extra_train = use_extra_train
        self.labels_as_logits = labels_as_logits

        # Files for the data, and make sure that the path exists
        this_files_dir = os.path.dirname(__file__)
        svhn_dir = os.path.join(this_files_dir, "data", "svhn")
        train_file = os.path.join(svhn_dir, "training")
        val_file = os.path.join(svhn_dir, "test")
        extra_file = os.path.join(svhn_dir, "extra")

        if not os.path.exists(svhn_dir):
            os.makedirs(svhn_dir)

        # Apply torchvision transforms appropriately to be able to normalize and/or subtract means
        transs = [transforms.ToTensor()]
        if normalize:
            mean = 0.0 if subtract_mean else 0.5
            transs.append(transforms.Normalize((mean,), (1.0,)))
        trans = transforms.Compose(transs)

        # Finally make the SVHN instances needed
        self.data_file = train_file if train else val_file
        split = 'train' if train else 'test'
        self.dataset = dset.SVHN(root=self.data_file, split=split, transform=trans, download=True)
        if train and use_extra_train:
            self.extra_file = extra_file
            self.extra_dataset = dset.SVHN(root=self.extra_file, split='extra', transform=trans, download=True)

        # Stats needed for indexing into training set (split across the 'train' and 'extra' datasets)
        self.main_dataset_size = len(self.dataset)



    def _index_dataset(self, index):
        """ 
        Helper to index into the training set with extra training set. 
        """
        if self.train and self.use_extra_train and index >= self.main_dataset_size:
            index -= self.main_dataset_size
            return self.extra_dataset[index]
        return self.dataset[index]




    def __getitem__(self, index):
        """
        Gets item at 'index' from the dataset
        """
        if not self.labels_as_logits:
            return self._index_dataset(index)
        x, y = self._index_dataset(index)
        y_onehot = t.zeros((10,))
        y_onehot.scatter_(0, y, 1)
        return (x, y_onehot)



    def __len__(self):
        """
        Gets length of the dataset.
        """
        if self.train and self.use_extra_train:
            return len(self.dataset) + len(self.extra_dataset)
        return len(self.dataset)
