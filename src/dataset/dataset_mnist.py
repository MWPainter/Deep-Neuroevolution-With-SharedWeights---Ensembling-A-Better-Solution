import torchvision.datasets as dset
import torchvision.transforms as transforms

import os


class MnistDataset(dset.MNIST):
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
        mnist_dir = os.path.join(__file__, "data", "mnist")
        train_file = os.path.join(mnist_dir, "training.pt")
        val_file = os.path.join(mnist_dir, "test.pt")

        if not os.path.exists(mnist_dir):
            os.mkdir(mnist_dir)

        # If files dont exists, download them
        self.data_file = train_file if train else val_file
        download = not os.path.isfile(train_file) or not os.path.isfile(val_file)

        # Apply torchvision transforms appropriately to be able to normalize and/or subtract means
        # Also apply a transform to pad to (32,32) images, rather than (28,28), because power of two
        trans = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
        if normalize:
            mean = 0.0 if subtract_mean else 0.5
            trans = transforms.Compose([trans, transforms.Normalize((mean,), (1.0,))])

        # Finally make the MNIST instance
        self.dataset = dset.MNIST(root=self.data_file, train=train, transform=trans, download=download)



    def __getitem__(self, index):
        """
        Gets item at 'index' from the dataset
        """
        print(self.dataset[index])
        raise Exception("Need to debug and pad to make it (32,32) rather than (28,28)... and so on") # TODO: sanity check
        return self.dataset[index]



    def __len__(self):
        """
        Gets length of the dataset.
        """
        return len(self.dataset)
