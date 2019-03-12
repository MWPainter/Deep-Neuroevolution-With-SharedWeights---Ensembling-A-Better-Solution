import torch as t
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os

from torch.utils.data import Dataset


class CifarDataset(Dataset):
    """
    A wrapper around PyTorch's CIFAR dataset implementation, making it useful for us
    """
    def __init__(self, train=True, normalize=True, subtract_mean=True, labels_as_logits=True, augment_training=True):
        """

        :param train: If this is a training dataset (else this is a test set)
        :param normalize: If samples should be normalized
        :param subtract_mean: If samples should be mean subtrracted (so that mean sample from dataset is zero).
        :param labels_as_logiits: If we should return lab
        :param augment_training: If we should augment training data
        """
        Dataset.__init__(self)
        self.labels_as_logits = labels_as_logits

        # Files for the data, and make sure that the path exists
        this_files_dir = os.path.dirname(__file__)
        cifar_dir = os.path.join(this_files_dir, "data", "cifar")
        if not os.path.exists(cifar_dir):
            os.makedirs(cifar_dir)
        train_dir = os.path.join(cifar_dir, "train")
        val_dir = os.path.join(cifar_dir, "test")

        # Apply torchvision transforms appropriately to be able to normalize and/or subtract means
        transs = []
        if train and augment_training:
            transs.append(transforms.RandomCrop(32, padding=4))
            transs.append(transforms.RandomHorizontalFlip())
        transs.append(transforms.ToTensor())
        if normalize:
            mean = [0.507, 0.487, 0.441] if subtract_mean else [0.0, 0.0, 0.0]
            transs.append(transforms.Normalize(mean=mean, std=[0.267, 0.256, 0.276]))
        trans = transforms.Compose(transs)

        # Finally make the SVHN instances needed
        self.data_dir = train_dir if train else val_dir
        self.dataset = dset.CIFAR10(root=self.data_dir, train=train, transform=trans, download=True)




    def __getitem__(self, index):
        """
        Gets item at 'index' from the dataset
        """
        if not self.labels_as_logits:
            return self.dataset[index]
        x, y = self.dataset[index]
        y_onehot = t.zeros((10,))
        y_onehot.scatter_(0, y, 1)
        return (x, y_onehot)



    def __len__(self):
        """
        Gets length of the dataset.
        """
        return len(self.dataset)

















# from __future__ import print_function

# from builtins import range
# from six.moves import cPickle as pickle
# import numpy as np
# import os
# import platform
# import torch as t

# from torch.utils.data import Dataset






# """
# Code adapted from Stanford's CS231N assignments starter code. (This doesn't contain any solutions at all). 
# You need to run the get_cifar shell script in the data directory first, and this code assumes that you have done that.
# """





# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'cifar10'))



# def load_pickle(f):
#     version = platform.python_version_tuple()
#     if version[0] == '2':
#         return  pickle.load(f)
#     elif version[0] == '3':
#         return  pickle.load(f, encoding='latin1')
#     raise ValueError("invalid python version: {}".format(version))



# def load_CIFAR_batch(filename):
#     """ load single batch of cifar """
#     with open(filename, 'rb') as f:
#         datadict = load_pickle(f)
#         X = datadict['data']
#         Y = datadict['labels']
#         X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
#         Y = np.array(Y).astype("float")
#         return X, Y



# def load_CIFAR10():
#     """ load all of cifar """
#     xs = []
#     ys = []
#     for b in range(1,6):
#         f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
#         X, Y = load_CIFAR_batch(f)
#         xs.append(X)
#         ys.append(Y)
#     Xtr = np.concatenate(xs)
#     Ytr = np.concatenate(ys)
#     del X, Y
#     Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#     return Xtr, Ytr, Xte, Yte



# def get_data(num_training=49000, num_validation=1000, num_test=1000,
#                      subtract_mean=True):
#     """
#     Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
#     it for classifiers. These are the same steps as we used for the SVM, but
#     condensed to a single function.
#     """
#     # Load the raw CIFAR-10 data
#     X_train, y_train, X_test, y_test = load_CIFAR10()

#     # Subsample the data
#     mask = list(range(num_training, num_training + num_validation))
#     X_val = X_train[mask]
#     y_val = y_train[mask]
#     mask = list(range(num_training))
#     X_train = X_train[mask]
#     y_train = y_train[mask]
#     mask = list(range(num_test))
#     X_test = X_test[mask]
#     y_test = y_test[mask]

#     # Normalize the data: subtract the mean image
#     if subtract_mean:
#         mean_image = np.mean(X_train, axis=0)
#         X_train -= mean_image
#         X_val -= mean_image
#         X_test -= mean_image

#     # Transpose so that channels come first
#     X_train = X_train.transpose(0, 3, 1, 2).copy()
#     X_val = X_val.transpose(0, 3, 1, 2).copy()
#     X_test = X_test.transpose(0, 3, 1, 2).copy()

#     # Package data into a dictionary
#     return {
#       'X_train': X_train, 'y_train': y_train,
#       'X_val': X_val, 'y_val': y_val,
#       'X_test': X_test, 'y_test': y_test,
#     }









# class CifarDataset(Dataset):
#     """
#     Above are helper functions, taken from the cs231n starter code (publicly available :)) 
#     We wrangle them into a sublass of the Dataset object we've defined.
#     """
#     def __init__(self, mode="train", subtract_mean=True, validation_set_size=1000, labels_as_logits=True):
#         Dataset.__init__(self)
#         self.labels_as_logits = labels_as_logits
        
#         data = get_data(num_training=50000-validation_set_size,
#                         num_validation=validation_set_size,
#                         subtract_mean=subtract_mean)

#         if mode.lower() == "train":
#             self.xs = data['X_train']
#             self.ys = data['y_train']
#         elif mode.lower() in ["val", "validation"]:
#             self.xs = data['X_val']
#             self.ys = data['y_val']
#         elif mode.lower() == "test":
#             self.xs = data['X_test']
#             self.ys = data['y_test']
#         else:
#             raise Exception("Invalid mode specified in Cifar Dataset")
        
#         self.validation_set_size = validation_set_size
#         self.test_set_size = data['X_test'].shape[0]



#     def __getitem__(self, index):
#         """
#         Get the 'index'th item from the dataset.
#         """
#         xs = t.tensor(self.xs[index]).float()
#         ys = t.tensor(self.ys[index]).long()
#         if not self.labels_as_logits:
#             return xs, ys
#         ys_onehot = t.zeros((10,))
#         ys_onehot.scatter_(0, ys, 1)
#         return (xs, ys_onehot)



#     def __len__(self):
#         """
#         Get the length of the dataset
#         """
#         return self.xs.shape[0]