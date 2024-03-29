from __future__ import print_function

import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets






class ImagenetDataset(Dataset):
    """
    Our wrapper for the imagenet dataset (just an "ImageFolder" dataset) and it's corresponding normalizations.
    """

    def __init__(self, mode="train", inception=True):
        Dataset.__init__(self)

        # Directories
        ROOT =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'imagenet'))
        # ROOT = "/raid/local_scratch/mmp10-ihp03-{jobid}/imagenet".format(jobid=os.environ["SLURM_JOB_ID"]) # will raise an exception if $SLURM_JOB_ID is not a bash variable.
        ROOT_TRAIN = os.path.join(ROOT, 'train')
        ROOT_VAL = os.path.join(ROOT, 'val')
        ROOT_TEST = os.path.join(ROOT, 'test')


        # Normalization stats
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize_dim = 341 if inception else 256
        input_dim = 299 if inception else 224

        # Make the dataset, using the correct directory and random/center cropping
        dataset = None
        mode = mode.lower()
        if mode == "train":
            dataset = datasets.ImageFolder(
                ROOT_TRAIN,
                transforms.Compose([
                    transforms.RandomResizedCrop(input_dim),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif mode in ["val", "validation"]:
            dataset = datasets.ImageFolder(
                ROOT_VAL,
                transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.CenterCrop(input_dim),
                    transforms.ToTensor(),
                    normalize,
                ]))

        if mode == "test":
            dataset = datasets.ImageFolder(
                ROOT_TEST,
                transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.CenterCrop(input_dim),
                    transforms.ToTensor(),
                    normalize,
                ]))

        self.dataset = dataset



    def __getitem__(self, index):
        """
        Get the 'index'th item from the dataset.
        """
        return self.dataset[index]



    def __len__(self):
        """
        Get the length of the dataset
        """
        return len(self.dataset)





def get_imagenet_dataloader(mode="train", batch_size=0, shuffle=True, num_workers=1, pin_memory=True):
    return DataLoader(dataset=ImagenetDataset(mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)