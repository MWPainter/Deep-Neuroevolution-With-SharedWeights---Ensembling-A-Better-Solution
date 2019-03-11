
from torch.utils.data import Dataset

class ProductDataset(Dataset):
    """
    Implements the cartesian product of two datasets. Leads to biasing on 
    """
    def __init__(self, dataset1, dataset2):
        """
        :param dataset1: THe first dataset
        :param dataset2: The second dataset
        """
        self.d1 = dataset1
        self.d2 = dataset2

        self.len1 = len(self.d1)
        self.len2 = len(self.d2)
        self.min_len = min(self.len1, self.len2)
        self.max_len = max(self.len1, self.len2)


    def __getitem__(self, index):
        i = index % self.len1
        j = index % self.len2
        x1,y1 = self.d1[i]
        x2,y2 = self.d2[j]
        return x1, y1, x2, y2


    def __len__(self):
        return self.max_len