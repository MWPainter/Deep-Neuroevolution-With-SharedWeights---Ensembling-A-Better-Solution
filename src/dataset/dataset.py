
class Dataset:
    def __init__(self):
        self.batch_size = -1
        self.name = "abstract"
        self.data_dims = []
        self.width = -1
        self.height = -1
        self.train_size = -1
        self.val_size = -1
        self.test_size = -1
        self.range = [0.0, 1.0]

    """ Get next training batch """
    def next_batch(self, batch_size):
        self.handle_unsupported_op()
        return None
    
    def next_val_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def next_test_batch(self, batch_size):
        self.handle_unsupported_op()
        return None
        
    def get_val_set_size(self):
        self.handle_unsupported_op()
        return None
    
    def val_set(self, batch_size=None):
        self.handle_unsupported_op()
        return None
    
    def get_test_set_size(self):
        self.handle_unsupported_op()
        return None
        
    def test_set(self, batch_size=None):
        self.handle_unsupported_op()
        return None

    def display(self, image):
        return image

    """ After reset, the same batches are output with the same calling order of next_batch or next_test_batch"""
    def reset(self):
        self.handle_unsupported_op()

    def handle_unsupported_op(self):
        print("Unsupported Operation")
        raise(Exception("Unsupported Operation"))
        
        
        
        
class DatasetCudaWrapper:
    def __init__(self, dataset, use_cuda=True):
        self.dataset = dataset
        self.use_cuda = use_cuda
    
    def next_batch(self, batch_size=None):
        xs, ys = self.dataset.next_batch(batch_size)
        if self.use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        return xs, ys
    
    def next_val_batch(self, batch_size=None):
        xs, ys = self.dataset.next_val_batch(batch_size)
        if self.use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        return xs, ys

    def next_test_batch(self, batch_size=None):
        xs, ys = self.dataset.next_test_batch(batch_size)
        if self.use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        return xs, ys    
    
    def get_val_set_size(self):
        return self.dataset.get_val_set_size()
    
    def val_set(self, batch_size=None):
        for xs, ys in self.dataset.val_set(batch_size):
            xs = xs.cuda()
            ys = ys.cuda()
            yield xs, ys
    
    def get_test_set_size(self):
        return self.dataset.get_test_set_size()
        
    def test_set(self, batch_size=None):
        for xs, ys in self.dataset.test_set(batch_size):
            xs = xs.cuda()
            ys = ys.cuda()
            yield xs, ys

    def display(self, image):
        return self.dataset.display(image)
    
    def reset(self):
        self.dataset.reset()
        
    def __iter__(self):
        for xs, ys in self.dataset:
            xs = xs.cuda()
            ys = ys.cuda()
            yield xs, ys

        
