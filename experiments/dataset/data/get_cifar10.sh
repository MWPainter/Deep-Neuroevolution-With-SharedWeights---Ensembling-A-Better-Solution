# Get CIFAR10
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
mkdir cifar10
mv cifar-10-batches-py/* cifar10
rm -r cifar-10-batches-py
