# pytorch-cifar-predictor
Predicting CIFAR classes in pytorch

## Build instructions:

First download the CIFAR-10 dataset (python version) from original source linked below
https://www.cs.toronto.edu/~kriz/cifar.html

```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

Extract the tar.gz file and place it into `/data/`

```
`tar xfvz cifar-10-python.tar.gz`
```

Then run the jupyter notebook
```
$> jupyter notebook predictor.ipnyb
```

### Creating the dataLoader class

The dataset for CIFAR is loaded into my class cifarDataset. This is implemented such that that we can use the `torch.utils.data.DataLoader()` iterable from pytorch to extract instances from the dataset. In order to interface with the` DataLoader`, the class `cifarDataset` contains a number of functions required by `DataLoader` including:

`__loadImages__` which extracts the `pickled` file from our folder and reshapes the images.

`__getitem__` which extracts an item of idx and normalizes the image section before returning the tensor.

`__len__` which returns the length of the dataset.
