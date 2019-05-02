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

### Load the dataset

The CIFAR dataset is split into `5` different batches, each with `10000` images each. We extract each file, and then concatenate them to load the fill dataset. This results in two loaders. `trainloader` which interfaces with the training data and `testloader` that will be used to load test data.

Below is a sample extracted image from our dataset.
<p align="center">
  <img width="600"  src="./digits.png">
</p>

### Create a CNN model

I then created a CNN model based on the following architecture, utilizing Conv2d, Maxpool and Relu in the first section and 3 layers of fully connected nodes in the second. Lastly I used a softmax activation function on the final layer which is recommended for Negative Log Likelihood loss

```
#Define the neural net.
class CNN(nn.Module):

    def __init__(self):
        #Define the network
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5) #We have 3 channels. Output 6 feature map with 5x5 kernel
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 10)

        # Adding a layer for LogSoftmax to obtain log probabilities
        # As recommended in documentation for Negative log likelihood loss https://pytorch.org/docs/stable/nn.html#nllloss
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return(x)
```

### Training the model

After loading the data and creating an instance of the CNN class, it is time to train. In order to train the model, I used `optim.SGD` stochastic gradient descent module. The parameters chosen for the network is `lr=0.005` and `momentum=0.8` which is effective, but could be tuned especially as the loss function was found to fluctuate near the end of training. The loss function used is the a Negative Log Likelihood loss, and the parameters are updated after each passthrough

After training for 10 epochs (20 sets of 25000 images):
<p align="center">
  <img width="600"  src="./digits.png">
</p>


### Evaluating the accuracy.

Finally, the test set was evaluated using our model. The accuracy is the percent of classes the that model predicted correctly. The value obtained was `62%`. This is much lower than the previous MNIST task, but it is to be expected as the images are quite unclear in many cases. Nevertheless, the network architecture could be improved if we could introduce dropout layers, and increase the depth of the network without suffering from overfitting.
