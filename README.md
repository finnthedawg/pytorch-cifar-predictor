# pytorch-cifar-predictor
Predicting CIFAR classes in pytorch

## Build:
```
$> jupyter notebook predictor.py
```

## Walkthrough:

Build CIFAR DataLoader `torch.utils.data.DataLoader`
Dataset is split into test and training sections.

Realize a CNN established using `nn.Module` and `torch.nn.functional`.
The system uses two conv layers, pooling, RELU and FC.

Implementation of a training framework for classification pipeline:
Apply DataLoader
Apply CNN
Implement Negative Log Likelyhood loss
Optimzie using optimizer from SDG/Adam and use Pytorch Autograd.

Write a testing framework to evaluate pipeline.
Apply dataloader
Apply CNN model
Print Accuracy of model's prediction.

Overall modules implemented: `Dataloader`, `CNN model`, `Loss`, `Training`, `Evaluation`.
