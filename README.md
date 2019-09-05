# Bayesian-Neural-Network-Pytorch

This is a lightweight repository of bayesian neural network for Pytorch.
There are bayesian versions of pytorch layers and some utils.
The aim is to help construct bayesian neural network intuitively.

## Usage

### Dependencies

- torch 1.2.0
- python 3.6

### Installation

- `pip install torchbnn` or
- `git clone https://github.com/Harry24k/bayesian-neural-network-pytorch`

```python
import torchbnn
```

## Thanks to

* @kumar-shridhar [github:PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
* @xuanqing94 [github:BayesianDefense](https://github.com/xuanqing94/BayesianDefense)

## Update Records

### Version 0.1
* **modules** : BayesLinear, BayesConv2d, BayesBatchNorm2d
* **utils** : convert_model(nonbayes_to_bayes, bayes_to_nonbayes)
* **functional** : bayesian_kl_loss
