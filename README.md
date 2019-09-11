# Bayesian-Neural-Network-Pytorch

This is a lightweight repository of bayesian neural network for Pytorch.
There are bayesian versions of pytorch layers and some utils.
To help construct bayesian neural network intuitively, all codes are modified based on the original pytorch codes.

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
<<<<<<< HEAD

### Version 0.2
* **prior_sigma** is used when initialize modules and functions instead of **prior_log_sigma**
	* **Modules(BayesLinear, BayesConv2d, BayesBatchNorm2d)** are re-defined with prior_sigma instead of prior_log_sigma.
	* **convert_model(nonbayes_to_bayes, bayes_to_nonbayes)** is also changed with prior_sigma instead of prior_log_sigma.
* **Modules(BayesLinear, BayesConv2d, BayesBatchNorm2d)** : Base initialization method is changed to the method of Adv-BNN from the original torch method.
* **functional** : **bayesian_kl_loss** is changed similar to ones in **torch.functional**
* **loss** : **BKLLoss** is added based on bayesian_kl_loss similar to ones in **torch.loss**
=======
>>>>>>> 8be43015d291edb1fc413d4ec25e21d238374f70
