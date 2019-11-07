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

### Demos
* **Bayesian Neural Network with Iris Data** ([code](https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20with%20Iris%20Data.ipynb)): 
To classify Iris data, in this demo, two-layer bayesian neural network is constructed and tested with plots. It shows how bayesian-neural-network works and randomness of the model.

## Thanks to

* @kumar-shridhar [github:PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
* @xuanqing94 [github:BayesianDefense](https://github.com/xuanqing94/BayesianDefense)

## Update Records

### Version 0.1
* **modules** : BayesLinear, BayesConv2d, BayesBatchNorm2d are added.
* **utils** : convert_model(nonbayes_to_bayes, bayes_to_nonbayes) is added.
* **functional.py** : bayesian_kl_loss is added.

### Version 0.2
* **prior_sigma** is used when initialize modules and functions instead of **prior_log_sigma**.
	* **modules** are re-defined with prior_sigma instead of prior_log_sigma.
	* **utils/convert_model.py** is also changed with prior_sigma instead of prior_log_sigma.
* **modules** : Base initialization method is changed to the method of Adv-BNN from the original torch method.
* **functional.py** : **bayesian_kl_loss** is changed similar to ones in **torch.functional**.
* **modules/loss.py** : **BKLLoss** is added based on bayesian_kl_loss similar to ones in **torch.loss**.

### Version 0.3
* **functional.py** :
    * **bayesian_kl_loss returns tensor.Tensor([0]) as default** : In the previous version, bayesian_kl_loss returns 0 of int type if there is no Bayesian layers. However, considering all torch loss returns tensor and .item() is used to make them to int type, they are changed to return tensor.Tensor([0]) if there is no Bayesian layers.

### Version 0.4
* **functional.py** :
    * **bayesian_kl_loss is modified** : In some cases, the device problem(cuda/cpu) has occurred. Thus, losses are initialized with tensor.Tensor([0]) on the device on which the model is.
    
### Version 0.5
* **utils/convert_model.py** :
    * **nonbayes_to_bayes, bayes_to_nonbayes is modified** : Before this version, they replace the original model. From now, we can handle it with the 'inplace' argument. Set 'inplace=True' for replace the input model and 'inplace=False' for getting a new model. 'inplace=True' is recommended cause it shortens memories and no future problems with deepcopy.
    
### Version 0.6
* **utils/freeze_model.py** :
    * **freeze, unfreeze methods are added** : bayesian modules always returns different outputs even if inputs are same. It is because of their randomized forward propagation. Sometimes, however, we need to freeze this randomized process for analyzing the model deeply. Then you can use this freeze method for changing the bayesian model into non-bayesian model with same parameters.
* **modules** : For supporting **freeze** method, freeze, weight_eps and bias_eps is added to each modules. If freeze is False (Defalt), weight_eps and bias_eps will be initialized with normal noise at every forward. If freeze is True, weight_eps and bias_eps won't be changed. 

### Version 0.7
* **DO NOT USE**
    
### Version 0.8
* **modules** : For supporting **freeze** method, weight_eps and bias_eps is changed to buffer with register_buffer method. Thorugh this change, it provides save and load even if bayesian neural network is freezed.
    * **BayesModule is added** : Bayesian version of torch.nn.Module. Not being used currently.
* **utils/freeze_model.py** :
    * **freeze, unfreeze methods are modified** : Previous methods didn't work on single layer network.
* **Demos are uploaded** : "Bayesian Neural Network with Iris Data".
