# Bayesian-Neural-Network-Pytorch

<p>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/bayesian-neural-network-pytorch" /></a>
  <a href="https://img.shields.io/pypi/v/torchbnn"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchbnn.svg" /></a>
  <a href="https://bayesian-neural-network-pytorch.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/bayesian-neural-network-pytorch/badge/?version=latest" /></a>
</p>

This is a lightweight repository of bayesian neural network for PyTorch.

## Usage

### :clipboard: Dependencies

- torch 1.2.0
- python 3.6



### :hammer: Installation

- `pip install torchbnn` or
- `git clone https://github.com/Harry24k/bayesian-neural-network-pytorch`

```python
import torchbnn
```

## Citation
If you use this package, please cite the following BibTex (SemanticScholar, GoogleScholar):

```
@article{lee2021graddiv,
  title={GradDiv: Adversarial Robustness of Randomized Neural Networks via Gradient Diversity Regularization},
  author={Lee, Sungyoon and Kim, Hoki and Lee, Jaewook},
  journal={arXiv preprint arXiv:2107.02425},
  year={2021}
}
```

## :mag_right: Update Records

Here is [update records](Update%20Records.md) of this package.



## :rocket: Demos

* **Bayesian Neural Network Regression** ([code](https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Regression.ipynb)): 
In this demo, two-layer bayesian neural network is constructed and trained on simple custom data. It shows how bayesian-neural-network works and randomness of the model.
* **Bayesian Neural Network Classification** ([code](https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Classification.ipynb)): 
To classify Iris data, in this demo, two-layer bayesian neural network is constructed and trained on the Iris data. It shows how bayesian-neural-network works and randomness of the model.
* **Convert to Bayesian Neural Network** ([code](https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Convert%20to%20Bayesian%20Neural%20Network.ipynb)): 
To convert a basic neural network to a bayesian neural network, this demo shows how `nonbayes_to_bayes` and `bayes_to_nonbayes` work.
* **Freeze Bayesian Neural Network** ([code](https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Freeze%20Bayesian%20Neural%20Network.ipynb)): 
To freeze a bayesian neural network, which means force a bayesian neural network to output same result for same input, this demo shows the effect of `freeze` and `unfreeze`.



## Thanks to

* @kumar-shridhar [github:PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
* @xuanqing94 [github:BayesianDefense](https://github.com/xuanqing94/BayesianDefense)
