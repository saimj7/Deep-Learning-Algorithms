# Autoencoders and Restricted Boltzmann Machines (RBMs)
### (The building blocks of Deep Belief Networks)

> An autoencoder is a feedforward neural network that attempts to learn a compressed representation of a dataset. This network architecture consists of one input layer, at least one hidden layer, and finally an output layer:

<div align="center">
<img src=mylib/misc/1.png?raw=true "Autoencoder" width=570 >
</div>

---

## Simple Theory
## 1. Autoencoder:
- An autoencoder is trained to 'reconstruct' its input — we are essentially trying to obtain (almost) the same output from the network as we put into it, but 'compressed' in some manner.
- The goal is not to learn a direct mapping between the training data and class labels, but rather learn the structure of the data itself. Thus, the number of nodes in the hidden layer should be smaller than the size of the input and output layers, forcing the network to learn only the utmost important and discriminative features. This also serves as a form of dimensionality reduction.

> A simpler representation of the data allows us to avoid overfitting (or at the very least, reduce the possibility of overfitting).

## 2. RBMs

- RBMs are comprised of hidden and visible layers. Unlike traditional feedforward networks, the connections between visible and hidden layers of the RBM are undirected, implying that 'information' can travel in both the visible-to-hidden and hidden-to-visible directions:

<div align="center">
<img src=mylib/misc/2.png?raw=true "RBMs" width=570 >
</div>

- RBMs became widely used with the contrastive divergence algorithm (has three phases: a positive phase, a negative phase, and a weight update phase).
- In the positive phase, an input sample v is presented to the input layer. The vector v is then propagated to the hidden layer. We denote the hidden layer activations as h.
During the negative phase, we take h and propagate it back through the visible layer, called v’. We then take v’ and propagate it back to the hidden layer a second time, yielding h’. Finally, we update the weights.

> The positive phase (v and h) reflect the network’s initial representation of the original input vectors. The negative phase attempts to reconstruct the original input vector (v’ and h’). The goal is for the generated data to be as close as possible to the original input data.

## Inference

- To train and test the RBM on MNIST dataset: ```python run.py```

<div align="center">
<img src=mylib/misc/3.png?raw=true "RBMs" width=570 >
</div>

> We apply min/max scaling to the dataset such that the pixel intensities are transformed to the range [0, 1].

- Notice how the RBMs were able to regenerate the original data.

## References
- RBMs paper: https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf
- Contrastive divergence paper: http://proceedings.mlr.press/v9/sutskever10a/sutskever10a.pdf
- Min/max scaling: https://docs.tibco.com/pub/spotfire/5.5.0-march-2013/UsersGuide/norm/norm_scale_between_0_and_1.htm

---

saimj7/ 17-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
