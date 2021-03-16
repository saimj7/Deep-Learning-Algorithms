# Backpropagation Algorithm
### A multi-layer network architecture

> To obtain perfect classification accuracy on the XOR problem, we need a feedforward neural network with at least a single hidden layer, so let’s consider a 2 - 2 - 1 multi-layer architecture (3-3-1 with the bias trick):

<div align="center">
<img src=mylib/misc/1.png?raw=true "Architecture" width=570 >
</div>

---

## Training Mechanism
## 1. The Forward Pass:
- The purpose of the forward pass is to propagate our inputs through the network by applying a series of dot products and activations until we reach the output layer of the network (i.e., our predictions):

<div align="center">
<img src=mylib/misc/2.png?raw=true "Training" width=570 >
</div>

- First up, the input vector [0, 1, 1] is presented to the network.
- The dot product between the inputs and weights are taken, followed by applying the sigmoid activation function to obtain the values in the hidden layer (0.899, 0.593, and 0.378, respectively).
- Finally, the dot product and sigmoid activation function is computed for the final layer, yielding an output of 0.506. Applying the step function to 0.506 yields 1, which is indeed the correct target class label.

> Note that the final accuracy is just 0.506, which is pretty much near the threshold. In order to get better, we move on to the backward pass.

## 2. The Backward Pass

- We compute the gradient (partial derivative of the error) of the loss function at the final layer (i.e., predictions layer) of the network and use this gradient to recursively apply the chain rule to update the weights in our network (also known as the weight update phase).
- To understand it better: [**Ref1**](http://neuralnetworksanddeeplearning.com/chap2.html) and [**Ref2**](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

## Inference

- To train and test the XOR problem: ```python run.py```

```
[INFO] data=[0 0], ground-truth=0, pred=0.0108, step=0
[INFO] data=[0 1], ground-truth=1, pred=0.9875, step=1
[INFO] data=[1 0], ground-truth=1, pred=0.9894, step=1
[INFO] data=[1 1], ground-truth=0, pred=0.0148, step=0
```
> As you can see, our network accurately predicted the non-linear XOR dataset.

- Current architecture is an input layer with two nodes (i.e., our two inputs), a single hidden layer with two nodes and an output layer with one node.
- Further justification (now without multi-layers): change the architecture from ```[2, 2, 1] in line 10 to [2, 1]``` and run again:

```
[INFO] data=[0 0], ground-truth=0, pred=0.5161, step=1
[INFO] data=[0 1], ground-truth=1, pred=0.5000, step=0
[INFO] data=[1 0], ground-truth=1, pred=0.4839, step=0
[INFO] data=[1 1], ground-truth=0, pred=0.4678, step=0
```
> Pretty poor accuracy on the non-linear data.

- Hence, it is safe to say that a multi-layer networks with nonlinear activation functions trained via backpropagation are so important — they enable us to learn patterns in datasets that are otherwise nonlinearly separable.

## References
- Backpropagation paper: http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf
- Top tutorials: [**Coursera**](https://www.coursera.org/learn/machine-learning); [**Stanford**](http://cs231n.stanford.edu/); [**Ref1**](http://neuralnetworksanddeeplearning.com/chap2.html) and [**Ref2**](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- Non-linearity: https://books.google.se/books/about/Perceptrons.html?id=PLQ5DwAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false
- In-depth theory: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
---

saimj7/ 16-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
