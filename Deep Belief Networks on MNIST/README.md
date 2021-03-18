# Deep Belief Networks (DBNs)
### Training and testing on the MNIST, CIFAR-10 datasets

> DBN is a set of RBMs stacked on top of each other (the output from one RBM feeds into the next one, creating a series), allowing us to learn more complex features in the higher-level layers of the network:

<div align="center">
<img src=mylib/misc/1.png?raw=true "Autoencoder" width=570 >
</div>

---

## Simple Theory

- First up, the hidden layers of an RBM can be used as a form of “feature vector”.
- The hidden layer RBM 'i' acts as the visible layer for RBM 'i+1'. To train a DBN:
- Train the first RBM 'i=1' using contrastive divergence algorithm.
- Then train the second RBM 'i=2'. The visible layer for 'i=2' will be the hidden layer for 'i=1'. Thus, contrastive divergence is again applied, only this time using the hidden layer for 'i=1' outputs as the inputs to 'i=2'. Repeat these steps for all layers in the network.

> In order to map the input data and the output labels (mapping nodes from the final layer to the actual class label), we apply a fine tuning phase where the errors of the network are 'backpropagated' through the network.


## Inference

- ```pip install nolearn gdbn```: nolearn is no-longer maintained, so proceed with caution (expect errors).
- To train and test the DBN on MNIST dataset: ```python mnist.py```

<div align="center">
<img src=mylib/misc/3.png?raw=true "Result" width=450 >
</div>

- DBN correctly classified the digits with pretty good accuracy (98%).
> We apply min/max scaling to the dataset such that the pixel intensities are transformed to the range [0, 1].

- To train and test the DBN on CIFAR-10 dataset: ```python cifar.py```

<div align="center">
<img src=mylib/misc/2.png?raw=true "Cifar" width=500 >
</div>

- Result:
```
precision    recall  f1-score   support
0       0.00      0.00      0.00      1000
1       0.00      0.00      0.00      1000
2       0.00      0.00      0.00      1000
3       0.10      1.00      0.18      1000
4       0.00      0.00      0.00      1000
5       0.00      0.00      0.00      1000
6       0.00      0.00      0.00      1000
7       0.00      0.00      0.00      1000
8       0.00      0.00      0.00      1000
9       0.00      0.00      0.00      1000
avg / total       0.01      0.10      0.02     10000
```

> Accuracy is worse than a random guess: DBN was not able to learn anything at all. They are not suited for image classification where there exists much variance in pose, color, and orientation as in the CIFAR-10 dataset.

- NOTE: DBNs are rarely used for image classification as CNNs have demonstrated to be significantly more powerful and robust image classifiers.

## References
- RBMs: https://github.com/saimj7/Deep-Learning-Algorithms/tree/main/Autoencoders%20and%20RBMs
- In-depth theory: https://www.pyimagesearch.com/pyimagesearch-gurus/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

---

saimj7/ 18-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
