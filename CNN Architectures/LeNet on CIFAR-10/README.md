# LeNet
### Training and testing on the CIFAR-10 dataset

> This network is very simple, consisting of only an INPUT  layer, a single CONV => RELU  layer, and an output softmax classifier (a generalization of Logistic Regression used to handle multiple classes and return probabilities associated with each class label):

<div align="center">
<img src=mylib/misc/1.png?raw=true "Autoencoder" width=500 >
</div>

---

## Simple Theory
- Notice how the LeNet uses the TANH activation function rather than the more popular ReLU. This is because back in 1998, ReLU had not been used in the context of deep learning. Today, its common to swap out TANH  for ReLU.
- LeNet uses two sets of ```CONV => ACTIVATION => POOL``` layer sets. It is common to see the number of filters increase in subsequent sets (K = 20 in first, 50 in second).
- By stacking multiple sets, we can construct deeper networks which will generalize better to large datasets.

## Inference
- First up, ```pip install keras==2.1.5 tensorflow-gpu==1.13.1```.
- To train and test LeNet on CIFAR-10 dataset: ```python train.py --network lenet --model output/cifar10_lenet.hdf5 --epochs 20```.

- Result:
```
Epoch 19/20
50000/50000 [==============================] - 7s 131us/step - loss: 0.0011 - acc: 1.0000
Epoch 20/20
50000/50000 [==============================] - 7s 132us/step - loss: 0.0010 - acc: 1.0000
10000/10000 [==============================] - 1s 67us/step
[INFO] accuracy: 71.11%
```
> Testing accuracy is better than a standard shallow CNN!

## References
- LeNet paper: https://ieeexplore.ieee.org/document/726791
- Keras: https://keras.io/
- SGD parameters: https://keras.io/api/optimizers/#sgd
- In-depth theory: https://www.pyimagesearch.com/pyimagesearch-gurus/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

---

saimj7/ 26-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
