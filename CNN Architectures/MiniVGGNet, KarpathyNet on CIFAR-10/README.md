# MiniVGGNet and KarpathyNet
### KarpathyNet on the CIFAR-10 dataset

> One of the primary benefits of this network architecture is that the memory footprint is small enough that it can be implemented using JavaScript and run fast enough to train a network on the CIFAR-10 dataset inside your browser (on resource constrained systems). Our architecture:

<div align="center">
<img src=mylib/misc/1.png?raw=true "Architecture" width=450 >
</div>

---

## Simple Theory
- Our architecture has three sets of ```CONV => RELU => POOL``` layers, followed by an FC + RELU and a final SOFTMAX classifier. Note that we have included optional DROPOUT layers not used in the original implementation of KarpathyNet as well as FC + RELU layers.
- Dropout is intended to battle overfitting by randomly disconnecting nodes from the current layer to the next layer during training time. Doing this helps prevent nodes from co-adapting and overfitting to the input data.
- ```model.add(Dropout(0.25)``` implies 25% of the nodes connecting to the next layer are randomly disconnected.

## Inference
- First up, ```pip install keras==2.1.5 tensorflow-gpu==1.13.1```.
- To train and test without dropout: ```python train.py --network karpathynet --model output/cifar10_karpathynet_without_dropout.hdf5 --epochs 100```.
- Result: Testing accuracy of 10% is very bad!

> We started to overfit during training. Overfitting occurs when a network is trying to model the training data too closely — in this case, trying to learn the patterns of the training data too closely == ultimately ends in disaster.

- Now let us test with dropout: ```python train.py --network karpathynet --model output/cifar10_karpathynet_with_dropout.hdf5 --dropout 1 --epochs 100```. Result: 67% acc is a significant improvement to 10% without dropout.

---
---

### MiniVGGNet on the CIFAR-10 dataset

> We stack multiple CONV => RELU layers prior to applying a single POOL layer. Doing this allows the network to learn more rich features from the CONV layers prior to applying a destructive POOL operation. Our architecture:

<div align="center">
<img src=mylib/misc/2.png?raw=true "Architecture" width=450 >
</div>

## Simple Theory
- We define 2 sets of ```CONV => RELU => CONV => RELU => POOL``` layers.
- To train and test with dropout: ```python train.py --network minivggnet --model output/cifar10_minivggnet_with_dropout.hdf5 --epochs 200```.
- Testing acc is 83.64%.
- Without dropout: ```python train.py --network minivggnet --model output/cifar10_minivggnet_without_dropout.hdf5 --epochs 200```.
- Testing acc is 79.15% #Overfitting!

>  As the net get deeper, more filters are learned and a higher dropout percentage is used. It is common to see dropout percentages in the range [0.25, 0.5], with larger values occurring in higher level layers of the network, especially amongst FC layers. Otherwise, we overfit.

## References
- VGGNet: https://www.robots.ox.ac.uk/~vgg/research/very_deep/
- KarpathyNet: https://cs.stanford.edu/people/karpathy/convnetjs/
- More on Dropout: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
- SGD parameters: https://keras.io/api/optimizers/#sgd
- In-depth theory: https://www.pyimagesearch.com/pyimagesearch-gurus/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

---

saimj7/ 29-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
