# A ShallowNet CNN
### Training and testing on the CIFAR-10 dataset

> This network is very simple, consisting of only an INPUT  layer, a single CONV => RELU  layer, and an output softmax classifier (a generalization of Logistic Regression used to handle multiple classes and return probabilities associated with each class label):

<div align="center">
<img src=mylib/misc/1.png?raw=true "Autoencoder" width=500 >
</div>

---

## Simple Theory
- In the case of our first (and only) CONV  layer, we will learn K=32 filters each with a receptive field (i.e., kernel size) of 3 x 3.
- We can either increase/decrease the number of learned filters along with the spatial size of each filter.

> It is common to either scale the input data into the range [0, 1]: trainData = trainData.astype("float") / 255.0 or mean-center the data points.

- Training and testing labels can all be integers — a single number representing the class each data point belongs to. However, Keras requires that we convert these single integers into vectors in the range [0, numClasses]: ```trainLabels = np_utils.to_categorical(trainLabels, 10)```. Doing this generates a vector for each label, where the index of the label is set to 1 and all other entries to 0.
- For example, given the class label 3, our label vector would look like: ```[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]```.
- It allows us to apply loss functions other than the standard mean-squared error or squared loss. We can utilize categorical cross-entropy, which enables us to leverage information theory and measure the number of bits between the true distribution of class labels versus the approximated (i.e., predicted) set of class labels: ```model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])```.

## Inference
- First up, ```pip install keras==2.1.5 tensorflow-gpu==1.13.1```.
- To train and test the CNN on CIFAR-10 dataset: ```python train.py --network shallownet --model output/cifar10_shallownet.hdf5 --epochs 20```.

- Result:
```
Epoch 19/20
50000/50000 [==============================] - 5s 100us/step - loss: 0.5150 - acc: 0.8193
Epoch 20/20
50000/50000 [==============================] - 5s 100us/step - loss: 0.4807 - acc: 0.8318
10000/10000 [==============================] - 1s 63us/step
[INFO] accuracy: 55.63%
```
> Testing accuracy is better than a standard DBN!

- NOTE: CNNs are better suited for image datasets that exhibit high variability in object appearance. Furthermore, they are also extremely useful when it becomes non-trivial to define handcrafted features (i.e., BOVW, HOG, etc.) that quantify the contents of an image.

## References
- Keras: https://keras.io/
- SGD parameters: https://keras.io/api/optimizers/#sgd
- In-depth theory: https://www.pyimagesearch.com/pyimagesearch-gurus/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

---

saimj7/ 23-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
