# The Perceptron Algorithm
### Performing bitwise operations

> Perceptron contains N input nodes (which define feature vectors/raw pixel intensities), one for each entry in the input row of the design matrix, followed by only one layer in the network with just a single node in that layer:

<div align="center">
<img src=mylib/misc/1.png?raw=true "Architecture" width=570 >
</div>

---

## Training Mechanism

- First up, we initialize our weight vector 'w' with small random values. Until convergence or training termination, we perform the following operations:

<div align="center">
<img src=mylib/misc/2.png?raw=true "Training" width=570 >
</div>

- After performing the dot product between the weight and feature vectors, the ouput yj is passed throught the step function to return 1 if y>0 and 0 otherwise.
- The delta rule handles updating the weights: (dj-yj) determines if the output classification is correct or not. If correct, then this difference will be zero. Otherwise, the difference will be either positive or negative, giving us the direction in which our weights will be updated (ultimately bringing us closer to the correct classification).
- We then multiply the difference xj, moving us closer to the correct classification.
- Finally, we add in the previous weight vector at time t, wj(t) which completes the process of 'stepping' towards the correct classification.

## Inference

- To train and test the perceptron on bitwise operations: ```python perceptron_and.py```
```
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=0
[INFO] data=[0 1], ground-truth=0, pred=0
[INFO] data=[1 0], ground-truth=0, pred=0
[INFO] data=[1 1], ground-truth=1, pred=1
```
> As observed, bitwise AND is true only when both the input bits (0, 1) are true. Perceptron correctly classified.

- Biwise OR: ```python perceptron_or.py```

```
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=0
[INFO] data=[0 1], ground-truth=1, pred=1
[INFO] data=[1 0], ground-truth=1, pred=1
[INFO] data=[1 1], ground-truth=1, pred=1
```
> Bitwise OR is true only when at least one of the input bits (0, 1) is true. Perceptron correctly classified.

- What about non-linear data? Biwise XOR: ```python perceptron_xor.py```

```
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=1
[INFO] data=[0 1], ground-truth=1, pred=1
[INFO] data=[1 0], ground-truth=1, pred=1
[INFO] data=[1 1], ground-truth=0, pred=0
```
> Bitwise XOR is true only when one of the input bits (0, 1) is true but not both. Perceptron incorrectly classified.

- Further justification: ```python perceptron_iris.py```
```
[INFO] evaluating...
              precision    recall  f1-score   support

      setosa       0.88      1.00      0.94        15
  versicolor       1.00      0.27      0.43        11
   virginica       0.67      1.00      0.80        12

    accuracy                           0.79        38
   macro avg       0.85      0.76      0.72        38
weighted avg       0.85      0.79      0.75        38
```
- The iris dataset is non-linear and the perceptron accuracy is only 85%. Thus, to obtain better accuracy, we would need to leverage a non-linear models, such as SVMs and multi-layer networks.

## References
- Perceptron paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf
- Non-linearity: https://books.google.se/books/about/Perceptrons.html?id=PLQ5DwAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false
- Sklearn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
- In-depth theory: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
---

saimj7/ 15-03-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
