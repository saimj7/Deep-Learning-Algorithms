# Transfer Learning
### Feature Extraction with VGG16; Testing on Kaggle cats vs. dogs

> Basically we stop the propagation (feed forward) at an arbitrary layer, such as an activation or pooling layer, extract the values from the network at this time, and then use them as feature vectors.

<div align="center">
<img src=mylib/misc/tl.png?raw=true "Architecture" width=430 >
</div>

---

## Simple Theory
- After removing the last FC layer from the VGG16 net above (right), we are left with a max pooling layer with output shape of 7 × 7 × 512 implying there are 512 filters each of size 7 × 7.
- If we were to forward propagate an image, we would be left with 512, 7 × 7 activations that have either activated or not based on the image contents. Therefore, we can actually take these 7 × 7 × 512 = 25,088 values and treat them as a feature vector that quantifies the contents of an image.
- If we repeat this process for an entire dataset of images (including datasets that VGG16 was not trained on), we’ll be left with a design matrix of N images, each with 25,088 columns used to quantify their contents (i.e., feature vectors).
- Given our feature vectors, we can train an off-the-shelf machine learning model such a Linear SVM, Logistic Regression classifier, or Random Forest to classify new datasets.

## Inference
- Extract the dataset and place the images in respective directory (mylib/datasets).
- Make sure your machine has sufficient memory (check buffer and batch sizes in the code).
- To extract features: ```python extract_features.py --dataset mylib/datasets/kaggle_dogs_vs_cats/train --output mylib/datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5```.
- To train a model (logistic regression) on top of the extracted features: ```python train.py --db mylib/datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.cpickle```:

```
precision    recall  f1-score   support
cat       0.98      0.98      0.98      3108
dog       0.98      0.98      0.98      3142
avg / total       0.98      0.98      0.98      6250
```

> We were able to reach 98% acc. even though the net was not trained on dogs vs cats.

## References

- Dataset: https://www.kaggle.com/c/dogs-vs-cats
- More theory: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/

---

saimj7/ 23-04-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
