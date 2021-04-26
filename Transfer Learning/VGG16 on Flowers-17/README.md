# Transfer Learning
### Feature Extraction with VGG16; Training a LR classifier/Testing on Flowers-17 dataset

> Basically we stop the propagation (feed forward) at an arbitrary layer, such as an activation or pooling layer, extract the values from the network at this time, and then use them as feature vectors.

<div align="center">
<img src=mylib/misc/17.png?raw=true "Architecture" width=430 >
</div>

---

## Inference
- Extract the dataset and place the images in respective directory (mylib/datasets).
- Make sure your machine has sufficient memory (check buffer and batch sizes in the code).
- To extract features: ```python extract_features.py --dataset mylib/datasets/flowers17/images --output mylib/datasets/flowers17/hdf5/features.hdf5```.
- To train a model (logistic regression) on top of the extracted features: ```python train.py --db mylib/datasets/flowers17/hdf5/features.hdf5 --model flowers17.cpickle```:

```
             precision    recall  f1-score   support
avg / total       0.93      0.92      0.92       340
```

> We were able to reach a final acc. of 93% even though the net was not trained on flowers17.

## References

- Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
- More theory: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/

---

saimj7/ 23-04-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
