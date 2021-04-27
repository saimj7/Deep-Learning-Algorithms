# Transfer Learning
### Feature Extraction with VGG16; Training a LR classifier/Testing on Caltech-101 dataset

> Basically we stop the propagation (feed forward) at an arbitrary layer, such as an activation or pooling layer, extract the values from the network at this time, and then use them as feature vectors.

<div align="center">
<img src=mylib/misc/ct.png?raw=true "Dataset" width=450 >
</div>

---

## Inference
- Extract the dataset and place the images in respective directory (mylib/datasets).
- Make sure your machine has sufficient memory (check buffer and batch sizes in the code).
- To extract features: ```python extract_features.py --dataset mylib/datasets/caltech-101/images --output mylib/datasets/caltech-101/hdf5/features.hdf5```.
- To train a model (logistic regression) on top of the extracted features: ```python train.py --db mylib/datasets/caltech-101/hdf5/features.hdf5 --model ct101.cpickle```:

```
[INFO] evaluating...
                 precision    recall  f1-score   support
          Faces       1.00      0.98      0.99       114
     Faces_easy       0.98      1.00      0.99       104
       Leopards       1.00      1.00      1.00        44
     Motorbikes       1.00      1.00      1.00       197
...
  windsor_chair       0.92      0.92      0.92        13
         wrench       0.88      0.78      0.82         9
       yin_yang       1.00      1.00      1.00        11
    avg / total       0.96      0.96      0.96      2170
```

> We were able to reach a final acc. of 96% even though the net was not trained on Caltech-101.

- Networks such as VGG are capable of performing transfer learning, encoding their discriminative features into output activations that we can use to train our own custom classifiers!

## References
- Dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- More theory: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/

---

saimj7/ 27-04-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
