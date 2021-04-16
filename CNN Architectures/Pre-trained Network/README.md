# Transfer Learning with ShallowNet
### Testing a pre-trained net with/without CIFAR-10

---

## Inference
- To test the pre-trained net on CIFAR-10: ```python test.py --model output/cifar10_shallownet.hdf5 --test-images test_images```. Note that the network is actually trained on CIFAR-10, nothing fancy here.

- Later the script will also execute the pre-trained net on a new set of images, now that's interesting.

> The purpose is to demonstrate how we can serialize and load our networks from disk and utilize them to classify images that are not part of the original dataset they were trained on.

## References

- ShallowNet: https://github.com/saimj7/Deep-Learning-Algorithms/tree/main/ShallowNet%20CNN%20on%20CIFAR-10
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

---

saimj7/ 16-04-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
