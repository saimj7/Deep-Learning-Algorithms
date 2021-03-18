# import the necessary packages
import pickle

def load_cifar10(path):
	# load the CIFAR-10 dataset batch
	f = open(path, "rb")
	data = pickle.load(f)
	f.close()

	# return the CIFAR-10 batch
	return data