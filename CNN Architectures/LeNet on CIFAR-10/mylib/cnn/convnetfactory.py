# import the necessary packages
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras import backend as K

class ConvNetFactory:
	def __init__(self):
		pass

	@staticmethod
	def build(name, *args, **kargs):
		# define the network (i.e., string => function) mappings
		mappings = {"lenet": ConvNetFactory.LeNet}

		# grab the builder function from the mappings dictionary
		builder = mappings.get(name, None)

		# if the builder is None, then there is not a function that can be used
		# to build to the network, so return None
		if builder is None:
			return None

		# otherwise, build the network architecture
		return builder(*args, **kargs)

	@staticmethod
	def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
		# initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the second set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))

		# define the second FC layer
		model.add(Dense(numClasses))

		# lastly, define the soft-max classifier
		model.add(Activation("softmax"))
		model.summary()

		# return the network architecture
		return model
