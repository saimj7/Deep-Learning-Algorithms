# import the necessary packages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn import datasets

# load the MNIST dataset and split it into training and testing data
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(mnist.data, mnist.target,
	test_size=0.25, random_state=42)

# train the Perceptron
print("[INFO] training...")
model = Perceptron(max_iter=30, eta0=1.0, random_state=84)
model.fit(trainData, trainLabels)

# evaluate the Perceptron
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))
