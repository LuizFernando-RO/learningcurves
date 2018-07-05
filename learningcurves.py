import numpy as np
from sklearn import datasets, model_selection
from sklearn.metrics import accuracy_score
import wisardpkg as wp
import matplotlib.pyplot as plt

def groupByClass(X, Y):

	d = {}

	for i in range(len(X)):

		if Y[i] in d.keys():

			d[Y[i]].append(X[i])

		else:

			d[Y[i]] = [X[i]]

	return d

def getIthExamples(group, index):

	examples = []
	classes = []

	for k, v in group.items():

		for i in range(index+1):

			examples.append(v[i])
			classes.append(k)

	return examples, classes

def binarize(X, nBits):

	n = X.shape[0]
	m = X.shape[1]

	bin_X = []

	for j in range(m):

		max = np.max(X[:,j])
		min = np.min(X[:,j])

		step = (max - min) / nBits

		bounds = [min]

		for k in range(nBits-1):

			if bounds == []:

				bounds.append(min)

			else:

				bounds.append(bounds[-1] + step)

		for i in range(n):

			if j == 0:

				bin_X.append([])

			for k in range(nBits):

				if X[i,j] > bounds[k]:

					bin_X[i].append(1)

				else:

					bin_X[i].append(0)

	return bin_X

def correction(Y):

	Y2 = []

	for i in range(Y.shape[0]):

		Y2.append(str(Y[i]))

	return Y2

def wisardCurve(X_train, X_test, Y_train, Y_test):

	bin_X_train = binarize(X_train, nBits[name])
	bin_X_test = binarize(X_test, nBits[name])

	Y_train = correction(Y_train)
	Y_test = correction(Y_test)

	group = groupByClass(bin_X_train, Y_train)

	min = np.inf

	for k in group.keys():

		if len(group[k]) < min:

			min = len(group[k])

	a = []

	for i in range(min):

		X_batch, Y_batch = getIthExamples(group, i)

		wsd = wp.Wisard(address[name])

		wsd.train(X_batch, Y_batch)

		pred = wsd.classify(bin_X_test)

		acc = accuracy_score(Y_test, pred)

		a.append(acc)

	return a

def evenArray(data):

	l = []

	min = np.inf

	for i in range(len(data)):

		if len(data[i]) < min:

			min = len(data[i])

	for i in range(len(data)):

		l.append(data[i][:min])

	return l

#loading datasets

iris = datasets.load_iris()
digits = datasets.load_digits()
diabetes = datasets.load_diabetes()
wine = datasets.load_wine()
breastCancer = datasets.load_breast_cancer()

datasets = {'iris': iris}#, 'digits' : digits, 'diabetes' : diabetes, 'wine' : wine, 'breast cancer' : breastCancer}
nBits = {'iris': 20}#, 'digits' : digits, 'diabetes' : diabetes, 'wine' : wine, 'breast cancer' : breastCancer}
address = {'iris': 20}#, 'digits' : digits, 'diabetes' : diabetes, 'wine' : wine, 'breast cancer' : breastCancer}
datasetAcc = {'iris': []}#, 'digits' : digits, 'diabetes' : diabetes, 'wine' : wine, 'breast cancer' : breastCancer}
modelsAcc = {'wisard': {'iris' : []}}

#algorithm

for i in range(50):

	for name, dataset in datasets.items():

		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3)

		modelsAcc['wisard'][name].append(wisardCurve(X_train, X_test, Y_train, Y_test))

for model, dataset in modelsAcc.items():

	print(model)

	for name, data in dataset.items():

		acc = np.array(evenArray(data))

		mean = np.mean(acc, axis=0)

		plt.plot(mean, label=model)
		plt.ylabel("Accuracy")
		plt.xlabel("Size of training data")
		plt.legend()
		plt.title(name)
		plt.show()



def importDataset(path, delimiter):

	X = []
	Y = []

	with open(path) as file:

		for line in file:

			v = line.strip().split(delimiter)

			X.append(v[:-1])
			Y.append(v[-1])

	return X, Y

def oneHotEncoding(Y):

	classes = []

	for i in Y:

		if i not in classes:

			classes.append(i)

	OH_Y = np.zeros((Y.shape[0], len(classes)))

	for i in range(Y.shape[0]):

		OH_Y[i, classes.index(Y[i])] = 1.

	return OH_Y

#BINARY

def banana():

	X, Y = importDataset("banana.dat", ',')

	X = np.array(X)
	Y = np.array(Y)

	OH_Y = oneHotEncoding(Y)

	print(X.shape, Y.shape, OH_Y.shape)

	return

def diabetes():

	X, Y = importDataset("pima-indians-diabetes.data", ',')

	X = np.array(X)
	Y = np.array(Y)

	OH_Y = oneHotEncoding(Y)

	print(X.shape, Y.shape, OH_Y.shape)

	return

def liver():

	X, Y = importDataset("bupa.data", ',')

	X = np.array(X)
	Y = np.array(Y)

	OH_Y = oneHotEncoding(Y)

	print(X.shape, Y.shape, OH_Y.shape)

	return

banana()
diabetes()
liver()