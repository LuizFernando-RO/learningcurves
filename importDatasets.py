import numpy as np

def importVehicle():

	classHeader = ('opel','saab','bus','van')
	
	X = []
	Y = []
	
	with open('Datasets/vehicle.dat') as f:
	
		for line in f:
		
			v = line.strip().split(' ')
			
			X.append(v[:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importIris():

	classHeader = ('Iris-setosa','Iris-versicolor','Iris-virginica')
	
	X = []
	Y = []
	
	with open('Datasets/iris.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			X.append(v[:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	X = X.astype(np.float)
	
	Y = np.array(Y)
	
	return X, Y
	
def importGlass():

	classHeader = ('1','2','3','4','5','6','7')
	
	X = []
	Y = []
	
	with open('Datasets/glass.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[1:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importEcoli():

	classHeader = ('cp','im','pp','imU','om','omL','imL','imS')
	
	X = []
	Y = []
	
	with open('Datasets/ecoli.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[1:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importLiver():

	X = []
	Y = []
	
	with open('Datasets/bupa.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importBanana():

	X = []
	Y = []
	
	with open('Datasets/banana.dat') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importWine():

	classHeader = ('1','2','3')
	
	X = []
	Y = []
	
	with open('Datasets/wine.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[1:])
			Y.append(v[0])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y
	
def importDiabetes():

	X = []
	Y = []
	
	with open('Datasets/pima-indians-diabetes.data') as f:
	
		for line in f:
		
			v = line.strip().split(',')
			
			X.append(v[:-1])
			Y.append(v[-1])
			
	X = np.array(X)
	Y = np.array(Y)
	
	X = X.astype(np.float)

	return X, Y