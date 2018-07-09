import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import wisardpkg as wp
import matplotlib.pyplot as plt
import importDatasets as id

nBits = {'banana':20,'diabetes':20,'liver':20,'ecoli':20,'glass':20,'iris': 20,'vehicle':20,'wine':20}
address = {'banana':20,'diabetes':20,'liver':20,'ecoli':20,'glass':20,'iris': 20,'vehicle':20,'wine':20}

nTimes = 1

#mins = {{'banana':np.inf,'diabetes':np.inf,'liver':np.inf,'ecoli':np.inf,'glass':np.inf,'iris':np.inf,'vehicle':np.inf,'wine':np.inf}}

#confirmed

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

	return np.array(bin_X)

def oneHotEncoding(Y, labels):

	OH_Y = np.zeros((Y.shape[0], len(labels)))

	for i in range(Y.shape[0]):

		OH_Y[i, labels.index(Y[i])] = 1.

	return OH_Y

def groupByClass(X, Y):

	d = {}

	for i in range(len(X)):

		if Y[i] in d.keys():

			d[Y[i]].append(i)

		else:

			d[Y[i]] = [i]

	return d

def additiveActivation(a, b, x):
    
    return 1.0 / 1.0 + np.exp( -1 * ( np.dot(a, x) + b) )

def hiddenLayerMatrix(N,L,a,b,X):
    
    H = np.zeros((N,L))
    
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i,j] = additiveActivation(a[j],b[j],X[i])
    
    return H

def normalize(X, minimum, maximum):
    
    normX = X
    
    maxAttr = []
    minAttr = []
    
    for i in range(X.shape[1]):
        
        maxAttr.append( max(normX[:,i]) )
        minAttr.append( min(normX[:,i]) )
        
    for i in range(X.shape[0]):
        
        for j in range(X.shape[1]):
            
            if maxAttr[j] - minAttr[j] == 0:
            
                normX[i,j] = 0
                
            else:
                
                std = (X[i,j] - minAttr[j]) / (maxAttr[j] - minAttr[j])
            
                normX[i,j] = std * (maximum - minimum) + minimum

    return normX

#unconfirmed
	
def getIthExamples(data, Y, group, index):

	batchX = []
	batchY = []

	for k, v in group.items():
	
		sliceX = data[v[:index]]
		sliceY = Y[v[:index]]
	
		for i in range(sliceX.shape[0]):
	
			batchX.append(sliceX[i])
			batchY.append(sliceY[i])
		
	return np.array(batchX), np.array(batchY)

'''
def getIthExamples(group, index):

	examples = []
	classes = []

	for k, v in group.items():

		for i in range(index+1):

			examples.append(v[i])
			classes.append(k)

	examples = np.array(examples)
	classes = np.array(classes)
			
	return examples, classes
'''
	
def wisardCurve(X_train, X_test, Y_train, Y_test, name, group):

	min = np.inf

	for k in group.keys():

		if len(group[k]) < min:

			min = len(group[k])

	a = []

	for i in range(1, min+1):

		wsd = wp.Wisard(address[name])
	
		for cls, idx in group.items():
		
			wsd.train(X_train[idx[:i]], Y_train[idx[:i]])

		pred = wsd.classify(X_test)

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

#experiments

def banana():

	print('BANANA')
    
	name = 'banana'
	
	labels = ["1.0","-1.0"]

	X, Y = id.importBanana()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**-22
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('banana')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()

#
#banana()
	
def diabetes():

	print('DIABETES')
    
	name = 'diabetes'
	
	labels = ['1', '0']

	X, Y = id.importDiabetes()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**-2
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('diabetes')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()

#ok
diabetes()
	
def liver():

	print('LIVER')
    
	name = 'liver'
	
	labels = ['1', '2']

	X, Y = id.importLiver()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**1
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('liver')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()
	
#ok
liver()
	
def ecoli():

	print('ECOLI')
    
	name = 'ecoli'
	
	labels = ['cp','im','pp','imU','om','omL','imL','imS']

	X, Y = id.importEcoli()
	
	ret = []
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		group = groupByClass(bin_X_train, Y_train)

		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		accs = [0.]
		
		for i in range(1, min+1):

			wsd = wp.Wisard(address[name])
		
			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			acc = accuracy_score(Y_test, pred)

			accs.append(acc)
		
		ret.append(accs)
	 
	ret = np.array(evenArray(ret))
		
	mean = np.mean(ret, axis=0)

	plt.plot(mean, label='wisard')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('ecoli')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()
	
#balance problems
#ecoli()
	
def glass():

	print('GLASS')
    
	name = 'glass'
	
	labels = ['1','2','3','4','5','6','7']

	X, Y = id.importGlass()
	
	ret = []
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		group = groupByClass(bin_X_train, Y_train)

		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		accs = [0.]
		
		for i in range(1, min+1):

			wsd = wp.Wisard(address[name])
		
			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			acc = accuracy_score(Y_test, pred)

			accs.append(acc)
		
		ret.append(accs)
	 
	ret = np.array(evenArray(ret))
		
	mean = np.mean(ret, axis=0)

	plt.plot(mean, label='wisard')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('glass')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()
	
#balance problems
#glass()
	
def iris():

	print('IRIS')
    
	name = 'iris'
	
	labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

	X, Y = id.importIris()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**5
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('iris')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()
	
#ok	
iris()

def vehicle():

	print('VEHICLE')
    
	name = 'vehicle'
	
	labels = ['opel','saab','bus','van']

	X, Y = id.importVehicle()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**7
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('vehicle')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()

#ok
vehicle()
	
def wine():

	print('WINE')
    
	name = 'wine'
	
	labels = ['1','2','3']

	X, Y = id.importWine()
	
	wisardRet = []
	elmRet = []
	
	L = 1000
	l = 2**-1
	
	for j in range(nTimes):
	
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

		#for wisard
		bin_X_train = binarize(X_train, nBits[name])
		bin_X_test = binarize(X_test, nBits[name])
		
		#for elm
		OH_Y_train = oneHotEncoding(Y_train, labels)
		OH_Y_test = oneHotEncoding(Y_test, labels)

		elm_X_train = normalize(X_train, -1, 1)
		elm_X_test = normalize(X_test, -1, 1)
		
		group = groupByClass(bin_X_train, Y_train)
		group_train = groupByClass(elm_X_train, Y_train)

		N = elm_X_train.shape[0]
		d = elm_X_train.shape[1]
		
		a = np.random.rand(L,d)
		b = np.random.rand(L)
		
		min = np.inf

		for k in group.keys():

			if len(group[k]) < min:

				min = len(group[k])
		
		wisardAccs = [0.]
		elmAccs = [0.]
		
		for i in range(1,min):
			
			#wisard
			wsd = wp.Wisard(address[name])

			for cls, idx in group.items():
			
				wsd.train(bin_X_train[idx[:i]], Y_train[idx[:i]])

			pred = wsd.classify(bin_X_test)

			wisardAcc = accuracy_score(Y_test, pred)

			wisardAccs.append(wisardAcc)
			
			#elm
			
			batchX, batchY = getIthExamples(elm_X_train, OH_Y_train, group_train, i)			
						
			trainH = hiddenLayerMatrix(batchX.shape[0],L,a,b,batchX)
			
			transp = np.transpose(trainH)
			eye = np.eye(trainH.shape[0])
			big = np.dot( transp, np.linalg.inv( (eye / l) + np.dot(trainH, transp) )  )
			
			B = np.dot( big , batchY )
			
			testH = hiddenLayerMatrix(elm_X_test.shape[0],L,a,b,elm_X_test)
        
			elm_pred = np.dot(testH, B)
			
			elm_acc = 0.0
        
			for i in range(OH_Y_test.shape[0]):
            
				if(np.argmax(OH_Y_test[i]) == np.argmax(elm_pred[i])):
                
					elm_acc += 1
					
			elmAcc = elm_acc / OH_Y_test.shape[0]
			
			elmAccs.append(elmAcc)
				
		wisardRet.append(wisardAccs)
		elmRet.append(elmAccs)
	
	wisardRet = np.array(evenArray(wisardRet))
	elmRet = np.array(evenArray(elmRet))
		
	wisardMean = np.mean(wisardRet, axis=0)
	elmMean = np.mean(elmRet, axis=0)

	plt.plot(wisardMean, label='wisard')
	plt.plot(elmMean, label='elm')
	plt.ylabel("Accuracy")
	plt.xlabel("Size of training data")
	plt.legend()
	plt.title('wine')
	plt.ylim(ymax=1.0, ymin=0.0)
	
	plt.show()

#ok
wine()