import os
import math
import numpy as np

def Rho(x,w):
	return 1.0/(1.0 + np.exp(-1*np.dot(x,w)))

def LogReg_ReadInputs(filepath):
	
	XTrain = np.loadtxt(open(filepath + "/LogReg_XTrain.csv", "rb"), delimiter=",")
	yTrain = np.loadtxt(open(filepath + "/LogReg_yTrain.csv", "rb"), delimiter=",")
	XTest = np.loadtxt(open(filepath + "/LogReg_XTest.csv", "rb"), delimiter=",")
	yTest = np.loadtxt(open(filepath + "/LogReg_yTest.csv", "rb"), delimiter=",")
	
	yTrain = yTrain.reshape(yTrain.shape[0],1)
	yTest = yTest.reshape(yTest.shape[0],1)
	
	bias = np.ones((XTrain.shape[0],1),dtype="float")
	XTrain = np.concatenate([bias,XTrain],axis=1)
	
	bias = np.ones((XTest.shape[0],1),dtype="float")
	XTest = np.concatenate([bias,XTest],axis=1)
	
	return (XTrain, yTrain, XTest, yTest)
    
def LogReg_CalcObj(X, y, w):
    
    cll =  np.sum(y*np.log(Rho(X,w)) + (1.0-y)*np.log(1-Rho(X,w)))/X.shape[0]
    return cll
    
def LogReg_CalcSG(x, y, w):
    
    sg = (y - Rho(x,w))*x
    return sg
        
def LogReg_UpdateParams(w, sg, eta):
    
    w = w + eta*sg
    return w
    
def LogReg_PredictLabels(X, y, w):
    
    yPred = np.argmin(np.concatenate([Rho(X,w),0.5*np.ones((X.shape[0],1))],axis=1),axis=1)
    PerMiscl = np.sum(np.abs(yPred.reshape(y.shape) - y))/(1.0*y.shape[0])
    
    return (yPred, PerMiscl)    
			

def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    epochs = 5
    trainPerMiscl = []
    testPerMiscl = []
    w = 0.5*np.ones((XTrain.shape[1],1))
    obs = 0
    for epoch in range(epochs):
		for i in range(XTrain.shape[0]):
			eta = 0.5/math.sqrt(obs + 1.0)
			sg= LogReg_CalcSG(XTrain[i], yTrain[i], w)
			sg = sg.reshape(sg.shape[0],1)
			w = LogReg_UpdateParams(w, sg, eta)
			if obs%200000 == 0:
				(yPred, PerMiscl) = LogReg_PredictLabels(XTrain, yTrain, w)
				trainPerMiscl.append(PerMiscl)
				(yPred, PerMiscl) = LogReg_PredictLabels(XTest, yTest, w)
				testPerMiscl.append(PerMiscl)
				print "epoch: " + str(epoch) + " with training error: " + str(trainPerMiscl[-1]) + " and test error: " + str(testPerMiscl[-1])
			obs = obs + 1	
    (yPred, PerMiscl) = LogReg_PredictLabels(XTest, yTest, w)
    return (w, trainPerMiscl, testPerMiscl, yPred)
	
def plot(trainPerMiscl, testPerMiscl):     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    import matplotlib.pyplot as plt
    t = np.arange(0, len(trainPerMiscl), 1) 
    plt.plot(t,trainPerMiscl,t,testPerMiscl)
    plt.show()
    return None
