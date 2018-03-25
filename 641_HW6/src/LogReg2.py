import os
import math
import numpy as np
import pandas as pd

''' P(y/x) '''
def Rho(X,W, categories = 2):
	temp = np.exp(np.dot(X,W))
	return np.divide(temp,np.transpose(np.tile(np.sum(temp,axis=1),(categories,1))))

''' Log Likelihood objective function '''
def LogReg_CalcObj(X, y_onehot, W, L = 1.0):
	obj1 = np.trace(np.dot(y_onehot,np.transpose(np.log(Rho(X,W)))))
	obj2 = np.sum(np.linalg.norm(W,axis=0))
	print "Pure Objective:" + str(obj1) + ", Regularization:" + str(obj2) 
	return (obj1 - obj2)
    
''' Stochastic Gradient for a batch of observations '''
def LogReg_CalcSG(X, y_onehot, W, L = 1.0):
    
    sg = np.dot(np.transpose(X),(y_onehot - Rho(X,W))) - L*W
    return sg
    
''' Hard and soft label prediction '''
def LogReg_PredictLabels(X, W):
    
    yPred_hard = np.array([-1,1])[np.argmax(Rho(X,W),axis=1)]
    return yPred_hard  

''' Batch-SGD '''
def LogReg_SGA(X, y, L=100.0, batch = 10000, validation_size = 10000, eta_initial = 0.0005, epochs = 100, W = 0, categories = 2):
	if type(W) == int:
		W = 0.5*np.ones((X.shape[1],categories))
	e = 1.0
	y_onehot = pd.get_dummies(y).as_matrix()
	
	validation_ind = np.random.uniform(low=0,high=X.shape[0] - 1,size=(validation_size)).astype("int")
	train_ind = list(set(range(X.shape[0])) - set(validation_ind))
	X_train = X[train_ind,:]
	X_validation = X[validation_ind,:]
	y_train = y[train_ind]
	y_validation = y[validation_ind]
	y_train_onehot = pd.get_dummies(y_train).as_matrix()
	y_validation_onehot = pd.get_dummies(y_validation).as_matrix()
	
	for epoch in range(epochs):
		# Stopping criteria removed in favor of strict number of epochs for best accuracy reporting.
		for i in range(int(np.floor(X_train.shape[0]/batch))):
			eta = eta_initial/math.sqrt(e)
			sg= LogReg_CalcSG(X_train[i*batch:(i+1)*batch], y_train_onehot[i*batch:(i+1)*batch,:], W, L = L)
			W = W + eta*sg
			e = e + 1.0	
			
		obj = LogReg_CalcObj(X_validation, y_validation_onehot, W, L = L)
		yPred_hard  = LogReg_PredictLabels(X_validation, W)
		print "epoch: " + str(epoch) + ", batch:" + str(i+1) + ", Objective:" + str(obj) + ", Accuracy:" + str(Eval_Accuracy(yPred_hard,y_validation))
			
	return W
	

def Eval_Accuracy(y_Pred,y_True):
	return np.sum((y_Pred - y_True) == 0.0)*1.0/(len(y_True)*1.0)

def Eval_RMSE(y_Pred,y_True):
	return math.sqrt(np.sum(np.power((y_Pred - y_True),2))/(len(y_True))*1.0)
