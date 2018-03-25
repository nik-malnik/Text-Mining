import sys
import numpy as np
import math
import scipy
from scipy.spatial.distance import cosine
from scipy.sparse import lil_matrix
import pandas as pd
import timeit
from time import time
from sklearn.preprocessing import normalize
import pdb
from os import listdir
import tarfile
from os.path import isfile, join
from sklearn import preprocessing
from time import time
import eval_ndcg
	

def Read_Data():
	
	tar = tarfile.open("HW4_data.tar.gz", "r:gz")
	tar.extractall()
	tar.close()
	
	train = pd.read_csv('HW4_data/train.csv',header=None,names = ['movie_id','user_id','rating','date'],dtype={'rating': np.float64},parse_dates=['date'])
	dev = pd.read_csv('HW4_data/dev.csv',header=None,names = ['movie_id','user_id'])
	test = pd.read_csv('HW4_data/test.csv',header=None,names = ['movie_id','user_id'])
	
	max_vals = train.max()
	M_train = lil_matrix((max_vals['user_id'] + 1,max_vals['movie_id'] + 1),dtype=np.float)
	for index, row in train.iterrows():
		M_train[row['user_id'],row['movie_id']] = row['rating']
	
	return (M_train,train,test,dev)

''' Performs user-user, movie-movie and pcc based on the arguments supplied'''
def Collaborative_Filtering(M_train, k,is_cosine = True, is_weighted = False, is_memorybased = True, is_pcc = False):
	
	M = M_train
	M_train = (M_train.todense()) - 3.0
	M_train[M_train == -3.0] = 0.0
	M_train = lil_matrix(M_train)
	
	''' Normalization for pcc'''
	if is_pcc and (is_memorybased == False):
		temp = []
		for i in range(M.shape[0]):
			mn = np.nan_to_num(M[i][M[i].nonzero()].mean())
			t = np.array(M[i].todense() - mn).flatten()
			t[t==-1*mn] = 0.0
			temp.append(t)
		M = lil_matrix(temp)
	else:
		M = M_train
	
	if is_memorybased: 
		temp = M
		if is_cosine or is_pcc:
			temp = normalize(temp, norm='l2', axis=1)
		UU = (temp*temp.transpose()).todense()
	else:
		temp = M
		if is_cosine or is_pcc:
			temp = normalize(temp, norm='l2', axis=0)
		UU = (temp.transpose()*temp).todense()
		
	''' Since Self should not be a nearest neighbour '''
	np.fill_diagonal(UU,-float("inf"))
	inds = UU.argsort(axis= 1)[:,:UU.shape[1] - k]
	for i in range(UU.shape[0]):
		UU[i,inds[i]] = 0
	
	''' If all the nearest neighbours are equally weighted '''
	if is_weighted == False:
		UU = UU != 0
		
	UU = normalize(UU, norm='l1', axis=1)	
	
	'''predicting every single movie user pair '''
	if is_memorybased:
		M_pred = UU*M_train
	else:
		M_pred = M_train*UU.transpose()
	
	return M_pred

def Experiment123(M_train, dev, filename = 'dev'):

	for is_memorybased in [True, False]:
		for is_cosine in [True, False]:
			for k in [10,100,500]:
				t1 = time()
				M_pred = Collaborative_Filtering(lil_matrix(M_train), k,is_cosine = is_cosine, is_weighted = False, is_memorybased = True, is_pcc = False)
				t2 = time()
				Predict_Ratings(dev,M_pred, filename = 'dev')
				t3 = time()
				experiment = {'is_memorybased': is_memorybased,'is_cosine': is_cosine, 'k': k, 'TrainingTime': str(t2-t1), 'PredictionTime': str(t3-t2) }
				print "Completed: " + str(experiment)


''' Objective function for PMF '''
def PMF_Error_Objective(M,U,V,I,Sm):
	Su = np.var(U.todense())
	Sv = np.var(V.todense())
	
	E = M - (U.transpose())*V
	E1 = (1.0/(2.0*Sm))*(I.multiply(E).multiply(E).sum()) 
	E2 = (1.0/(2.0*Su))*np.sum(np.linalg.norm(U.toarray(),axis=0)) + (1.0/(2.0*Sv))*np.sum(np.linalg.norm(V.toarray(),axis=0))
	#print "Training Error:" + str(E1) + " Regularization:" + str(E2)
	E = E1 + E2

	return E

''' Derivative of objective function '''
def PMF_Obj_Derivative(M,U,V,I,Sm):
	Su = np.var(U.todense())
	Sv = np.var(V.todense())
	
	E = M - (U.transpose())*V
	dEdU = (1.0/Su)*U - (1.0/Sm)*V*(I.multiply(E)).transpose()
	dEdV = (1.0/Sv)*V - (1.0/Sm)*U*(I.multiply(E))

	return (dEdV, dEdU)

def PMF_GD(M,dim=2,step_size = 0.00005):
	U = scipy.sparse.rand(dim, M.shape[0], density=1.0, format='lil')
	V = scipy.sparse.rand(dim, M.shape[1], density=1.0, format='lil')
	I = M != 0
	
	M = (M.todense()) - 3.0
	M[M == -3.0] = 0.0
	M = lil_matrix(M)
	
	Sm = np.var(M.todense())
	Loss = float('inf')
	change = 1.0
	i = 1
	
	''' Stopping criteria '''
	while (i < 30 or change > 0.00001 or change < 0) and i < 300:
		(dEdV, dEdU) = PMF_Obj_Derivative(M,U,V,I,Sm)

		U_new = U - step_size*dEdU
		V_new = V - step_size*dEdV
		Loss_new = PMF_Error_Objective(M,U_new,V_new,I,Sm)
		
		if i%50 == 0:
			print "Loss=" + str(Loss_new) + ", change=" + str((Loss - Loss_new)/Loss) + ", gradient=" + str(abs(dEdU.sum()) + abs(dEdV.sum()))
			print "Step size:" + str(step_size)
		
		if Loss_new > Loss or Loss_new == float('inf'):
			step_size = step_size/1.25
		else:
			U = U_new
			V = V_new
			change = (Loss - Loss_new)/Loss
			Loss = Loss_new
			if i%10 == 0:
				step_size = step_size*1.25
		
		i = i + 1
	
	M_pred = (U.transpose())*V
	
	return (U,V,M_pred,i)

def Experiment4(M_train, dev, filename = 'dev'):
	for dim in [10,20,50,100]:
		try:
			t1 = time()
			(U,V,M_pred,iterations) = PMF_GD(M_train,dim=dim)
			Predict_Ratings(dev,M_pred, filename = 'dev')
			t2 = time()
			print {'dim': dim, 'iterations': iterations, 'time': str(t2-t1)}
		except:
			print "Error Encountered"

def Predict_Ratings(dev,M_pred, filename = 'dev'):
	
	pred_rating = []
	for index,row in dev.iterrows():
		pred_rating.append({'user_id':row['user_id'], 'movie_id': row['movie_id'], 'rating' : M_pred[row['user_id'],row['movie_id']] + 3.0})
	
	pred_rating = pd.DataFrame(pred_rating)
	Write_predicted_rating_file(pred_rating,filename = filename +'.pred')

def Write_predicted_rating_file(pred_rating,filename = 'dev.pred'):
	#print "\nWriting " + filename + "\n"
	f = open(filename, 'w')
	for index, row in pred_rating.iterrows():
		f.write(str(row['rating']) + '\n')	
	f.close()
	
	return eval_ndcg.Evaluate(your_prediction = filename)

'''
print "Starting to read Data"
(M_train,train,test,dev) = Read_Data()

print "Starting experiment 1,2,3"
Results1 = Experiment123(M_train, dev, filename = 'dev')

print "Starting PMF"
Results2 = Experiment4(M_train, dev, filename = 'dev')
'''
