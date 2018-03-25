import collaborative_filtering as CF
import LogReg2 as LR2
import Rank_Prediction as RP
from sklearn.svm import LinearSVC as SVC
import numpy as np
import pandas as pd
from time import time


def Generate_Training_Set(U,V,train):
	
	X = []
	y = []
	temp = train[['user_id','movie_id','rating']]
	temp = temp[(temp['rating'] == 5.0) | (temp['rating'] == 1.0)]
	temp['X'] = map(lambda u,i: np.multiply(U[:,int(u)].todense(),V[:,int(i)].todense()), temp['user_id'],temp['movie_id'])
	temp = pd.merge(temp, temp, how = 'inner', left_on = 'user_id', right_on = 'user_id')
	temp = temp[temp['movie_id_x'] < temp['movie_id_y']][temp['rating_x'] != temp['rating_y']]
	
	temp['X'] = map(lambda left,right: np.array(left - right), temp['X_x'],temp['X_y'])
	temp['y'] = map(lambda left,right: np.sign(left - right), temp['rating_x'],temp['rating_y'])
	
	X = np.array(list(temp['X'])).reshape(len(temp),U.shape[0])
	y = np.array(list(temp['y']))
	
	return (X,y)				

def Generate_Dev_Set(U,V,dev):
	temp = dev[['user_id','movie_id']]
	temp['X'] = map(lambda u,i: np.multiply(U[:,int(u)].todense(),V[:,int(i)].todense()), temp['user_id'],temp['movie_id'])
	Xdev = np.array(list(temp['X'])).reshape(len(temp),U.shape[0])
	
	return Xdev

def main():
	
	print "Read Data"
	(M_train,train,test,dev) = CF.Read_Data()

	print "Evaluate Memory Based Methods"
	CF.Experiment123(M_train, dev, filename = 'dev')

	print "Evaluate PMF"
	CF.Experiment4(M_train, dev, filename = 'dev')

	print "Gathering all Latent Factors"
	data = []
	for dim in [10,20,50,100]:
		data.append(CF.PMF_GD(M_train,dim=dim))
	
	for i in range(len(data)):
		(U,V,M_pred,iterations) = data[i]
		print "Training Data"
		(X,y) = Generate_Training_Set(U,V,train)
		
		t1 = time()
		
		print "Train Linear SVC"
		model = SVC(C = 0.002).fit(X,y)
		print "Evaluate Dev Data"
		Xdev = Generate_Dev_Set(U,V,dev)
		pred_rating = np.dot(model.coef_,Xdev.T).reshape(len(Xdev))
		CF.Write_predicted_rating_file(pd.DataFrame({'rating':pred_rating}),filename = 'PMF_SVC_dev_' + str(i) + '.pred')
		print "Generate Test Predictions"
		Xtest = Generate_Dev_Set(U,V,test)
		pred_rating = np.dot(model.coef_,Xtest.T).reshape(len(Xtest))
		CF.Write_predicted_rating_file(pd.DataFrame({'rating':pred_rating}),filename = 'PMF_SVC_test_' + str(i) + '.pred')
		
		t2 = time()
		
		print "Train LogReg"
		W = LR2.LogReg_SGA(X,y,batch=10000,validation_size=10000,epochs = 2)
		print "Evaluate Dev Data"
		pred_rating = LR2.Rho(Xdev,W)[:,1]
		CF.Write_predicted_rating_file(pd.DataFrame({'rating':pred_rating}),filename = 'PMF_LR_dev_' + str(i) + '.pred')
		print "Generate Test Predictions"
		pred_rating = LR2.Rho(Xtest,W)[:,1]
		CF.Write_predicted_rating_file(pd.DataFrame({'rating':pred_rating}),filename = 'PMF_LR_test_' + str(i) + '.pred')

		t3 = time()
		
		print "Time for LinearSVC:" + str(t2-t1) + ", time for LogReg:" + str(t3-t2)

main()
