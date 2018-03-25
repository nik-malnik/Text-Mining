import sys
import numpy as np
import math
import scipy
from scipy.spatial.distance import cosine
from scipy.sparse import lil_matrix,dok_matrix
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
import json
import string
import re
from collections import Counter
import collections
import LogReg as LR
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
from sklearn.svm import LinearSVC as SVC

def Read_Stopwords():
	f = open('stopword.list', 'rb')
	words = []
	for line in iter(f):
		words.append(line.strip())
	f.close()
	return words

def Read_Data(filename = 'yelp_reviews_train.json'):
	f = open(filename, 'rb')
	Data = []
	for line in iter(f):
		temp = json.loads(line)
		text = re.sub('[^\w\s]|[\\n]|([a-z]+[0-9]+|[0-9]+[a-z]+)[a-z0-9]*', '', temp['text']).lower().strip()
		if 'train' in filename:
			Data.append({'review_id':temp['review_id'], 'text':text, 'stars':int(temp['stars'])})
		else:
			Data.append({'review_id':temp['review_id'], 'text':text, 'stars': -1})
	f.close()
	return Data
	

def Process_Data(Data, dictionaries = [], n_features = 2000, is_tfidf = False):
	
	stopwords = Read_Stopwords()
	stopwords.append('')
	corpus = [x['text'] for x in Data]
	y = np.array([x['stars'] for x in Data])
	
	if dictionaries == []:
		if is_tfidf:
			v = TfidfVectorizer(min_df=1,stop_words=stopwords)
		else:
			v = CountVectorizer(min_df=1,stop_words=stopwords)
		X = v.fit_transform(corpus)
		CTF_inds = np.argsort(np.sum(X,axis=0))[0,-1*n_features:]
		CTF_inds = np.array(np.transpose(CTF_inds)).reshape(n_features)
		DF_inds = np.argsort(np.sum(X>0,axis=0))[0,-1*n_features:]
		DF_inds = np.array(np.transpose(DF_inds)).reshape(n_features)
		CTF_dictionary = np.array(v.get_feature_names())[CTF_inds]
		DF_dictionary = np.array(v.get_feature_names())[DF_inds]
	
		[X_CTF, X_DF] = [X[:,CTF_inds], X[:,DF_inds]]
		del X
		return ([X_CTF, X_DF],[CTF_dictionary,DF_dictionary],y)
	
	else:
		X_CTF = CountVectorizer(min_df=1,vocabulary=dictionaries[0]).fit_transform(corpus)
		X_DF = CountVectorizer(min_df=1,vocabulary=dictionaries[1]).fit_transform(corpus)
		return ([X_CTF, X_DF],dictionaries,y)

def Train():
	
	W = []
	models = []
	
	print "\nReading Train Data both CTF and DF ..."
	Data = Read_Data(filename = 'yelp_reviews_train.json')
	(Xs, dictionaries,y) = Process_Data(Data,n_features = 2000, is_tfidf = False)
	
	print "\nLOGISTIC REGRESSION"
	W.append(LR.LogReg_SGA(Xs[0], y, L=1.0, batch = 50000, validation_size = 20000, eta_initial = 0.00005, epochs = 200))
	(yPred_hard, yPred_soft)  = LR.LogReg_PredictLabels(Xs[0], W[-1])	
	print "Training Accuracy CTF:" + str(LR.Eval_Accuracy(yPred_hard,y)) + ", RMSE:" + str(LR.Eval_RMSE(yPred_soft,y))
	
	print "\nTraining DF..."
	W.append(LR.LogReg_SGA(Xs[1], y, L=1.0, batch = 50000, validation_size = 20000, eta_initial = 0.00005, epochs = 200))
	(yPred_hard, yPred_soft)  = LR.LogReg_PredictLabels(Xs[1], W[-1])	
	print "Training Accuracy DF:" + str(LR.Eval_Accuracy(yPred_hard,y)) + ", RMSE:" + str(LR.Eval_RMSE(yPred_soft,y))
	
	print "\nLinearSVM..."
	models.append(SVC(C = 0.002).fit(Xs[0],y))
	print "Training Accuracy CTF:" + str(models[-1].score(Xs[0],y))
	models.append(SVC(C = 0.002).fit(Xs[1],y))
	print "Training Accuracy DF:" + str(models[-1].score(Xs[1],y))
	
	Test(W,models,dictionaries)
	
	return (W,models,dictionaries)

def Test(W,models,dictionaries):	
	
	print "Reading Dev and Test Data ..."
	Data_dev = Read_Data(filename = 'yelp_reviews_dev.json')
	Data_test = Read_Data(filename = 'yelp_reviews_test.json')
	(Xdevs, dictionaries,temp) = Process_Data(Data_dev, dictionaries)
	(Xtests, dictionaries,temp) = Process_Data(Data_test, dictionaries)
	
	print "Generating Dev and Test Predictions LogisticRegression ..."
	for i in range(2):
		(yPred_hard, yPred_soft) = LR.LogReg_PredictLabels(Xdevs[i], W[i])
		Write_predicted_rating_file(yPred_hard, yPred_soft,filename = 'dev.pred.LogReg.' + str(i))
		(yPred_hard, yPred_soft) = LR.LogReg_PredictLabels(Xtests[i], W[i])
		Write_predicted_rating_file(yPred_hard, yPred_soft,filename = 'test.pred.LogReg.' + str(i))
	
	print "Generating Dev and Test Predictions LinearSVM ..."
	for i in range(2):
		yPred_hard  = models[i].predict(Xdevs[i])
		Write_predicted_rating_file(yPred_hard, yPred_hard,filename = 'dev.pred.SVM.' + str(i))
		yPred_hard = models[i].predict(Xtests[i])
		Write_predicted_rating_file(yPred_hard, yPred_hard,filename = 'test.pred.SVM.' + str(i))
	
def Write_predicted_rating_file(yPred_hard, yPred_soft,filename = 'dev.pred'):
	f = open(filename, 'wb')
	for i in range(len(yPred_hard)):
		f.write(str(int(yPred_hard[i])) + " " + str(yPred_soft[i]) + '\n')	
	f.close()

''' Inefficient sklearn LinearSVM used in code '''
def SVMLibLinear_file(X,y, filesuffix):
	f = open('SVM_train_' + filesuffix, 'wb')
	X = np.transpose(dok_matrix(X))
	for i in range(X.shape[1]):
		string = str(y[i]) + " "
		iterator = X[:,i].items()
		iterator.sort(key=lambda tup: tup[0][0])
		for j in range(len(iterator)):
			string += (str(iterator[j][0][0]) + ':' + str(iterator[j][1]) + " ")
		f.write(string.strip() + '\n')	
	f.close()

Train()
