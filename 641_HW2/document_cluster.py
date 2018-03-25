import sys
import numpy as np
import math
from scipy.spatial.distance import cosine
#import pandas as pd
import timeit


def Read_Data():
	f = open('HW2_dev.df','r')
	freq_of_idx = {}
	for line in f.readlines():
		freq_of_idx[int(line.strip().split(':')[0])] = int(line.strip().split(':')[1])
	f.close()
	
	f = open('HW2_test.df','r')
	freq_of_idx_test = {}
	for line in f.readlines():
		freq_of_idx_test[int(line.strip().split(':')[0])] = int(line.strip().split(':')[1])
	f.close()
	
	f = open('HW2_dev.dict','r')
	word_of_idx = {}
	for line in f.readlines():
		word_of_idx[int(line.strip().split(' ')[1])] = line.strip().split(' ')[0]
	f.close()
	
	f = open('HW2_dev.dict','r')
	idx_of_word = {}
	for line in f.readlines():
		idx_of_word[line.strip().split(' ')[0]] = int(line.strip().split(' ')[1])
	f.close()
	
	f = open('HW2_test.dict','r')
	word_of_idx_test = {}
	for line in f.readlines():
		word_of_idx_test[int(line.strip().split(' ')[1])] = line.strip().split(' ')[0]
	f.close()


	f = open('HW2_dev.docVectors','r')
	docs = []
	tokens = []
	for line in f.readlines():
		tokens = line.strip().split(' ')
		doc = {}
		for token in tokens:
			doc[int(token.strip().split(':')[0])] =  int(token.strip().split(':')[1])
		docs.append(doc)
	f.close()
	
	f = open('HW2_test.docVectors','r')
	docs_test = []
	tokens = []
	for line in f.readlines():
		tokens = line.strip().split(' ')
		doc = {}
		for token in tokens:
			doc[int(token.strip().split(':')[0])] =  int(token.strip().split(':')[1])
		docs_test.append(doc)
	f.close()

	return [docs, docs_test, freq_of_idx, freq_of_idx_test, word_of_idx, idx_of_word, word_of_idx_test]

def Create_TF_Matrix(docs, vocab_size):
	tf_mat = np.zeros((len(docs),vocab_size))
	i = 0
	for doc in docs:
		for key, val in doc.iteritems():
			tf_mat[i][key] = val 
		i = i + 1
	return tf_mat

def Create_TF_Matrix_Test(docs_test, word_of_idx_test, idx_of_word,vocab_size):
	tf_mat = np.zeros((len(docs_test),vocab_size))
	i = 0
	for doc in docs_test:
		for key, val in doc.iteritems():
			try:
				key_map = idx_of_word[word_of_idx_test[key]]
				tf_mat[i][key] = val
			except:
				pass 
		i = i + 1
	return tf_mat

def Create_TFIDF_Matrix(docs,freq_of_idx, vocab_size):
	tf_mat = Create_TF_Matrix(docs, vocab_size)
	N_docs = tf_mat.shape[0]
	idf_vec = np.zeros(vocab_size)
	for key, value in freq_of_idx.iteritems():
		idf_vec[key] = math.log(N_docs/(value))
	tfidf_mat = np.multiply(tf_mat,idf_vec)
	return tfidf_mat

def Create_TFIDF_Matrix_Test(docs_test,freq_of_idx, word_of_idx_test, idx_of_word, vocab_size):
	tf_mat = Create_TF_Matrix_Test(docs_test, word_of_idx_test, idx_of_word,vocab_size)
	N_docs = tf_mat.shape[0]
	idf_vec = np.zeros(vocab_size)
	for key, value in freq_of_idx.iteritems():
		idf_vec[key] = math.log(N_docs/(value))
	tfidf_mat = np.multiply(tf_mat,idf_vec)
	return tfidf_mat

def Calculate_Kmeans_Centroids(data,classification,k):
	centroids = np.zeros((k,data.shape[1]))
	cluster_size = np.ones(k)
	i = 0
	for cluster in classification:
		centroids[cluster] = centroids[cluster] + data[i]
		cluster_size[cluster] = cluster_size[cluster] + 1
		i = i + 1
	
	for i in range(k):
		if cluster_size[i] == 0:
			cluster_size[i] = 1
	
	centroids = centroids/cluster_size[:,None]
		
	return centroids

# Using Iterative Cosine Similarity
def Classify_Kmeans_obs(data,k,centroids):
	classification = []
	for doc in data:
		least_distance = -1
		most_similar_cluster = -1
		for cluster in range(k):
			distance = cosine(centroids[cluster],doc)
			if least_distance == -1 or least_distance > distance:
				most_similar_cluster = cluster
				least_distance = distance
		classification.append(most_similar_cluster)
	return classification

#Using Norm-d Matrix Multiplication
def Classify_Kmeans_obs_II(data,k,centroids):
	centroids = np.array(centroids)
	centroids_norm = np.transpose(np.transpose(centroids)/np.linalg.norm(np.transpose(centroids),axis=0))
	data_norm = np.transpose(data)/np.linalg.norm(np.transpose(data),axis=0)
	distance_mat = np.dot(centroids_norm,data_norm)
	classification = np.argmin(distance_mat,axis=0)
	return classification

# Using Iterative Cosine Similarity
'''
def Select_Kmeans_plus_Centroids(data,k):
	centroids = []
	centroids.append(data[int(np.random.rand(1)*data.shape[0]),:])
	while len(centroids) < k:
		D = []
		for i in range(data.shape[0]):
			least_distance = -1
			for j in range(len(centroids)):
				distance = cosine(centroids[j],data[i])
				if least_distance == -1 or least_distance > distance:
					least_distance = distance
			D = D + [i]*int(least_distance*10.0)
		centroids.append(data[D[int(np.random.rand(1)*len(D))],:])
	
	return np.array(centroids)
'''

#Using Norm-d Matrix Multiplication
def Select_Kmeans_plus_Centroids(data,k):
	centroids = []
	centroids.append(data[int(np.random.rand(1)*data.shape[0]),:])
	centroids = np.array(centroids)
	while centroids.shape[0] < k:
		centroids_norm = np.transpose(np.transpose(centroids)/np.linalg.norm(np.transpose(centroids),axis=0))
		data_norm = np.transpose(data)/np.linalg.norm(np.transpose(data),axis=0)
		distance_mat = np.dot(centroids_norm,data_norm)
		prob_dist = np.min(distance_mat,axis=0)
		idx = np.random.choice(data.shape[0],1,p=prob_dist/np.sum(prob_dist))[0]
		centroids = np.append(centroids,[np.transpose(data[idx])],axis=0)
	
	return centroids			

def Get_Kmeans_Clusters(data, k = 2, initialization = 0):
	if initialization == 0:
		centroids = np.random.rand(k,data.shape[1])
	else:
		centroids = Select_Kmeans_plus_Centroids(data,k)
	
	centroid_move = True
	i = 0
	while centroid_move:
		classification = Classify_Kmeans_obs(data,k,centroids)
		centroids_new = Calculate_Kmeans_Centroids(data,classification,k)
		
		'''
		centroid_move = 0.0
		for i in range(len(centroids)):
			centroid_move = centroid_move + cosine(centroids[i],centroids_new[i])
		print "Centroids moving" + str(centroid_move)
		'''
		
		# Change stopping criteria
		if np.array_equal(centroids_new,centroids):
			centroid_move = False
		centroids = centroids_new
		i = i + 1
		
	print str(k) + "Centroids Stable after moving " + str(i) + " times"
	return [centroids, classification]

def Search_Optimal_K(unique_ks=40, k_min = 0, iter_per_k = 3):
	
	[docs, docs_test, freq_of_idx, freq_of_idx_test, word_of_idx, idx_of_word, word_of_idx_test] = Read_Data()
	tf_mat = Create_TF_Matrix(docs, len(freq_of_idx))
	tfidf_mat = Create_TFIDF_Matrix(docs, freq_of_idx,len(freq_of_idx))
	
	ks = np.delete(np.arange(unique_ks + 1)*5 + k_min,0)
	f1_scores = []
	
	for iteration in range(iter_per_k):
		for k in ks:
			print "Running k means original"
			try:
				start = timeit.default_timer()
				[centroids, classification] = Get_Kmeans_Clusters(tf_mat,k=k)
				duration = timeit.default_timer() - start
				Write_Cluster_Output(classification)
				f1_scores.append({ 'algortihm' : 'basic', 'macro_f1_score' : Evaluate_F1_Score('HW2_dev.gold_standards','nmalik1-dev-cluster.txt'), 'k' : k, 'iteration' : iteration + 1, 'duration' : duration})
			except:
				print "Error!!!!"
				
			print "Running k means++"
			try:
				start = timeit.default_timer()
				[centroids, classification] = Get_Kmeans_Clusters(tf_mat,k=k,initialization=1)
				duration = timeit.default_timer() - start
				Write_Cluster_Output(classification)
				f1_scores.append({ 'algortihm' : 'kmeansplusplus', 'macro_f1_score' : Evaluate_F1_Score('HW2_dev.gold_standards','nmalik1-dev-cluster.txt'), 'k' : k, 'iteration' : iteration + 1, 'duration' : duration})
			except:
				print "Error!!!!"
				
			print "Running custom"
			try:
				start = timeit.default_timer()
				[centroids, classification] = Get_Kmeans_Clusters(tfidf_mat,k=k,initialization=1)
				duration = timeit.default_timer() - start
				Write_Cluster_Output(classification)
				f1_scores.append({ 'algortihm' : 'custom', 'macro_f1_score' : Evaluate_F1_Score('HW2_dev.gold_standards','nmalik1-dev-cluster.txt'), 'k' : k, 'iteration' : iteration + 1, 'duration' : duration})
			except:
				print "Error!!!!"
			
			# If the results need to be saved.
			#pd.DataFrame(f1_scores).to_csv('/home/malnik/python_env/641_HW2/Result.csv', sep = '\t', encoding = 'utf-8')
	
	return f1_scores

def Classify_Test_Data(centroids):
	[docs, docs_test, freq_of_idx, freq_of_idx_test, word_of_idx, idx_of_word, word_of_idx_test] = Read_Data()
	
	tfidf_mat = Create_TFIDF_Matrix_Test(docs_test,freq_of_idx, word_of_idx_test, idx_of_word,len(freq_of_idx))
	classification = Classify_Kmeans_obs(tfidf_mat,centroids.shape[0],centroids)
	Write_Cluster_Output(classification, filename = 'nmalik1-test-cluster.txt')

def Write_Cluster_Output(classification, filename = 'nmalik1-dev-cluster.txt'):
	out = ""
	for i in range(len(classification)):
		out = out + str(i) + " " + str(classification[i]) + "\n"
	
	f = open(filename,'wb')
	f.write(out)
	f.close()

def Evaluate_F1_Score(true_f,predicted_f):

	input = open(predicted_f, "r")
	gold = open(true_f, "r")
	
	sysClusters = []
	goldClusters = []
	
	for line in input:
		tokens = line.strip().split(" ")
		cluster = int(tokens[1])
		doc = int(tokens[0])
		while len(sysClusters) <= cluster:
			sysClusters.append([])
		sysClusters[cluster].append(doc)
	input.close()
	
	goldDict = dict()
	eventCounter = 0
	docCounter = 0
	for line in gold:
		cluster = line.strip()
	
		if cluster != "unlabeled":
			if cluster not in goldDict:
				goldDict[cluster] = eventCounter
				eventCounter += 1
	
			clusterID = goldDict[cluster]
			while len(goldClusters) <= clusterID:
				goldClusters.append([])
			goldClusters[clusterID].append(docCounter)
	
		docCounter += 1
	gold.close()
	
	
	# for each gold cluster, find the system cluster that maximizes F1
	clusterF1s = []
	for goldCluster in goldClusters:
		bestF1 = -1
	
		for sysCluster in sysClusters:
			tp = 0
			fp = 0
			fn = 0
			for item in goldCluster:
				if item in sysCluster:
					tp += 1.0
				else:
					fn += 1.0
			for item in sysCluster:
				if item not in goldCluster:
					fp += 1.0
	
			# if none match, just ignore	
			if tp == 0:
				continue
	
			precision = tp / (tp+fp)
			recall = tp / (tp+fn)
			f1 = 2*precision*recall/(precision+recall)
	
			if f1 > bestF1:
				bestF1 = f1
		
		clusterF1s.append(bestF1)
	
	macroF1 = 0
	for item in clusterF1s:
		macroF1 += item
	macroF1 = macroF1 / len(clusterF1s)
	
	print "Macro F1 = " + str(macroF1)
	return macroF1


Search_Optimal_K(unique_ks=40, k_min = 0, iter_per_k = 3)
