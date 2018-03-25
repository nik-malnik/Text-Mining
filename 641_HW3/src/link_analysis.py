import sys
import numpy as np
import math
from scipy.spatial.distance import cosine
from scipy.sparse import lil_matrix
import pandas as pd
import timeit
from time import time
from sklearn.preprocessing import normalize
import pdb
from os import listdir
import zipfile
from os.path import isfile, join

''' Read the first 4 files needed for Page Rank calculations'''
def Read_Data():
	f = open('transition.txt','r')
	entries = []
	for line in f.readlines():
		entries.append([int(line.strip().split(' ')[0]),int(line.strip().split(' ')[1]),int(line.strip().split(' ')[2])])
	M_entries = np.array(entries)
	f.close()
	
	M = lil_matrix((np.max(M_entries),np.max(M_entries)),dtype=np.float)
	M.setdiag(1.0)
	for entry in M_entries:
		M[entry[0]-1,entry[1]-1] = entry[2]

	f = open('user-topic-distro.txt','r')
	entries = []
	for line in f.readlines():
		entry = line.strip().split(' ')
		entries.append({ 'user_id': int(entry[0]), 'query': int(entry[1]), 'topic_distribution' : entry[2:len(entry)] })
	PrTgU = entries
	
	f = open('query-topic-distro.txt','r')
	entries = []
	for line in f.readlines():
		entry = line.strip().split(' ')
		entries.append({ 'user_id': int(entry[0]), 'query': int(entry[1]), 'topic_distribution' : entry[2:len(entry)] })
	PrTgQ = entries

	f = open('doc_topics.txt','r')
	entries = []
	for line in f.readlines():
		entries.append([int(line.strip().split(' ')[0]),int(line.strip().split(' ')[1])])
	DocClass_entries = np.array(entries)
	
	DocClass = np.zeros((np.max(DocClass_entries,axis=0)[0],np.max(DocClass_entries,axis=0)[1]))
	for entry in DocClass_entries:
		DocClass[entry[0]-1,entry[1]-1] = 1
	
	return (M,DocClass,PrTgU,PrTgQ)

'''Reads the zip folder of the search relevance documents and scores '''
def Read_indri_files():
	
	path = "indri-lists"
	zip_ref = zipfile.ZipFile(path + '.zip', 'r')
	zip_ref.extractall(path)
	zip_ref.close()

	filenames = [f for f in listdir(path) if isfile(join(path, f))]
	
	entries = []
	for filename in filenames:
		f = open(path + "/" + filename,'r')
		user_id = filename.split('-')[0]
		query_id = filename.split('-')[1].split('.')[0]
		for line in f.readlines():
			components = line.strip().split(' ')
			entries.append({'user_id' : user_id, 'query_id': query_id,'doc_id' : components[2], 'original_rank' : int(components[3]), 'original_score' : float(components[4])})
		f.close()
	q_original = pd.DataFrame(entries)
	return q_original

''' Page Rank Algorithm, given a Transition Matrix M and the vector p '''
def PageRank(M,p, alpha = 0.2):
	
	M = normalize(M, norm='l1', axis=1)
	M = M.T
	r_history = [np.zeros((M.shape[0],1)),np.ones((M.shape[0],1))/M.shape[0]]
	# Ensure at lest 11 iterations to be able to generate the 10th iteration output file.
	while np.sum(np.abs(r_history[-1] - r_history[-2])) > 0.01 or len(r_history) < 12:
		#print "Convergence: difference ratio: " + str((np.linalg.norm(r_history[-2] - r_history[-1]))/(np.linalg.norm(r_history[-1]))) + " L2norm of ranks: " + str(np.linalg.norm(r_history[-1]))
		r = (1.0 - alpha)*(M*r_history[-1])
		r = r + (alpha*p)
		r_history.append(r)
	r = r_history[-1]/np.sum(r_history[-1])
	return (r,r_history[11]/np.sum(r_history[11]))

''' Topic Sensitive Page Rank. Creates the p vector for each topic '''
def TS_PageRank(M,DocClass, alpha = 0.2, beta = 0.1):
	
	p = np.ones((M.shape[0],1))/M.shape[0]
	p_topics = []
	r_topics = []
	r_topics_10th = []
	topic_vectors = (DocClass/np.sum(DocClass,axis=0))
	for topic in range(topic_vectors.shape[1]):
		# Calculate p vector from the topic tagging on documents
		p_topics.append(beta*p + (alpha - beta)*topic_vectors[:,topic].reshape(topic_vectors.shape[0],1))
		(temp1, temp2) = PageRank(M,p_topics[-1])
		r_topics.append(temp1)
		r_topics_10th.append(temp2)
	return (r_topics,r_topics_10th)

''' Combines the offline topic sensitive ranks with the P(t/u) or P(t/q) '''
def Online_TS_PageRank(M,DocClass,topic_probs,r_topics,r_topics_10th,filename):
		
	r_queries = []
	for query in topic_probs:
		r_query =  np.zeros((M.shape[0],1))
		r_query_10th =  np.zeros((M.shape[0],1))
		for topic in query['topic_distribution']:
			topic_weight = float(topic.strip().split(':')[1])
			topic_name = int(topic.strip().split(':')[0])
			# Weight the topic page rank by the topic distribution probabilities
			r_query = r_query + topic_weight*r_topics[topic_name-1]
			r_query_10th = r_query_10th + topic_weight*r_topics_10th[topic_name-1]
		r_queries.append(r_query)
		if query['query'] == 1 and query['user_id'] == 2:
			Write_PageRank_file(r_query_10th,filename)
	
	return r_queries

''' Calculates all of the GPR, QTSPR and PTSPR one by one '''
def Run_Ranking():
	(M,DocClass,PrTgU,PrTgQ) = Read_Data()
	p = np.ones((M.shape[0],1))/M.shape[0]
	
	# Standard Page Rank
	t = time()
	(r,r_10th) = PageRank(M,p)
	print "Time for GPR:" + str(time() - t)
	t = time()
	Write_PageRank_file(r_10th,"GPR-10.txt")
	
	# Topic Sensitive Page Rank
	(r_topics,r_topics_10th) = TS_PageRank(M,DocClass)
	print "Time for TSPR:" + str(time() - t)
	t = time()
	
	#Query based Topic Sensitive Page Rank
	r_queries = Online_TS_PageRank(M,DocClass,PrTgQ,r_topics,r_topics_10th,'QTSPR-U2Q1-10.txt')
	print "Time for QTSPR:" + str(time() - t)
	t = time()
	
	#User based Topic Sensitive Page Rank
	r_users = Online_TS_PageRank(M,DocClass,PrTgU,r_topics,r_topics_10th, 'PTSPR-U2Q1-10.txt')
	print "Time for PTSPR:" + str(time() - t)
	t = time()
	
	return (r,r_topics,r_queries,r_users)

''' converts the page ranks to a pandas dataframe format'''
def Read_PageRank_Output(r,r_queries,r_users,PrTgU,PrTgQ):
	entries = []
	for i in range(len(PrTgU)):
		for j in range(len(PrTgQ)):
			if PrTgU[i]['user_id'] == PrTgQ[j]['user_id'] and PrTgU[i]['query'] == PrTgQ[j]['query']:
				for k in range(len(r)):
					entries.append({'user_id': str(PrTgU[i]['user_id']), 'query_id': str(PrTgU[i]['query']), 'doc_id' : k + 1,
					'GPR': r[k][0],'QTSPR' : r_queries[j][k][0],'PTSPR' : r_queries[i][k][0] })
	df_rank = pd.DataFrame(entries)
	return df_rank	

def Write_PageRank_file(r,filename):
	print "\nWriting " + filename + "\n"
	f = open(filename, 'w')
	for i in range(len(r)):
		f.write(str(i+1) + " " + str(format(r[i][0],'f')) + '\n')	
	f.close()
	
def Write_treceval_file(df,method):
	f = open('output_' + method, 'w')
	for index, row in df.iterrows():
		f.write(str(row['user_id_x']) + "-" + str(row['query_id_x']) + ' Q0 ' + str(row['doc_id_x']) + ' ' + str(row['original_rank']) + ' ' + str(format(row[method],'f')) + ' run-1\n')	
	f.close()

''' Houses the weighting scheme and the generation of all 9 evaluation trec-eval files'''
def Generate_Evaluation_Files(r,r_queries,r_users,PrTgU,PrTgQ):
	q_original = Read_indri_files()
	df_rank = Read_PageRank_Output(r,r_queries,r_users,PrTgU,PrTgQ)
	
	q_original['join_key'] = map(lambda a,b,c:str(a) + "-" + str(b) + "." + str(c), q_original['user_id'],q_original['query_id'],q_original['doc_id'])
	df_rank['join_key'] = map(lambda a,b,c:str(a) + "-" + str(b) + "." + str(c), df_rank['user_id'],df_rank['query_id'],df_rank['doc_id'])
	df = pd.merge(q_original,df_rank,how="inner",left_on="join_key",right_on="join_key")

	w = 0.1
	t = time()
	for method in ['GPR','QTSPR','PTSPR']:
		
		df['NS_' + method] = map(lambda y: 10000.0*y, df[method])
		print "Time for NS_" + method + ":" + str(time() - t)
		t = time()
		
		df['WS_' + method] = map(lambda x,y: w*x + (1.0-w)*y, df['original_score'],df[method])
		print "Time for WS_" + method + ":" + str(time() - t)
		t = time()
		
		df['CM_' + method] = map(lambda x,y: w*x + (1.0-w)*math.log(y), df['original_score'],df[method])
		print "Time for CM_" + method + ":" + str(time() - t)
		t = time()
		
		Write_treceval_file(df,'NS_' + method)
		Write_treceval_file(df,'WS_' + method)
		Write_treceval_file(df,'CM_' + method)
		print "Time for write 3 " + method + " Evaluation files:" + str(time() - t)
	
	return df
	

(M,DocClass,PrTgU,PrTgQ) = Read_Data()
(r,r_topics,r_queries,r_users) = Run_Ranking()
df = Generate_Evaluation_Files(r,r_queries,r_users,PrTgU,PrTgQ)

