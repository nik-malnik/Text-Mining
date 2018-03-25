import sys
from math import pow, log

def NDCG(actual, predict, k):
    I = sorted(range(len(predict)), key=lambda k: predict[k], reverse = True)
    I = I[:k]
    dcg = 0.0;
    pos = 1;
    for i in I:
        dcg += (pow(2,actual[i]) - 1) / (log(pos + 1))
        pos += 1

    I = sorted(range(len(actual)), key=lambda k: actual[k], reverse = True)
    I = I[:k]
    idcg = 0.0;
    pos = 1;
    for i in I:
        idcg += (pow(2,actual[i]) - 1) / (log(pos + 1))
        pos += 1

    return dcg/idcg


def Evaluate(your_prediction = '', querycsv = 'dev.csv', gold_prediction = 'dev.golden'):

	# read user item pairs
	itemQuery = list()
	userQuery = list()
	for l in open(querycsv):
		ss = l.strip().split(',')
		itemQuery.append(int(ss[0]))
		userQuery.append(int(ss[1]))

    # read prediction
	prediction = list()
	for l in open(your_prediction):
		prediction.append(float(l.strip()))

    # read gold answer
	gold = list()
	for l in open(gold_prediction):
		gold.append(float(l.strip()))

    # initialize ndcg
	ndcg = [0,0,0] # ndcg@10,20,30
	for user in set(userQuery):
		#print 'evaluating user %d' % (user)
		idx = [i for i, j in enumerate(userQuery) if j == user]
		pred = [j for i, j in enumerate(prediction) if i in idx]
		actual = [j for i, j in enumerate(gold) if i in idx]
		for i in [0,1,2]:
			ndcg[i] += NDCG(actual, pred, (i+1)*10)

    # print result
	print "ndcg @ 10,20,30: %f, %f, %f" % (ndcg[0]/len(set(userQuery)), ndcg[1]/len(set(userQuery)), ndcg[2]/len(set(userQuery)))
