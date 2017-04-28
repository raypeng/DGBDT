import sys
import time
import numpy as np
import xgboost as xgb

assert len(sys.argv) == 3, 'train_tsv, test_tsv'
train = np.genfromtxt(sys.argv[1], delimiter='\t')
test = np.genfromtxt(sys.argv[2], delimiter='\t')

# read in data
dtrain = xgb.DMatrix(train[:,1:], label=train[:,0])
dtest = xgb.DMatrix(test[:,1:], label=test[:,0])
# specify parameters via map
param = {'max_leaves':255, 'objective':'multi:softprob', 'num_class': 5, 'nthread': 1}
num_round = 1 # should make it regular decision tree
_t = time.time()
bst = xgb.train(param, dtrain, num_boost_round=num_round)
print 'traing elapsed time:', time.time() - _t
# make prediction
pred_prob = bst.predict(dtrain)
pred = np.argmax(pred_prob, axis=1)
print 'train accuracy', 1. * np.sum(train[:,0] == pred) / pred.size
pred_prob = bst.predict(dtest)
pred = np.argmax(pred_prob, axis=1)
print 'test accuracy', 1. * np.sum(test[:,0] == pred) / pred.size
