import sys
import time
import numpy as np
import pandas as pd
from sklearn import tree

assert len(sys.argv) == 3, 'train_tsv, test_tsv'

# train = np.genfromtxt(sys.argv[1], delimiter='\t')
# test = np.genfromtxt(sys.argv[2], delimiter='\t')
train = pd.read_csv(sys.argv[1], sep='\t', header=None).values
test = pd.read_csv(sys.argv[2], sep='\t', header=None).values

x_train, y_train = train[:,1:], train[:,0]
print 'train shape', x_train.shape, y_train.shape
# print 'train', x_train[0], y_train[0]
x_test, y_test = test[:,1:], test[:,0]
print 'test shape', x_test.shape, y_test.shape
# print 'test', x_test[0], y_test[0]

c = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=255)
print 'start train'
_t = time.time()
c.fit(x_train, y_train)
print 'training has taken', time.time() - _t

print 'train accuracy:', c.score(x_train, y_train)
print 'test accuracy:', c.score(x_test, y_test)

print 'params', c.get_params()
print 'num_nodes', c.tree_.node_count
