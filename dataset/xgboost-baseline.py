import sys
import xgboost as xgb

assert len(sys.argv) == 3, 'train_tsv, test_tsv'
train = np.genfromtxt(sys.argv[1], delimiter='\t')
test = np.genfromtxt(sys.argv[2], delimiter='\t')

# read in data
dtrain = xgb.DMatrix(train[:,:-1], label=train[:,-1])
dtest = xgb.DMatrix(test[:,:-1], label=test[:,-1])
# specify parameters via map
param = {'max_leaves':64}
num_round = 1 # should make it regular decision tree
bst = xgb.train(param, dtrain, num_boost_round=num_round)
# make prediction
preds = bst.predict(dtest)
