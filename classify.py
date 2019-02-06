'''
Script for running classifiers on metaphlan or kmer featuers for metapheno.
Author: Nathan LaPierre
'''

import argparse, math, random, sys, time
import numpy as np
#np.random.seed(0)  # set random seed for keras neural nets
import xgboost as xgb
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from gcforest.gcforest import GCForest


def parseargs():    # handle user arguments
	parser = argparse.ArgumentParser(
		description = 'Run NN or xgboost on Metaphlan or kmer features.')
	parser.add_argument('feature_dir',
		help='Dir with features. If kmers, random-level, not Dataset level,\
				 i.e. /local/chelseaju/MetaPheno/T2D/random0/')
	parser.add_argument('--classifier', default = 'neuralnet',
		choices = ['neuralnet', 'xgboost', 'randomforest', 'svm', 'deepforest'],
		help = 'Which classifier to use.')
	parser.add_argument('--feature_type', default = 'metaphlan',
		choices = ['metaphlan', 'kmer'], help = 'Which features to use.')
	parser.add_argument('--folds', default = 'NONE',
		help = 'File with k folds. If unspecified, folds are chosen randomly.\
				 Only relevant if --feature_type is metaphlan (default).')
	parser.add_argument('--kmer_feats', default='top_12mers_1000',
		choices = ['top_12mers_1000', 'top_12mers_5000', 'top_12mers_10000',
		 			'selected_12mers_001', 'selected_12mers_005'],
		help = 'Which kmer features to use. Default: top_12mers_1000')
	parser.add_argument('--no_autoencoder', action = 'store_true',
		help = 'Use if you do not want to run the autoencoder.')
	args = parser.parse_args()
	return args


# Extract numerical features from a file containing metaphlan species relative
# 	abundances in float form
def metaphlan_get_features_and_labels(feature_dir):
	X, y = [], []  # the feature matrix and label vector
	with(open(feature_dir + 'metaphlan-vectors.txt','r')) as featfile:
		featfile.readline()  # skips header line with variable names
		for featline in featfile:
			X.append(featline.strip().split('\t'))
	# sort patients by name to make sure they're in same order as labels
	X.sort(key=lambda a: a[0])
	X = [[float(j) for j in X[i][1:]] for i in range(len(X))]  # convert to #s

	with(open(feature_dir + 'patient-label-map.txt', 'r')) as mapfile:
		for line in mapfile:
			y.append(line.strip().split('\t'))
		y.sort(key=lambda a: a[0])  # again, sort alphabetically
	y = [float(i[1]) for i in y]  # again, convert to numerical form
	#y = [1.0 if i[1]=='1' else -1.0 for i in y]  # convert to 1 and -1 labels
	return np.array(X), np.array(y)


# Given data X, labels y, pool of possible test patients, and fold size:
#  generate a random fold of patients for test set and partition X and y
#  into training and test data and labels, then return these arrays
def random_fold(pool, fold_size, X, y):
	test_indices = list(random.sample(pool, fold_size))
	pool = pool.difference(test_indices)  # remove this fold from set
	# Training set is patients not in test set
	train_indices = list(np.arange(len(X)))
	train_indices = [x for x in train_indices if x not in test_indices]

	# Now grab the actual feature vectors for train and test sets
	train_X, test_X = X[train_indices][:], X[test_indices][:]
	train_y, test_y = y[train_indices], y[test_indices]
	return pool, train_X, test_X, train_y, test_y


# Given --folds and which fold number, read the training and test patients
#  for that fold specified in the fold file, then parition X and y into training
#  and test data and labels, then return these arrays
def read_fold_metaphlan(folds, X, y, fold_num):
	# grab and parse the lines with train and test patients for this fold
	train_patients, test_patients = [], []
	with(open(folds, 'r')) as featfile:
		for i in range(fold_num * 2):
			featfile.readline()  # skip previous fold lines
		train_patients = featfile.readline().strip().split()[-1].split(',')
		test_patients = featfile.readline().strip().split()[-1].split(',')

	# X and y are in alphabetical order by patient, so here we get the
	#  appropriate indices into them by sorting alphabetically, then return
	train_X, test_X, train_y, test_y = [], [], [], []
	sorted_patients = sorted(train_patients + test_patients)
	train_indices = [sorted_patients.index(i) for i in train_patients]
	test_indices = [sorted_patients.index(i) for i in test_patients]
	train_X, test_X = X[train_indices][:], X[test_indices][:]
	train_y, test_y = y[train_indices], y[test_indices]
	return train_X, test_X, train_y, test_y


# Extract features from svm_light format, which is used for kmer features
def extract_kmer_features(filename):
	data, numfeats = [], 0
	# Since data in sparse matrix, need to determine number of features in total
	# This number is the highest-valued feature out of all lines
	with(open(filename, 'r')) as featfile:
		for line in featfile:
			maxfeat = int(line.split()[-1].split(':')[0])
			numfeats = maxfeat if maxfeat > numfeats else numfeats

	with(open(filename, 'r')) as featfile:
		for line in featfile:
			splits = line.strip().split()[1:]
			feats = [0 for i in range(numfeats)]
			# Here we must be careful because features are in sparse matrix form
			for item in splits:
				featnum, featval = item.split(':')
				feats[int(featnum)-1] = float(featval)
			data.append(feats)
	return np.array(data)


# Normalize kmer counts by dividing
# Note that test set normalization is based on training set (no data leakage)
def normalize_kmer_features(train_X, test_X):
	norms = [np.linalg.norm(vec) for vec in train_X]
	train_X = [train_X[i] / norms[i] for i in range(len(train_X))]
	test_X = [test_X[i] / norms[i] for i in range(len(test_X))]
	return np.array(train_X), np.array(test_X)


# Given a fold number and --feature_dir, read the kmers feature file and
#  training and test patients into training and test data and labels, and return
def generate_fold_kmer(feature_dir, kmer_feats, classifier, fold_num):
	# read from Dataset0{fold_num} directory to fill train and test, then return
	train_X, test_X, train_y, test_y = [], [], [], []
	fold_dir = feature_dir + 'Dataset0' + str(fold_num+1) + '/'

	# Read labels in for train and test set -- very simple
	with(open(fold_dir + 'train.txt', 'r')) as trainlabs:
		for line in trainlabs:
			label = int(line.strip().split()[1])
			label = 0.0 if label == -1 else 1.0  # conver -1 labels to 0
			train_y.append(label)
	with(open(fold_dir + 'test.txt', 'r')) as testlabs:
		for line in testlabs:
			label = int(line.strip().split()[1])
			label = 0.0 if label == -1 else 1.0  # conver -1 labels to 0
			test_y.append(label)
	train_y, test_y = np.array(train_y), np.array(test_y)

	# Read features vectors in, normalize, and return everything
	train_featfile = fold_dir + 'ttest_' + kmer_feats + '_train.data'
	test_featfile = fold_dir + 'ttest_' + kmer_feats + '_test.data'
	train_X = extract_kmer_features(train_featfile)
	test_X = extract_kmer_features(test_featfile)
	if classifier == 'neuralnet':  # normalization only needed with NN
		train_X, test_X = normalize_kmer_features(train_X, test_X)
	return train_X, test_X, train_y, test_y


# generates both metaphlan and kmer train/test data & labels
#  	in a way that is convenient for the run-metapheno.py script
def gen_folds_metapheno(feature_dir, num_folds):
	# data & labels for metaphlan & kmer features, also the latter normalized
	phlan_folds, kmer_folds, kmer_norms = [], [], []
	for fold_num in range(num_folds):
		train_X, test_X, train_y, test_y = [], [], [], []
		zero = '0' if fold_num < 9 else ''  # zero in dir name if single digit
		fold_dir = feature_dir + 'Dataset' + zero + str(fold_num+1) + '/'

		# Read labels in for train and test set -- very simple
		with(open(fold_dir + 'train.txt', 'r')) as trainlabs:
			for line in trainlabs:
				label = int(line.strip().split()[1])
				label = 0.0 if label == -1 else 1.0  # conver -1 labels to 0
				train_y.append(label)
		with(open(fold_dir + 'test.txt', 'r')) as testlabs:
			for line in testlabs:
				label = int(line.strip().split()[1])
				label = 0.0 if label == -1 else 1.0  # conver -1 labels to 0
				test_y.append(label)
		train_y, test_y = np.array(train_y), np.array(test_y)

		# Read features vectors in, normalize, and return everything
		train_featfile = fold_dir + 'ttest_top_12mers_1000_train.data'
		test_featfile = fold_dir + 'ttest_top_12mers_1000_test.data'
		train_X = extract_kmer_features(train_featfile)
		test_X = extract_kmer_features(test_featfile)
		kmer_folds.append([train_X, test_X, train_y, test_y])
		train_X, test_X = normalize_kmer_features(train_X, test_X)
		kmer_norms.append([train_X, test_X, train_y, test_y])
		# We can parse metaphlan features the same way
		train_featfile = fold_dir + 'metaphlan_lowest_train.data'
		test_featfile = fold_dir + 'metaphlan_lowest_test.data'
		train_X = extract_kmer_features(train_featfile)  # also works for phlan
		test_X = extract_kmer_features(test_featfile)
		phlan_folds.append([train_X, test_X, train_y, test_y])
	return phlan_folds, kmer_folds, kmer_norms


# given true and predicted values, generate accuracy, precision, recall, F1, AUC
def gen_eval_metrics(y_true, y_pred, svm_binary=[]):
	auc = roc_auc_score(y_true, y_pred)
	if len(svm_binary) > 0:  # a workaround for inaccurate SVC predict_proba
		y_pred = svm_binary
	tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
	for i in range(len(y_true)):
		if y_true[i] == 1:
			if y_pred[i] >= 0.5:
				tp += 1.0
			else:
				fn += 1.0
		else:
			if y_pred[i] < 0.5:
				tn += 1.0
			else:
				fp += 1.0
	if tp > 0.0:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = (2 * precision * recall) / (precision + recall)
	else:
		precision, recall, f1 = 0.0, 0.0, 0.0
	accuracy = (tp + tn) / (tp + fp + tn + fn)
	return [accuracy, precision, recall, f1, auc]


def build_and_fit_autoencoder(x_train, layers, opt, learn_rate):
	input_dim = len(x_train[0])
	input_data = Input(shape=(input_dim,))
	# define encoding dimensions as input_dim reduced by half per layer
	encoded_layer_sizes = [int(input_dim / (2**i)) for i in range(0, layers+1)]

	# encoder layers
	encoded = Dense(encoded_layer_sizes[1], activation='relu',
		kernel_initializer='random_normal')(input_data) # encoded layer
	for i in range(1, layers):  # we already have initial encoder layer
		encoded = Dense(encoded_layer_sizes[i+1], activation='relu',
			kernel_initializer='random_normal')(encoded) # encoded layer

	# decoder layers
	decoded = Dense(encoded_layer_sizes[layers-1], activation='relu',
		kernel_initializer='random_normal')(encoded) # decoded layer
	for i in range(1, layers):  # we already have initial decoder layer
		decoded = Dense(encoded_layer_sizes[layers - (i+1)], activation='relu',
			kernel_initializer='random_normal')(decoded) # decoded layer

	# build, compile, and fit the model
	autoencoder = Model(inputs=input_data, outputs=decoded)
	#opt = optimizers.adam(lr=0.001)
	if opt == 'adam':
		opt = optimizers.adam(lr=learn_rate/10.0)  # adam needs slower rate
	elif opt == 'sgd':
		opt = optimizers.sgd(lr=learn_rate)
	else:
		opt = optimizers.adagrad(lr=learn_rate)
	autoencoder.compile(optimizer=opt, loss='mean_squared_error')
	autoencoder.fit(x_train, x_train, epochs=10, batch_size=32, shuffle=True,
		verbose = 0)
	return autoencoder


def autoencoder_pretrain(train_X, test_X, layers=2, opt='adam', lr=0.001):
	autoencoder = build_and_fit_autoencoder(train_X, layers, opt, lr)
	# Now apply the learned encoding to the input train/test data
	for i in range(len(autoencoder.layers)):
		if i < 1:  # ignore first layer
			continue
		layer = autoencoder.layers[i]
		if layer.input_shape[1] < layer.output_shape[1]:
			break  # break if we are in a decoder layer

		wts = layer.get_weights()[0]
		train_X = np.matmul(train_X, wts)
		test_X = np.matmul(test_X, wts)
	return train_X, test_X


def build_and_fit_model(train_X, test_X, train_y, test_y,
						numlayers=5, dropout=0.25, opt='adam',learn_rate=0.001):
	model = Sequential()
	# define initial layer size and uniform scaling-down factor per layer
	layersize, layer_scale = len(train_X[0]), 1.0 / float(numlayers + 1)

	# input layer, then scaled down fully connected layers, then output layer
	model.add(Dense(layersize, input_dim=layersize,
		kernel_initializer='normal', activation='relu'))
	for i in range(numlayers):
		this_layersize = layersize - int(layersize * (layer_scale * (i+1)))
		model.add(Dense(this_layersize,
			kernel_initializer='normal', activation='tanh'))
		model.add(Dropout(dropout))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	#opt = optimizers.adam(lr=0.0001)
	if opt == 'adam':
		opt = optimizers.adam(lr=learn_rate/10.0)  # adam needs slower rate
	elif opt == 'sgd':
		opt = optimizers.sgd(lr=learn_rate)
	else:
		opt = optimizers.adagrad(lr=learn_rate)
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
	model.fit(train_X, train_y, epochs = 50, verbose=0)
	ypred = np.array([i[0] for i in model.predict(test_X, batch_size=32)])
	metrics = gen_eval_metrics(test_y, ypred)
	accuracy = metrics[0]
	print('Fold accuracy: ' + str(accuracy))
	#score = model.evaluate(test_X, test_y, batch_size=32)
	return metrics


# Wrapper script that runs autoencoder+neuralnet for run-metapheno
def run_autonn(train_X, test_X, train_y, test_y, auto_layers=1, fc_layers=5,
				dropout=0.25, opt='adam', lr=0.001, seed=0):
	if auto_layers > 0:
		train_X, test_X = autoencoder_pretrain(train_X, test_X, auto_layers,
													opt, lr)
	metrics = build_and_fit_model(train_X, test_X, train_y, test_y,
									fc_layers, dropout, opt, lr)
	return metrics


def run_xgb(train_X, test_X, train_y, test_y, depth=6, a=0.0, l=1.5, seed=0):
	param = {'max_depth':depth, 'num_round':20, 'eta':0.3, 'silent':1,
				'objective':'binary:logistic', 'eval_metric':['auc', 'error'],
				'alpha': a, 'lambda':l }
	if seed != 0:  # specific random seed entered
		param['seed'] = seed
		param['colsample_bytree'] = 0.5
		param['colsample_bylevel'] = 0.5
	train_xgb = xgb.DMatrix(train_X, label=train_y)
	test_xgb = xgb.DMatrix(test_X, label=test_y)
	bst = xgb.train(param, train_xgb)
	ypred = bst.predict(test_xgb)
	metrics = gen_eval_metrics(test_y, ypred)
	accuracy = metrics[0]

	#cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
	#accuracy = cor / len(test_y)
	print('Fold accuracy: ' + str(accuracy))
	return metrics


def run_rf(train_X, test_X, train_y, test_y, depth=6, est=100, c='gini',seed=0):
	clf = RandomForestClassifier(n_estimators=est, max_depth=depth,
		criterion=c, random_state=seed)
	clf.fit(train_X, train_y)
	ypred = np.array([i[1] for i in clf.predict_proba(test_X)])
	metrics = gen_eval_metrics(test_y, ypred)
	accuracy = metrics[0]

	#cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
	#accuracy = cor / len(test_y)
	print('Fold accuracy: ' + str(accuracy))
	return metrics


def run_svm(train_X, test_X, train_y, test_y, c=1.0, kern='linear', seed=0):
	clf = SVC(C=c, kernel=kern, random_state=seed, probability=True)
	clf.fit(train_X, train_y)
	ypred = np.array([i[1] for i in clf.predict_proba(test_X)])
	ypred_binary = clf.predict(test_X)
	metrics = gen_eval_metrics(test_y, ypred, svm_binary=ypred_binary)
	accuracy = metrics[0]

	#cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
	#accuracy = cor / len(test_y)
	print('Fold accuracy: ' + str(accuracy))
	return metrics


def get_toy_config(rounds, layers, seed):  # config file for deepforest
	config = {}
	ca_config = {}
	ca_config["random_state"] = seed
	ca_config["max_layers"] = layers
	ca_config["early_stopping_rounds"] = rounds
	ca_config["n_classes"] = 2
	ca_config["estimators"] = []
	ca_config["estimators"].append({'n_folds': 5, "type": "XGBClassifier",
		'max_depth':6, 'num_round':20, 'eta':0.3, 'silent':1,
		'objective':'binary:logistic', 'eval_metric':['auc', 'error'],
		'lambda':1.5 })
	ca_config["estimators"].append({"n_folds": 5,
		"type": "RandomForestClassifier", "n_estimators": 10,
		"max_depth": None, "n_jobs": -1})
	ca_config["estimators"].append({"n_folds": 5,
		"type": "ExtraTreesClassifier", "n_estimators": 10,
		"max_depth": None, "n_jobs": -1})
	ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
	config["cascade"] = ca_config
	return config


def run_gcforest(train_X, test_X, train_y, test_y, rounds=3, layers=100,seed=0):
	config = get_toy_config(rounds, layers, seed)
	gc = GCForest(config) # should be a dict
	X_train_enc = gc.fit_transform(train_X, train_y)
	ypred = np.array([i[1] for i in gc.predict_proba(test_X)])
	metrics = gen_eval_metrics(test_y, ypred)
	accuracy = metrics[0]

	#cor = sum([int(ypred[i] + 0.5) == test_y[i] for i in range(len(ypred))])
	#accuracy = cor / len(test_y)
	print('Fold accuracy: ' + str(accuracy))
	return metrics


def main():
	random.seed(0)
	args = parseargs()
	if not args.feature_dir.endswith('/'):
		args.feature_dir += '/'
	X, y = [], []  # data (feature vectors) and label vector
	if args.feature_type == 'metaphlan':
		X, y = metaphlan_get_features_and_labels(args.feature_dir)

	num_folds, fold_accs = 5, []  # for k-fold cross validation
	pool = set(np.arange(len(X)))  # pool to randomly select folds from
	for i in range(num_folds):
		print("Running Cross Validation %d" %(i+1))
		# Based on type of features and whether random or specified in file,
		#  choose right method to generate train and test data and labels
		if args.feature_type == 'metaphlan':
			if args.folds == 'NONE':  # generate random fold
				fold_size = int(len(X) / num_folds) + (
					(i+1) > (num_folds - (len(X) % num_folds)))
				pool, train_X, test_X, train_y, test_y = random_fold(pool,
				 											fold_size, X, y)
			else:
				train_X, test_X, train_y, test_y = read_fold_metaphlan(
															args.folds, X, y, i)
		else:
			train_X, test_X, train_y, test_y = generate_fold_kmer(
				args.feature_dir, args.kmer_feats, args.classifier, i)

		# Pretrain with autoencoder, make predictions, print & record results
		if not args.no_autoencoder:
			train_X, test_X = autoencoder_pretrain(train_X, test_X)
		if args.classifier == 'neuralnet':
			metrics = build_and_fit_model(train_X, test_X, train_y, test_y)
			#print('[Loss, Accuracy] = ' + str(score) + '\n')
		elif args.classifier == 'xgboost':
			metrics = run_xgb(train_X, test_X, train_y, test_y)
		elif args.classifier == 'deepforest':
			metrics = run_gcforest(train_X, test_X, train_y, test_y)
		elif args.classifier == 'svm':
			metrics = run_svm(train_X, test_X, train_y, test_y)
		else:
			metrics = run_rf(train_X, test_X, train_y, test_y)
		print(metrics)
		acc = metrics[0]
		fold_accs.append(acc)

	# Now print results for each fold and average over all folds
	print('\nAll fold accuracies: ' + str(fold_accs))
	print('Mean of all folds: ' + str(np.average(fold_accs)))
	var = np.var(fold_accs)
	print('Standard Deviation over all folds: ' + str(math.sqrt(var)))
	print('Variance over all folds: ' + str(var))


if __name__ == '__main__':
	main()
#

