'''
Script for running metapheno experiments, using the classify.py script.
First runs hyperparameter grid search for each algorithm based on 5-fold CV.
Then, using top parameters, test on different random seeds and fold partitions.
Author: Nathan LaPierre
'''

import argparse, itertools, math
import numpy as np
np.random.seed(0)  # set random seed for keras neural networks
import classify


# define globals
classifiers = ['svm', 'rf', 'xgb', 'gcforest', 'autonn']
metric_names = ['Accuracy','Precision','Recall','F1 Score', 'AUC-ROC']

# previously-validated best params on T2D data with full grid search
pre_best_phlan = {
	'svm': [[0,0,0], [0.75, 'linear']],
	'rf': [[0,0,0], [10, 50, 'entropy']],
	'xgb': [[0,0,0], [10, 0.5, 0.5]],
	'gcforest': [[0,0,0], [3, 0]],
	'autonn': [[0,0,0], [1, 5, 0, 'adagrad', 0.001]]
}
pre_best_kmer = {
	'svm': [[0,0,0], [0.25, 'linear']],
	'rf': [[0,0,0], [6, 10, 'gini']],
	'xgb': [[0,0,0], [6, 0, 1.5]],
	'gcforest': [[0,0,0], [5, 0]],
	'autonn': [[0,0,0], [1, 5, 0.25, 'adagrad', 0.001]]
}


def parseargs():    # handle user arguments
	parser = argparse.ArgumentParser(
		description = 'Run NN or xgboost on Metaphlan or kmer features.')
	parser.add_argument('disease', choices=['obesity', 't2d', 'wt2d', 'wt2d_10folds'],
		help='Which disease to analyze')
	parser.add_argument('--grid_search', default='comprehensive',
		choices = ['none', 'small', 'comprehensive'],
		help = 'Amount of grid search. Choices: none, small, comprehensive.')
	parser.add_argument('--seed_search', action='store_true',
		help = 'Whether to search across random seeds.')
	args = parser.parse_args()
	return args


# Setup experiment parameters based on user arguments
def setup(args):
	seeds = [0, 2736136, 741180, 1057096, 2505548, 8988168] if args.seed_search else [0]
	feature_dir = args.disease + '_data/random0/'
	phlan_folds, kmer_folds, kmer_norms = classify.gen_folds_metapheno(feature_dir, args.folds)

	# set hyperparameter options for grid search
	params = {
		'svm': [[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], ['linear', 'poly']],
		'rf': [[2, 6, 10], [10, 50, 100], ['gini', 'entropy']],
		'xgb': [[2, 6, 10], [0, 0.25, 0.5], [0.5, 1.0, 1.5]],
		'gcforest': [[3 ,5], [0, 50, 100]],
		'autonn': [[0,1,2,3], [3,5,10], [0,0.25,0.5], ['sgd','adagrad','adam'], [0.01,0.001]]
	}
	param_names = {
		'svm': ['C', 'kernel'],
		'rf': ['max_depth', 'n_estimators', 'criterion'],
		'xgb': ['max_depth', 'alpha', 'lambda'],
		'gcforest': ['early_stopping_rounds', 'max_layers'],
		'autonn': ['encoder_layers', 'feedforward_layers', 'dropout',
						'optimizer', 'learning_rate']
	}
	if args.grid_search == 'small':  # remove some obviously bad options
		params['svm'][1] = ['linear']
		params['gcforest'][1] = [0]
		params['autonn'] = [[1,2], [3,5,10], [0,0.25,0.5], ['adagrad','adam'], [0.001]]
	return seeds, params, param_names, phlan_folds, kmer_folds, kmer_norms


# given acc/precision/recall/f1/auc for multiple runs, compute mean, SD, variance of each
def process_multirun_metrics(multirun_metrics):
	# transpose gets same metric across runs in a row (array),
	#  	then compute mean/SD/var for each row (array)
	row_metrics = np.array(multirun_metrics).T
	return [[np.mean(i), math.sqrt(np.var(i)), np.var(i)] for i in row_metrics]


# find best hyperparameters per feature type for each classifier via grid search
def find_best_params(args, feature_type, params, param_names, phlan_folds, kmer_folds, kmer_norms):
	print('Running grid search for ' + str(feature_type) + '\n\n\n')
	best = {clf: [[[0]], []] for clf in classifiers}  # best results & params
	for clf in classifiers:
		# generate all combinations of parameters, and get parameter names
		param_combos = list(itertools.product(*params[clf]))
		#						*[v for k,v in params[clf].items()]))
		param_combos = [list(i) for i in param_combos]
		clf_param_names = param_names[clf]
		for p in param_combos:
			fold_mets = []
			print('Running ' + clf + ' classifier with parameters: ' +
					str(clf_param_names) + ' = ' + str(p))
			for i in range(args.folds):
				print("Running Cross Validation %d" %(i+1))
				if feature_type == 'metaphlan':
					train_X, test_X, train_y, test_y = phlan_folds[i]
				else:
					train_X, test_X, train_y, test_y = kmer_folds[i]
					if clf == 'autonn':
						train_X, test_X, train_y, test_y = kmer_norms[i]

				# run appropriate classifier & params
				metrics = eval('classify.run_'+clf+
								'(train_X, test_X, train_y, test_y, *p)')
				fold_mets.append(metrics)

			# printing information for user
			multirun_metrics = process_multirun_metrics(fold_mets)
			for m in range(len(metric_names)):
				print(metric_names[m] + ' [mean, SD, variance] : ' + str(multirun_metrics[m]))
			print('\n\n\n')  # spacing between runs

			# determine if best result for this classifier
			avg_acc = multirun_metrics[0][0]
			if avg_acc > best[clf][0][0][0]:
				best[clf] = [multirun_metrics, p]
	return best


def seed_partition_search(args, feature_type, best, seeds):
	print('\n\n\nRunning seed/partition tests for', str(feature_type), '\n\n\n')
	all_res = {clf:[] for clf in classifiers}  # all results across all settings
	for foldnum in range(args.partitions):
		feature_dir = args.disease + '_data/random' + str(foldnum) + '/'
		phlan_folds, kmer_folds, kmer_norms = classify.gen_folds_metapheno(feature_dir, args.folds)

		for cur_seed in seeds:
			np.random.seed(cur_seed)  # set random seed for neural networks
			for clf in classifiers:
				clf_best = best[clf][1]  # best parameters for this clf
				fold_mets = []
				print('Running ' + clf + ' classifier with partition: ' +
						str(foldnum) + ' and random seed ' + str(cur_seed))
				for i in range(args.folds):
					print("Running Cross Validation %d" %(i+1))
					if feature_type == 'metaphlan':
						train_X, test_X, train_y, test_y = phlan_folds[i]
					else:
						train_X, test_X, train_y, test_y = kmer_folds[i]
						if clf == 'autonn':
							train_X, test_X, train_y, test_y = kmer_norms[i]

					metrics = eval('classify.run_'+clf+
									'(train_X, test_X, train_y, test_y, '+
									'*clf_best, seed=cur_seed)')
					fold_mets.append(metrics)

				# printing and recording information
				multirun_metrics = process_multirun_metrics(fold_mets)
				for m in range(len(metric_names)):
					print(metric_names[m] + ' [mean, SD, variance] : ' + str(multirun_metrics[m]))
				print('\n\n\n')  # spacing between runs
				setting = 'Partition ' + str(foldnum) + ' Seed ' + str(cur_seed)
				all_res[clf].append([setting, multirun_metrics])
	return all_res


def print_best(best, feature_type, param_names):
	print('\n\n\nBest parameters and results for each classifier for ' +
			str(feature_type) + ' features...')
	for clf in classifiers:
		clf_param_names = param_names[clf]
		best_params = list(zip(clf_param_names, best[clf][1]))
		print(clf, 'best parameters:', str(best_params))
		for m in range(len(metric_names)):
			print(metric_names[m] + ' [mean, SD, variance] : ' + str(best[clf][0][m]))
		print('\n')  # spacing between runs


def print_seed_partition_res(feature_type, results, seeds):
	print('\n\n\n' + str(feature_type) + ' results across seed/partition settings:')
	for clf in classifiers:
		setting_metrics = [res[1] for res in results[clf]]
		across_setting_metrics = []
		for setting in setting_metrics:
			metric_means = []
			for metric in setting:
				metric_means.append(metric[0])
			across_setting_metrics.append(metric_means)

		print('\nMetrics for', clf, 'across settings:')
		multirun_metrics = process_multirun_metrics(across_setting_metrics)
		for m in range(len(metric_names)):
			print(metric_names[m] + ' [mean, SD, variance] : ' + str(multirun_metrics[m]))
		print('\n')  # spacing between runs


def main():
	args = parseargs()  # get user arguments, set up experiment parameters
	if args.disease == 'wt2d_10folds':
		args.folds = 10
	else:
		args.folds = 5
	args.partitions = 5
	seeds, params, param_names, phlan_folds, kmer_folds, kmer_norms = setup(args)

	# perform grid search if user desires, otherwise use previously-validated settings
	if args.grid_search == 'none':
		phlan_best = pre_best_phlan
		kmer_best = pre_best_kmer
	else:
		phlan_best = find_best_params(args, 'metaphlan', params, param_names, phlan_folds, kmer_folds, kmer_norms)
		kmer_best = find_best_params(args, 'kmer', params, param_names, phlan_folds, kmer_folds, kmer_norms)

	# evaluate results across different random seeds and data partitions
	print('\n\n\nRunning all settings across random seeds' +
			' and metaphlan settings across different paritions...')
	phlan_res = seed_partition_search(args, 'metaphlan', phlan_best, seeds)
	kmer_res = seed_partition_search(args, 'kmer', kmer_best, seeds)

	# Now print out results at the end
	if not (args.grid_search == 'none'):
		print_best(phlan_best, 'metaphlan', param_names)
		print_best(kmer_best, 'kmer', param_names)
	print_seed_partition_res('metaphlan', phlan_res, seeds)
	print_seed_partition_res('kmer', kmer_res, seeds)


if __name__ == '__main__':
	main()
#

