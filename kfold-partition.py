import numpy as np
import random, sys

if len(sys.argv) != 4:
	print('Usage: "python kfold-partition.py <# folds> <label file> <output file>"')
	sys.exit()

patients = []
with(open(sys.argv[2], 'r')) as mapfile:
	for line in mapfile:
		patients.append(line.split()[0])  # get patient name
patients = np.array(patients)

open(sys.argv[3], 'w').close()  # clear old file since we will append later
num_folds = int(sys.argv[1])
pool = set(np.arange(len(patients)))  # pool of patients to randomly select folds from

for i in range(num_folds):
		# Randomly select patients from pool to form this fold
		if i == (num_folds-1):  # last fold
			test_sample_id = list(pool)
		else:
			# fold size same except for remainder
			fold_size = int(len(patients) / num_folds) + ((i+1) > (num_folds - (len(patients) % num_folds)))
			test_sample_id = list(random.sample(pool, fold_size))
			pool = pool.difference(test_sample_id)  # remove this fold from set

		train_sample_id = list(np.arange(len(patients)))
		# Training set is patients not in test set
		train_sample_id = [x for x in train_sample_id if x not in test_sample_id]
		train_cur, test_cur = patients[train_sample_id], patients[test_sample_id]
		train_cur, test_cur = list(train_cur), list(test_cur)

		with(open(sys.argv[3], 'a')) as outfile:
			outfile.write('Fold ' + str(i+1) + ' training set: ')
			outfile.write(','.join(train_cur) + '\n')
			outfile.write('Fold ' + str(i+1) + ' test set: ')
			outfile.write(','.join(test_cur) + '\n')
#

