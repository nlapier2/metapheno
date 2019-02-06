"""
	Function: construct sparse matrix for metaphlan features in SVMlight format - extract the lowest abundance 
	To Run: python extract_metaphlan2_features.py
		-i metaphlan directory
		--train training samples
		--test test samples
		--train_out sparse matrix for training samples
		--test_out sparse matrix for testing samples
	Author: Chelsea Ju
"""

import sys, re, os, argparse, datetime, random

def echo(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(msg)))

def load_info(infile):
	info = []
	fh = open(infile, 'r')
	for line in fh:
		(pid, plabel) = line.rstrip().split()
		info.append( (pid, plabel))
	fh.close()
	return info

def extract_features_training(indir, train, train_out):

	echo("Loading training info")
	sample_info = load_info(train)

	features = {}
	f_count = 0

	outfh = open(train_out, 'w')

	for (s,l) in sample_info:
		if(s == "ERR260212" or s == "ERR260213"):
			filenames = [indir + "profiled_" + s + ".txt"]
		else:
			filenames = [indir + "profiled_" + s + "_1.txt", 
				    indir + "profiled_" + s + "_2.txt"]
		outfh.write("%s "%(l))
		abundance = [0.0]*f_count
		for f in filenames:
			fh = open(f, 'r')
			for line in fh:
				if line.startswith('k__Bacteria') and ('t__' in line or 'unclassified' in line):
					splits = line.split('\t')[0].split('|')  # splits line into different taxonomic levels
					lowlevel = splits[len(splits)-1]  # lowest level taxonomic classification
				
			#		if not features.has_key(lowlevel):
					if not lowlevel in features:
						features[lowlevel] = f_count
						f_count += 1

					## get feature ID	
					f_id = features[lowlevel]
					if(f_id >= len(abundance)):
						abundance.extend( [0.0]*(f_id - len(abundance) +1) )
					if(s == "ERR260212" or s == "ERR260213"):
						abundance[f_id] += (float(line.split('\t')[1].strip()) / 100.0) 
					else:
						abundance[f_id] += (float(line.split('\t')[1].strip()) / 200.0)
			fh.close()

		for i in range(len(abundance)):
			outfh.write("%d:%f " %(i, abundance[i]))
		outfh.write("\n")
	outfh.close()

	return features

def extract_features_testing(indir, test, test_out, features):
	echo("Loading info")
	sample_info = load_info(test)
	outfh = open(test_out, 'w')
	
	for (s,l) in sample_info:

		if(s == "ERR260212" or s == "ERR260213"):
			filenames = [indir + "profiled_" + s + ".txt"]
		else:
			filenames = [indir + "profiled_" + s + "_1.txt",
		
		indir + "profiled_" + s + "_2.txt"]

		outfh.write("%s " %(l))
		abundance = [0.0]*len(features.keys())
		for f in filenames:
			fh = open(f, 'r')
			for line in fh:
				if line.startswith('k__Bacteria') and ('t__' in line or 'unclassified' in line):
					splits = line.split('\t')[0].split('|')  # splits line into different taxonomic levels
					lowlevel = splits[len(splits)-1]  # lowest level taxonomic classification

					#if( features.has_key(lowlevel)):
					if( lowlevel in features ):	
						if(s == "ERR260212" or s == "ERR260213"):
							abundance[features[lowlevel]] += (float(line.split('\t')[1].strip()) / 100.0)
						else:
							abundance[features[lowlevel]] += (float(line.split('\t')[1].strip()) / 200.0)

			fh.close()

		for i in range(len(abundance)):
			outfh.write("%d:%f " %(i, abundance[i]))
		outfh.write("\n")
	outfh.close()

def main(parser):
	option = parser.parse_args()
	indir = option.indir
	train = option.train
	test = option.test
	train_out = option.train_out
	test_out = option.test_out

	echo("Program Starts")
	
	echo("Extracting Training Features")
	features = extract_features_training(indir, train, train_out)

	echo("Extracting Training Features")
	extract_features_testing(indir, test, test_out, features)

	echo("Done")




if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="extract_metaphlan2_features.py")
	parser.add_argument("-i", "--indir", dest="indir", type=str, help="metaphlan files", required = True)
	parser.add_argument("--train", dest="train", type=str, help="training samples", required = True)
	parser.add_argument("--test", dest="test", type=str, help="testing samples", required = True)
	parser.add_argument("--train_out", dest="train_out", type=str, help="sparse matrix for training", required = True)
	parser.add_argument("--test_out", dest="test_out", type=str, help="sparse matrix for testing", required = True)
	main(parser)
