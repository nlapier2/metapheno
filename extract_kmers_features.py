"""
	Function: construct matrix for significant kmers in svmlight format
	To Run: python extract_sig_features_sparse_v2.py
	Input:	-s file of significantly different kmers
		-i kmer count directory
		--train training samples
		--train_out sparse matrix for training samples
	Author: Chelsea Ju
	Note: need to export Jellyfish package to python path export PYTHONPATH=$PYTHONPATH:/home/chelseaju/MetaK/Jellyfish/jellyfish_software/lib/python3.6/site-packages/
"""

import sys, re, os, argparse, datetime, random
import jellyfish

def echo(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(msg)))

def load_sig_features(sig_file):
	sig_features = [] 
	fh = open(sig_file, 'r')
	for line in fh:
		kmer = line.rstrip()
		sig_features.append(kmer)
	fh.close() 

	return sig_features

def load_samples(filename):
	samples = []
	fh = open(filename, 'r')
	for line in fh:
		(sid, label) = line.rstrip().split()
		samples.append((sid, label))
	fh.close()
	return samples

def export_sparse_features(sigmers, sample, indir, outfile):

	outfh = open(outfile, 'w')
	i = 0 
	for (s,l) in sample:
		i = i + 1

		if(i % 50 == 0):
			echo("\t\t ... Completed %f" %(float(i) / float(len(sample))))


		filename = indir + s + "_count.jf"
		qf = jellyfish.QueryMerFile(filename)
		
		outfh.write("%s " %(l))
		j = 0
		for mer in sigmers:
			j = j + 1
			jmer = jellyfish.MerDNA(mer)
			jmer.canonicalize()
			
			if(qf[jmer] > 0):
				outfh.write("%d:%d " %(j, qf[jmer]))
				
		#	outfh.write("%d\t%d\t%d\n" %(i, sigmers[mer], qf[jmer]))
		outfh.write("\n")
	outfh.close()

def main(parser):
	option = parser.parse_args()
	sig_file = option.sig
	indir = option.indir
	train = option.train
	train_out = option.train_out

	echo("Start Processiong")
	echo("Load Significant Features")
	sigmers = load_sig_features(sig_file)

	echo("Load Samples Info")
	train_sample = load_samples(train)

	echo("Export Feature Matrix for Training")
	export_sparse_features(sigmers, train_sample, indir, train_out)
	
	echo("Done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="extract_kmers_features.py")
	parser.add_argument("-s", "--sig", dest="sig", type=str, help="list of significant features", required = True)
	parser.add_argument("-i", "--indir", dest="indir", type=str, help="jellyfish count files", required = True)
	parser.add_argument("--train", dest="train", type=str, help="training samples", required = True)
	parser.add_argument("--train_out", dest="train_out", type=str, help="sparse matrix for training", required = True)
	main(parser)
