"""
	Function: export the p-value and mean of each kmer feature from t-test,
	k-mer occurrences are normalized by total number of kmers calculated from jellyfish dump
	To Run: python compute_kmers_pval.py -i T2D_21mers -c kmer_candidates -l map2.txt -o AAAA_pvalues.txt -r read_info.txt
	Input: -i the directory of features input
		-k list of kmer candidates
		-l patients information
		-o output filename
		-r read length and read count
	Author: Chelsea Ju
	Note: need to export Jellyfish package to python path export PYTHONPATH=$PYTHONPATH:/home/chelseaju/MetaK/Jellyfish/jellyfish_software/lib/python3.6/site-packages
"""

import sys, re, os, argparse, datetime, random
import jellyfish
import scipy.stats as stats
import numpy
from decimal import Decimal


def echo(msg):
	print ("[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(msg)))

def parse_labels(label_file):
	labels = []
	fh = open(label_file, 'r')
	for line in fh:
		(pid, plabel) = line.rstrip().split()
		labels.append((pid, plabel))
	fh.close()
	return labels	


"""
	The total kmer count is computed from jellyfish dump
"""
def load_read_info(read_info, k):
	norm_factors = {}
	fh = open(read_info, 'r')
	for line in fh:
		(filename, tkmers) = line.rstrip().split()
		norm_factors[filename] = float(tkmers)
	fh.close()
	return norm_factors

def prepare_jellyfish(indir, label_file, read_info, k):
	positive = []
	negative = []

	positive_factor = []
	negative_factor = []

	norm_factors = load_read_info(read_info, k)

	labels = parse_labels(label_file) 

	for (p, l) in labels:
		filename = os.path.join(indir, p + "_count.jf")
		if(l == "-1"):
			negative.append(jellyfish.QueryMerFile(filename))
			negative_factor.append(norm_factors[p])
		else:
			positive.append(jellyfish.QueryMerFile(filename))
			positive_factor.append(norm_factors[p])	

	return (positive, negative, positive_factor, negative_factor)	

def ttest_kmer(positive_qfs, negative_qfs, positive_factor, negative_factor, kmer_candidates, outfile):

	kmer_fh = open(kmer_candidates, 'r')
	outfh = open(outfile, 'w')

	i = 0
	for line in kmer_fh:
		mer = jellyfish.MerDNA(line.rstrip())
		mer.canonicalize()

		positive = []
		negative = []

		for x in xrange(len(positive_qfs)):
			factor = positive_factor[x] 
			p_qfs = positive_qfs[x]
			positive.append(float(p_qfs[mer]) / float(factor))
	
		for j in xrange(len(negative_qfs)):
			factor = negative_factor[j]
			n_qfs = negative_qfs[j]
			negative.append(float(n_qfs[mer]) / float(factor))

		p_mean = numpy.mean(positive)
		n_mean = numpy.mean(negative)

		if(not p_mean == 0 and not n_mean == 0):
			t_stat, p_val = stats.ttest_ind(positive, negative, equal_var=False) ## running t-test	
		outfh.write("%s\t%E\t%E\t%f\t%E\n" %(mer, Decimal(p_mean), Decimal(n_mean), t_stat, Decimal(p_val)))

		if( i % 1 == 1000):
			echo("------ completed %d" %(i))
		i = i + 1
	kmer_fh.close()
	outfh.close()

def main(parser):
	option = parser.parse_args()
	indir = option.indir
	outfile = option.outfile
	kmer_candidates = option.candidates
	label_file = option.label
	read_info = option.readinfo	

	echo("Start Processiong")
	fh = open(kmer_candidates, 'r')
	kmer = fh.readline()
	kmer = kmer.rstrip()
	k = len(kmer)

	echo("Preparing Jellyfish File Handlers")
	(positive_qfs, negative_qfs, positive_factor, negative_factor) = prepare_jellyfish(indir, label_file, read_info, k)
	
	echo("Perform t-test")
	ttest_kmer(positive_qfs, negative_qfs, positive_factor, negative_factor, kmer_candidates, outfile)

	echo("Done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="compute_kmers_pval.py")
	parser.add_argument("-i", "--indir", dest="indir", type=str, help="sequencing directory", required = True)
	parser.add_argument("-c", "--candidates", dest="candidates", type=str, help="list of kmer candidates", required = True)
	parser.add_argument("-l", "--label", dest="label", type=str, help="patient label", required = True)
	parser.add_argument("-o", "--outfile", dest="outfile", type=str, help="output filename", required = True)
	parser.add_argument("-r", "--readinfo", dest="readinfo", type=str, help="read length and read count", required = True)
	main(parser)
