"""
	Function: compute the adjusted p-value for each kmer
	To Run: python adjust_kmers_pval.py -i T2D_12mers_sig/candidates_12mers.txt -o T2D_12mers/ttest_selected_features_001.txt -t 0.05
	Input: -i the file of pvalues 
		-o selected features 
		-t threshold ( <= 1 refers to adjusted p-value cutoff, > 1 refers to top n features)
	Author: Chelsea Ju
"""

import sys, re, os, argparse, datetime, random
import scipy.stats as stats
import statsmodels.sandbox.stats.multicomp
import numpy as np



def echo(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(msg)))

def load_pvalues(infile, cutoff):
	kmers = []
	pvalues = []
	fh = open(infile, 'r')
	for line in fh:
		mini_data = line.rstrip().split()
		kmers.append(mini_data[0])
		pvalues.append(float(mini_data[4]))
	
	fh.close()

	if(cutoff <= 1):
		correction = statsmodels.sandbox.stats.multicomp.multipletests(pvalues, alpha=cutoff, method='fdr_bh', returnsorted=False)
		return [kmers[x] for x in np.where(correction[0])[0].tolist()]
	else:
		return list(np.array(kmers)[np.argsort(pvalues)[:int(cutoff)]])
		

def export_data(sigmers, outfile):
	fh = open(outfile, 'w')
	for mer in sorted(sigmers):
		fh.write("%s\n" %(mer))
	fh.close()

def main(parser):
	option = parser.parse_args()
	infile = option.infile
	outfile = option.outfile
	threshold = float(option.threshold)

	echo("Start Processiong")
	echo("Adjusting p-values")
	selected_sigmer = load_pvalues(infile, threshold)

	echo("Exporting significant features")
	export_data(selected_sigmer, outfile)
	echo("Done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="adjust_kmers_pval.py")
	parser.add_argument("-i", "--infile", dest="infile", type=str, help="pvalue file", required = True)
	parser.add_argument("-o", "--outfile", dest="outfile", type=str, help="output filename", required = True)	
	parser.add_argument("-t", "--threshold", dest="threshold", type=float, help="threshold cutoff (<=1 for adj-pval; >1 for top n", required = True)
	main(parser)
