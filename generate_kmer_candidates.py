"""
	Function: generate a list of normalized kmer with a given size
	To Run: python generate_kmer_candidates_v1.py -k 12 -o candidate_12mers.txt
	Input: -k size of kmer
	 	-o output filename
	Author: Chelsea Ju
"""

import sys, re, os, argparse, datetime, random

INT2DNA = {0:'A', 1:'C', 2:'G', 3:'T'}

def echo(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(msg)))

def reverse_complimentary(sequence):
        sequence = sequence.upper()
        rev_seq = sequence[::-1]

        rev_seq = rev_seq.replace('A', '%')
        rev_seq = rev_seq.replace('T', 'A')
        rev_seq = rev_seq.replace('%', 'T')
        rev_seq = rev_seq.replace('C', '%')
        rev_seq = rev_seq.replace('G', 'C')
        rev_seq = rev_seq.replace('%', 'G')

        return rev_seq

def generate_kmers(ksize, outfile):

	combination = 4**ksize
	fh = open(outfile, 'w')

	for i in range(combination):
		if( i % 1000000 == 0):
			echo("....Completed %d out of %d (%f) " %(i, combination, float(i) / float(combination)))		
		x = [0]*ksize
		index = ksize-1
		result = i
		while(result > 0):
			x[index] = result % 4
			result = result // 4
			index = index - 1

		kmer = "".join([ INT2DNA[xi] for xi in x])
		rev_kmer = reverse_complimentary(kmer)
		min_kmer = min(kmer, rev_kmer)

		if(min_kmer == kmer):
			fh.write("%s\n" %(kmer))	
	fh.close()


def main(parser):
	option = parser.parse_args()
	outfile = option.outfile
	ksize = int(option.ksize)
	

	echo("Generating Kmers")
	generate_kmers(ksize, "tmpfile.txt")

	echo("Sorting Kmers")
	os.system("sort %s > %s" %("tmpfile.txt", outfile))

	echo("Remove tmp file")
	os.system("rm %s" %("tmpfile.txt"))

	echo("Done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="generate_kmer_candidates_v1.py")
	parser.add_argument("-k", "--ksize", dest="ksize", type=str, help="ksize", required = True)
	parser.add_argument("-o", "--outfile", dest="outfile", type=str, help="output filename", required = True)
	main(parser)
