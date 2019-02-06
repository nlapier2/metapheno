# metapheno
Repository for the T2D and obesity experiments run in the Metapheno paper.

To replicate the experiments run in metapheno, simply run:

```
unzip t2d_data.zip
unzip obesity_data.zip
python3 run-metapheno.py {t2d OR obesity}
```

Do "python3 run-metapheno.py -h" for help. There are not many options and they are simple to use. The defauls are the ones that were used in the paper. Even without grid search, the script may take a while to run, especially if GPUs are not available, and searching across seeds or using grid search will increase runtime dramatically, possibly to several days. The classify.py script can be used stand-alone to test individual machine learning algorithms, but is mostly used as an imported library for the wrapper script run-metapheno.py. Only run-metapheno has to be executed by the user to replicate the paper's results.

Please note that gcforest/deepforest and neural net results may not replicate exactly due to inherent randomness in the classifiers. This should hopefully be mitigated by running them in multiple settings and evaluating metrics across all runs.

Finally, note that you will need python3, tensorflow, keras, xgboost, gcforest, sklearn, numpy, and scipy installed to use these scripts.


### Feature Extraction - microbial abundance from MetaPhlAn2 
1. Run MetaPhlAn2 on metagenomic read data to generate the species abundance profile
2. Run extract_metaphlan2_features.py to extract the abundance into a sparse matrix

```
python extract_metaphlan2_features.py -i MetaPhlAn2_OUTDIR --train t2d_data/random0/Dataset01/train.txt  --test t2d_data/random0/Dataset01/test.txt --train_out t2d_data/random0/Dataset01/metaphlan_lowest_train.data --test_out t2d_data/random0/Dataset01/metaphlan_lowest_test.data
```

### Feature Extraction - kmer abundance from Jellyfish
1. Run Jellyfish on the read files to generate kmer counts
2. Run Jellyfish to count the total number of reads for each sample; save it to readcount_by_jellyfish.txt (filename_\t_readCount)  
3. Run generate_kmer_candidates.py to generate a list of all possible k-mers
4. Run compute_kmers_pval.py to compute the p-value for each k-mer between disease and healthy samples in the training set
5. Run adjust_kmers_pval.py to adjust p-values from multiple hypothesis testing, and pick the top N k-mers
6. Run extract_kmers_features.py to extract the abundance for selected k-mers in training and testing data

``` 
python generate_kmer_candidates.py -k 12 -o 12mers.txt
python compute_kmers_pval.py -i Jellyfish_OUTDIR -c 12mers.txt -l t2d_data/random0/Dataset01/train.txt  -o t2d_data/random0/Dataset01/candidate_12mers_pval.txt  -r readcount_by_jellyfish.txt
python adjust_kmers_pval.py -i t2d_data/random0/Dataset01/candidate_12mers_pval.txt -o t2d_data/random0/Dataset01/ttest_top_12mers_1000.txt -t 1000
python extract_kmers_features.py -s t2d_data/random0/Dataset01/ttest_top_12mers_1000.txt -i Jellyfish_OUTDIR --train t2d_data/random0/Dataset01/train.txt --train_out t2d_data/random0/Dataset01/ttest_top_12mers_1000_train.data
python extract_kmers_features.py -s t2d_data/random0/Dataset01/ttest_top_12mers_1000.txt -i Jellyfish_OUTDIR --train t2d_data/random0/Dataset01/test.txt --train_out t2d_data/random0/Dataset01/ttest_top_12mers_1000_test.data
```

