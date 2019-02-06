# metapheno
Repository for the T2D and obesity experiments run in the Metapheno paper.

To replicate the experiments run in metapheno, simply run:

```
unzip t2d_data.zip
unzip obesity_data.zip
python3 run-metapheno.py {t2d OR obesity}
```

Do "python3 run-metapheno.py -h" for help. There are not many options and they are simple to use. The defauls are the ones that were used in the paper. Even without grid search, the script may take a while to run, especially if GPUs are not available, and searching across seeds or using grid search will increase runtime dramatically, possibly to several days.

Please note that gcforest/deepforest and neural net results may not replicate exactly due to inherent randomness in the classifiers. This should hopefully be mitigated by running them in multiple settings and evaluating metrics across all runs.

Finally, note that you will need python3, tensorflow, keras, xgboost, gcforest, sklearn, numpy, and scipy installed to use these scripts.

.
