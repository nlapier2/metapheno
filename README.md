# metapheno
Repository for the (W)T2D experiments run in the Metapheno paper.

To replicate the experiments run in metapheno, simply run:

```
unzip t2d_data.zip
unzip wt2d_data.zip
python3 run-metapheno.py {t2d OR wt2d}
```

You can run a more comprehensive experiment with a parameter grid search and results across multiple random seeds using the --grid\_amount and --seed\_search parameters. Do "python3 run-metapheno.py -h" for help. Even without grid search, the script may take a while to run, especially if GPUs are not available, and searching across seeds or using grid search will increase runtime dramatically, possibly to several days.
 
