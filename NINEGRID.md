## How to run NineGrid experiments.

# Data Download

https://www.kaggle.com/code/beta3logic/soduku-vs-llm-task-0?scriptVersionId=302485603&select=gridcorpus.csv

and convert to parquet (I did it using pandas)

# Run evaluation script

Go to eval_pkg/run.sh to edit the experiment parameters (change data path, number of samples, difficulty etc.)

Run ./eval_pkg/run.sh

# Requirements

Python 3.10
Create conda environment using requirements_eva.txt