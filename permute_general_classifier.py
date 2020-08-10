# Import needed libraries
import sys, os
import logging
import pandas as pd
import numpy as np
from collections import namedtuple
from tools.run_classifier import run_classifier
from tools.RunInParallel import RunInParallel

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import different cross validation metrics
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

# Setup logging for output
logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
	datefmt='%m-%d-%y %H:%M')

#########################################
############ PROCESS ARGS ###############
#########################################
USE_CLI = True
INPUT_DEFAULT = 'data.csv'
OUTPUT_DEFAULT = '.'
CV_DEFAULT = 'LeaveOneOut()'

import argparse

parser = argparse.ArgumentParser(description='Runs 3 classification ML pipelines')
parser.add_argument('--input', help='CSV input for algorithm', required=True, default=INPUT_DEFAULT)
parser.add_argument('--output', help='output directory', default=OUTPUT_DEFAULT)
parser.add_argument('--cv', help='Cross validation metric to use', default=CV_DEFAULT)

if USE_CLI:
	cli_args = parser.parse_args()
else:
	cli_args = namedtuple('cli_args', ['input', 'output', 'cv'])
	cli_args = cli_args(INPUT_DEFAULT, OUTPUT_DEFAULT, CV_DEFAULT)

#########################################
############ Clean the data #############
#########################################

# Read in the data
logging.info("Reading input from: {} ...".format(cli_args.input))
df = pd.read_csv(cli_args.input)

# Binarize response variable to 0/1 if it's not already
current_responses = list(set(df['outcome'].values))
response_map = dict(zip(current_responses, [0,1]))
df['outcome'] = [response_map[val] for val in df['outcome'].values]


# Set up the hyperparameters
model1 = LogisticRegression() # change lambda here if you want
#model2 = SVC(probability=True)
#model3 = RandomForestClassifier(n_estimators=250, n_jobs=-1)
all_models = [model1]

# Get the cross validation metric
cv = eval(cli_args.cv)

# Setup our X, y, labels
y = df['outcome'].values.flatten()
df = df.drop(['outcome'], axis=1)
labels = df.columns
X = df.values
logging.info("Input data shape: {}".format(X.shape))
logging.info("Outcome data shape: {}".format(y.shape))
logging.info("N positive class: {}".format(sum(y)))
logging.info("N negative class: {}".format(len(y) - sum(y)))

# Iterate over the models we want to use
for model in all_models:
	### Run each model 1000 times in parallel
	# First setup the params into a large list
	params_per_permute = []

	for i in range(1000):
		np.random.shuffle(X)
		params_per_permute.append([model, X, y, cv, labels, cli_args.output])

	# Run them all in parallel
	# Pass in the function you want to run in parallel, and the params for each time it's run
	all_stats = RunInParallel(run_classifier, params_per_permute)

# Save the output to a file
stats_df = pd.DataFrame(all_stats)
print(stats_df)
stats_df.to_csv(os.path.join(cli_args.output, "results.csv"), index=False)
