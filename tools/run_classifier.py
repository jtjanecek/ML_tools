import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
import logging
import random

def run_classifier(model, X, y, cv, labels, output_dir) -> dict:
	# Setup random state
	RANDOM_STATE = 100
	random.seed(RANDOM_STATE)

	logging.info("Running {} ...".format(model))
	
	# Make sure the cross validation shuffles the splits each time
	cv.shuffle = True

	y_tests = []
	predicted = []
	predicted_probas = []

	# Run the CV 10 times to be robust
	num_cv = 10
	for i in range(num_cv):
		logging.debug("Running multiple CV iteration {} / {} ...".format(i+1, num_cv))
		for train, test in cv.split(X, y):
			# Fit the model for this split
			model.random_state = RANDOM_STATE
			model.fit(X[train], y[train])
			# Predict probability and actual prediction
			probas_ = model.predict_proba(X[test])
			predicted_ = model.predict(X[test])	
			
			# Add the predicted/true for this split to the master split
			y_tests.append(y[test])
			predicted_probas.append(probas_[:,1])
			predicted.append(predicted_)
			RANDOM_STATE += 1

	##### If our CV is LOO, then we can't generate confidence intervals
	if 'LeaveOneOut' in str(cv):
		# Flatten the lists so they are 1 dimensional
		y_tests = np.array(y_tests).flatten()
		predicted = np.array(predicted).flatten()
		predicted_probas = np.array(predicted_probas).flatten()

		# Generate confusion matrix
		tn, fp, fn, tp = confusion_matrix(y_tests, predicted).ravel()

		# Calculate AUC
		fpr, tpr, thresholds = roc_curve(y_tests, predicted_probas)
		ro_auc = auc(fpr, tpr)

		# Add values to the stats to be saved as a CSV
		stats = {'model': str(model).split("(")[0],
				 'tnr': tn / (tn+fn),
				 'fpr': fp / (fp+tn),
				 'fnr': fn / (fn+tp),
				 'tpr': tp / (tp+fp),
				 'ro_auc': ro_auc
				}
		pass
	
	##### Otherwise our CV is KFold or similar, so we can calculate confidence intervals by
	##### the splits
	##### TODO
	else: 
		pass

	logging.info("Done: {}".format(stats))
	return stats
