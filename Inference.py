from time import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score

USE_MODE, USE_MODE_THRESHOLD = True, 0.5  # This means use mode as coefficient and do regular logistic regression.
# logits greater or equal than THRESHOLD are considered to be fake reviews
USE_SAMPLES = False  # This means getting one z for each sampled betas, then use the mode of distribution and apply
APPLY_MODE, APPLY_MEAN, APPLY_MAX, USE_SAMPLES_THRESHOLD = False, False, True, 0.5  # sigmoid on this (APPLY_MODE)
# Another option is getting one logit for each sample and using the mean (APPLY_MEAN)
# Another option is getting one prediction for each sample and then using the mode (APPLY_MAX)

# TODO: Use logits from USE_SAMPLES APPLY_MEAN before taking the mean, and train a logistic regression with that (crazy I know)
# TODO: For JAGS, switch bernouilli with binomial with a lower threshold

BETA_DIR_NAME = "bdata_Pedro_05"
# This FEATURES needs to correspond with the betas (first element is beta1, second is beta2, etc...)
FEATURES = ["Reviewer_deviation", 'avg_posR', 'avg_revL', 'MNR', 'fBERT0', 'fBERT1', 'fBERT2']

data_test = pd.read_csv('prep_data/data_test.csv')
y = data_test["Fake"].values
betas_samples = {"Intercept": pd.read_csv(BETA_DIR_NAME + "/" + "beta0.csv")["x"].values}
for i in range(len(FEATURES)):
    betas_samples[FEATURES[i]] = pd.read_csv(BETA_DIR_NAME + "/" + "beta{}.csv".format(i + 1))["x"].values

if USE_MODE:

    betas_modes = {}
    for key in betas_samples.keys():
        betas_modes[key] = stats.mode(betas_samples[key])[0][0]

    z = betas_modes["Intercept"]
    for key in FEATURES:
        z += betas_modes[key] * data_test[key].values
    logits = 1 / (1 + np.exp(-z))

    logits_copy = np.copy(logits)
    logits_copy[logits_copy >= USE_MODE_THRESHOLD] = 1
    logits_copy[logits_copy < USE_MODE_THRESHOLD] = 0

    print("Accuracy:", accuracy_score(y, logits_copy), ", Recall:", recall_score(y, logits_copy))
    print(pd.crosstab(y, logits_copy, rownames=['True'], colnames=['Predicted'], margins=True))

    while input("You used {} threshold... Would you like to get the results for another one? ([y]/n) ".format(USE_MODE_THRESHOLD)) != "n":
        USE_MODE_THRESHOLD = float(input("Enter threshold: "))
        logits_copy = np.copy(logits)
        logits_copy[logits_copy >= USE_MODE_THRESHOLD] = 1
        logits_copy[logits_copy < USE_MODE_THRESHOLD] = 0
        print("Accuracy:", accuracy_score(y, logits_copy), ", Recall:", recall_score(y, logits_copy))
        print(pd.crosstab(y, logits_copy, rownames=['True'], colnames=['Predicted'], margins=True))


if USE_SAMPLES:

    start = time()

    # betas_samples (as a Numpy Array) is 15000 x (len(FEATURES) + 1)
    # data_test (as a Numpy Array) is 18463 x len(FEATURES)
    b_samples = np.empty((len(betas_samples[FEATURES[0]]), (1 + len(FEATURES))))
    for i, key in enumerate(betas_samples.keys()):
        b_samples[:, i] = betas_samples[key]
    z_samples = b_samples[:, 0] + np.dot(data_test[FEATURES].values, b_samples[:, 1:].T)

    if APPLY_MODE:
        z_modes = stats.mode(z_samples.T)[0]
        logits = 1 / (1 + np.exp(-z_modes))
    if APPLY_MEAN:
        logits_samples = 1 / (1 + np.exp(-z_samples))
        logits = np.mean(logits_samples, axis=1)
    if APPLY_MAX:
        logits_samples = 1 / (1 + np.exp(-z_samples))
        logits_samples_copy = np.copy(logits_samples)
        logits_samples_copy[logits_samples_copy >= USE_SAMPLES_THRESHOLD] = 1
        logits_samples_copy[logits_samples_copy < USE_SAMPLES_THRESHOLD] = 0
        y_pred = stats.mode(logits_samples_copy.T)[0].reshape(-1)
    else:
        logits_copy = np.copy(logits)
        logits_copy[logits_copy >= USE_SAMPLES_THRESHOLD] = 1
        logits_copy[logits_copy < USE_SAMPLES_THRESHOLD] = 0
        y_pred = logits_copy.reshape(-1)

    print(time() - start)
    print("Accuracy:", accuracy_score(y, y_pred), ", Recall:", recall_score(y, y_pred))
    print(pd.crosstab(y, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    while input("You used {} threshold... Would you like to get the results for another one? ([y]/n) ".format(USE_SAMPLES_THRESHOLD)) != "n":
        USE_SAMPLES_THRESHOLD = float(input("Enter threshold: "))
        if APPLY_MAX:
            logits_samples_copy = np.copy(logits_samples)
            logits_samples_copy[logits_samples_copy >= USE_SAMPLES_THRESHOLD] = 1
            logits_samples_copy[logits_samples_copy < USE_SAMPLES_THRESHOLD] = 0
            y_pred = stats.mode(logits_samples_copy.T)[0].reshape(-1)
        else:
            logits_copy = np.copy(logits)
            logits_copy[logits_copy >= USE_SAMPLES_THRESHOLD] = 1
            logits_copy[logits_copy < USE_SAMPLES_THRESHOLD] = 0
            y_pred = logits_copy.reshape(-1)
        print("Accuracy:", accuracy_score(y, y_pred), ", Recall:", recall_score(y, y_pred))
        print(pd.crosstab(y, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
