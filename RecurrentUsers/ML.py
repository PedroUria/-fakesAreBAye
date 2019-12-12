import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
USE_SAMPLES_THRESHOLD = .5

# FEATURES = ["Reviewer_deviation", "avg_revL", "fBERT2", "B2_len_int"]
FEATURES = ["Reviewer_deviation", 'avg_posR', 'avg_revL', 'MNR', "max_cos", 'fBERT0', 'fBERT1', 'fBERT2']

data_train = pd.read_csv("prep_data/data_train.csv")
data_test = pd.read_csv('prep_data/data_test.csv')

for column_name in FEATURES:
    if column_name not in data_train.columns:
        column_names_interaction = input("There seems to be an interaction term called {}. Please enter the column names that are interacting: ")
        data_train[column_name] = data_train[column_names_interaction.split()[0]] * data_train[column_names_interaction.split()[1]]
for column_name in FEATURES:
    if column_name not in data_test.columns:
        column_names_interaction = input("There seems to be an interaction term called {}. Please enter the column names that are interacting: ")
        data_test[column_name] = data_test[column_names_interaction.split()[0]] * data_test[column_names_interaction.split()[1]]

x_train, x_test = data_train[FEATURES].values, data_test[FEATURES].values
y_train, y_test = data_train["Fake"].values, data_test["Fake"].values

std_scaler = StandardScaler()
x_train_std = std_scaler.fit_transform(x_train)
x_test_std = std_scaler.transform(x_test)

lr = LogisticRegression(random_state=0)
lr.fit(x_train_std, y_train)
y_test_pred = lr.predict_proba(x_test_std)
ycop = y_test_pred[:, 1].copy()
ycop[ycop >= USE_SAMPLES_THRESHOLD] = 1
ycop[ycop < USE_SAMPLES_THRESHOLD] = 0
auc = roc_auc_score(y_test, ycop)
print('AUC: %.3f' % auc)

print("Accuracy:", accuracy_score(y_test, ycop), ", Recall:", recall_score(y_test, ycop))
print(pd.crosstab(y_test, ycop, rownames=['True'], colnames=['Predicted'], margins=True))

while input("You used {} threshold... Would you like to get the results for another one? ([y]/n) ".format(
        USE_SAMPLES_THRESHOLD)) != "n":
    USE_SAMPLES_THRESHOLD = float(input("Enter threshold: "))
    ycop = y_test_pred[:, 1].copy()

    ycop[ycop >= USE_SAMPLES_THRESHOLD] = 1
    ycop[ycop < USE_SAMPLES_THRESHOLD] = 0
    auc = roc_auc_score(y_test, ycop)
    print('AUC: %.3f' % auc)
    print("Accuracy:", accuracy_score(y_test, ycop), ", Recall:", recall_score(y_test, ycop))
    print(pd.crosstab(y_test, ycop, rownames=['True'], colnames=['Predicted'], margins=True))
# auc = roc_auc_score(y_test, logits)
# print('AUC: %.3f' % auc)
USE_SAMPLES_THRESHOLD = .5

svm = SVC(random_state=0)
svm.fit(x_train_std, y_train)
y_test_pred = svm.predict_proba(x_test_std)
ycop = y_test_pred.copy()
ycop[ycop >= USE_SAMPLES_THRESHOLD] = 1
ycop[ycop < USE_SAMPLES_THRESHOLD] = 0

print("Accuracy:", accuracy_score(y_test, ycop), ", Recall:", recall_score(y_test, ycop))
print(pd.crosstab(y_test, ycop, rownames=['True'], colnames=['Predicted'], margins=True))

while input("You used {} threshold... Would you like to get the results for another one? ([y]/n) ".format(
        USE_SAMPLES_THRESHOLD)) != "n":
    USE_SAMPLES_THRESHOLD = float(input("Enter threshold: "))
    ycop = y_test_pred.copy()
    ycop[ycop >= USE_SAMPLES_THRESHOLD] = 1
    ycop[ycop < USE_SAMPLES_THRESHOLD] = 0
    #y_pred = stats.mode(logits_samples_copy.T)[0].reshape(-1)
    print("Accuracy:", accuracy_score(y_test, ycop), ", Recall:", recall_score(y_test, ycop))
    print(pd.crosstab(y_test, ycop, rownames=['True'], colnames=['Predicted'], margins=True))



print("Accuracy:", accuracy_score(y_test, y_test_pred), ", Recall:", recall_score(y_test, y_test_pred))
print(pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
#
# clfs = {'lr': LogisticRegression(random_state=0),
#         'rf': RandomForestClassifier(random_state=0),
#         'svc': SVC(random_state=0)}
#
# # Dictionary of the Pipeline
#
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
#
# pipe_clfs = {}
# for name, clf in clfs.items():
#     pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()), ('clf', clf)])
#
# # Dictionary of parameter grids
#
# param_grids = {}
#
# # LogReg
# C_range = [10 ** i for i in range(-4, 5)]
# param_grid = [{'clf__multi_class': ['ovr'],
#                'clf__solver': ['newton-cg', 'liblinear'],
#                'clf__C': C_range},
#               {'clf__multi_class': ['multinomial'],
#                'clf__solver': ['newton-cg'],
#                'clf__C': C_range}]
# param_grids['lr'] = param_grid
#
# # Random Forest
# param_grid = [{'clf__n_estimators': [2, 10, 30],
#                'clf__min_samples_split': [2, 10, 30],
#                'clf__min_samples_leaf': [1, 10, 30]}]
# param_grids['rf'] = param_grid
#
# # SVM
# param_grid = [{'clf__C': [0.01, 0.1, 1, 10, 100],
#                'clf__gamma': [0.01, 0.1, 1, 10, 100],
#                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
# param_grids['svc'] = param_grid
#
# # Hyperparameter Tuning
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
#
# # The list of [best_score_, best_params_, best_estimator_]
# best_score_param_estimators = []
# # For each classifier
# for name in pipe_clfs.keys():
#     # GridSearchCV
#     gs = GridSearchCV(estimator=pipe_clfs[name],
#                       param_grid=param_grids[name],
#                       scoring='recall',
#                       n_jobs=-1,
#                       cv=StratifiedKFold(n_splits=10,
#                                          shuffle=True,
#                                          random_state=0))
#     # Fit the pipeline
#     gs = gs.fit(x_train, y_train)
#     # Update best_score_param_estimators
#     best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
#
# # Model Selection
# # Sort best_score_param_estimators in descending order of the best_score_
# best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)
# # For each [best_score_, best_params_, best_estimator_]
# for best_score_param_estimator in best_score_param_estimators:
#     # Print out [best_score_, best_params_, best_estimator_], where best_estimator_ is a pipeline
#     # Since we only print out the type of classifier of the pipeline
#     print([best_score_param_estimator[0], best_score_param_estimator[1], type(best_score_param_estimator[2].named_steps['clf'])], end='\n\n')
#
# y_test_pred = best_score_param_estimators[0][2].predict(x_test)
# print("Accuracy:", accuracy_score(y_test, y_test_pred), ", Recall:", recall_score(y_test, y_test_pred))
