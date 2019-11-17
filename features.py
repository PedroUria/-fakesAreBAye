import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from data_eda import data_restaurants
#### actually extracting the behavioral features

test12 = data_restaurants.groupby(['ReviewerID','Date']).count()
revz = [i[0] for i in test12.index]
revz = list(set(revz))
maxList = [test12.loc[i]['ReviewID'].max() for i in revz]
max_dict = {}
for Id in range(len(revz)):
    max_dict[revz[Id]] = maxList[Id]
new_thing = data_restaurants['ReviewerID'].apply(lambda x: max_dict[x])
data_restaurants['MNR'] = new_thing
data_restaurants['WC'] = data_restaurants['Review'].apply(lambda x: len(x.split(' '))) 
word_avg = data_restaurants.groupby('ReviewerID').mean()['WC']
data_restaurants['avg_revL'] = data_restaurants['ReviewerID'].apply(lambda x: word_avg[x] )
data_restaurants['posR'] =data_restaurants['Rating'].apply(lambda x: 1 if x >=4 else 0)
posR = data_restaurants.groupby('ReviewerID').mean()['posR']
data_restaurants['avg_posR'] = data_restaurants['ReviewerID'].apply(lambda x: posR[x] )
ProdPivot  =data_restaurants.pivot_table(index = 'Date',columns='ProductID',values='Rating')
avProdR = ProdPivot.mean()
data_restaurants['exp_Product_Rating'] = data_restaurants['ProductID'].apply(lambda x: avProdR[x])
data_restaurants['abs_prod_rating_dev'] = np.abs(data_restaurants['Rating'] - data_restaurants['exp_Product_Rating'])
exp_rev_frame = data_restaurants.groupby('ReviewerID').mean()['abs_prod_rating_dev']
data_restaurants['Reviewer_deviation'] = data_restaurants['ReviewerID'].apply(lambda x: exp_rev_frame[x])

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# %% ----------------------------------------- BERT Features -----------------------------------------------------------
SEQ_LEN = 100
N_LAYERS = 4
N_FEATURES = 3

features_bert_train = np.load("saved_features_BERT/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN))
features_bert_test = np.load("saved_features_BERT/features_test_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN))

# %% --------------------------------------- Behaviour Features --------------------------------------------------------
x = data_restaurants.drop(["Fake", "Review"], axis=1)
y = data_restaurants["Fake"].replace("N", 0).replace("Y", 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.3, stratify=y)

# TODO: Include check to see if BERT features and this new split match, juuust in case

# TODO: Include the code to get the behaviour features, for now replaced with dummy NumPy array of zeros

features_behaviour_train, features_behaviour_test = x_train[['Reviewer_deviation','avg_posR','avg_revL','MNR']], x_test[['Reviewer_deviation','avg_posR','avg_revL','MNR']]

# %% ----------------------------------------- Combine Features --------------------------------------------------------
features_train, features_test = np.hstack((features_behaviour_train, features_bert_train)), np.hstack((features_behaviour_test, features_bert_test))
if "saved_features" not in os.listdir():
    os.mkdir("saved_features")
# TODO: We are gonna have to replace this with something that R can read, probably using pandas ad .csv should be fine
np.save("saved_features/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_train)
np.save("saved_features/features_test_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_test)
