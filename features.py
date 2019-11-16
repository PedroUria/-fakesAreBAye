import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from data_eda import data_restaurants

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
features_behaviour_train, features_behaviour_test = np.zeros((len(x_train), 4)), np.zeros((len(x_test), 4))

# %% ----------------------------------------- Combine Features --------------------------------------------------------
features_train, features_test = np.hstack((features_behaviour_train, features_bert_train)), np.hstack((features_behaviour_test, features_bert_test))
if "saved_features" not in os.listdir():
    os.mkdir("saved_features")
# TODO: We are gonna have to replace this with something that R can read, probably using pandas ad .csv should be fine
np.save("saved_features/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_train)
np.save("saved_features/features_test_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_test)
