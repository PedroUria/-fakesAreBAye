# This brief script shows that the BERT features are actually already kinda standardized

import numpy as np
import matplotlib.pyplot as plt

SEQ_LEN = 100
N_LAYERS = 4
N_FEATURES = 3

features_bert_train = np.load("saved_features_BERT_sigmoid/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN))

plt.plot(features_bert_train[:, 0], features_bert_train[:, 1])
plt.plot(features_bert_train[:, 0], features_bert_train[:, 2])
plt.show()

fig, axs = plt.subplots(3)
fig.suptitle('BERT Features Histograms')
axs[0].hist(features_bert_train[:, 0], bins=100, color="r")
axs[1].hist(features_bert_train[:, 1], bins=100, color="b")
axs[2].hist(features_bert_train[:, 2], bins=100, color="g")
plt.savefig("BERTFeaturesHistograms.png")
plt.show()
