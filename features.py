import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_eda import data_restaurants

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# %% --------------------------------------- Behaviour Features --------------------------------------------------------
test12 = data_restaurants.groupby(['ReviewerID', 'Date']).count()
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
data_restaurants['avg_revL'] = data_restaurants['ReviewerID'].apply(lambda x: word_avg[x])
data_restaurants['posR'] = data_restaurants['Rating'].apply(lambda x: 1 if x >= 4 else 0)
posR = data_restaurants.groupby('ReviewerID').mean()['posR']
data_restaurants['avg_posR'] = data_restaurants['ReviewerID'].apply(lambda x: posR[x])
ProdPivot = data_restaurants.pivot_table(index='Date', columns='ProductID', values='Rating')
avProdR = ProdPivot.mean()
data_restaurants['exp_Product_Rating'] = data_restaurants['ProductID'].apply(lambda x: avProdR[x])
data_restaurants['abs_prod_rating_dev'] = np.abs(data_restaurants['Rating'] - data_restaurants['exp_Product_Rating'])
exp_rev_frame = data_restaurants.groupby('ReviewerID').mean()['abs_prod_rating_dev']
data_restaurants['Reviewer_deviation'] = data_restaurants['ReviewerID'].apply(lambda x: exp_rev_frame[x])

x = data_restaurants.drop(["Fake"], axis=1)
y = data_restaurants["Fake"].replace("N", 0).replace("Y", 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.3, stratify=y)

features_behaviour_train = x_train[['Reviewer_deviation', 'avg_posR', 'avg_revL', 'MNR']]
features_behaviour_test = x_test[['Reviewer_deviation', 'avg_posR', 'avg_revL', 'MNR']]

# %% ----------------------------------------- BERT Features -----------------------------------------------------------
SEQ_LEN = 100
N_LAYERS = 4
N_FEATURES = 3

features_bert_train = np.load("saved_features_BERT_sigmoid/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN))
features_bert_test = np.load("saved_features_BERT_sigmoid/features_test_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN))
print("BERT cls weights:")
with open("saved_models_BERT_sigmoid/BERT_last_weights{}layers_{}features_{}len.txt".format(N_LAYERS, N_FEATURES,
                                                                                            SEQ_LEN), "r") as s:
    print(s.read())

# This is a check to make sure the split made here for the Behaviour features matches the one made for BERT on BERT_features.py
# import torch
# import torch.nn as nn
# from transformers.modeling_bert import BertForSequenceClassification
# from transformers.tokenization_bert import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
# class BERTForFeatures(nn.Module):
#     def __init__(self, n_bert_layers=N_LAYERS, n_features=N_FEATURES, extract_features=False):
#         super(BERTForFeatures, self).__init__()
#         self.extract_features = extract_features
#         self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
#         self.bert.bert.encoder.layer = self.bert.bert.encoder.layer[:n_bert_layers]
#         self.bert.classifier = nn.Linear(768, n_features)
#         self.cls = nn.Linear(n_features, 2)
#
#     def forward(self, p, attn_mask):
#         features, *_ = self.bert(p, attention_mask=attn_mask)
#         if self.extract_features:
#             return features
#         return self.cls(features)
#
# model = BERTForFeatures(extract_features=True)
# model.load_state_dict(torch.load("saved_models_BERT/BERT_{}layers_{}features_{}len.pt".format(N_LAYERS, N_FEATURES, SEQ_LEN)))
# model.eval()
#
# x, x_mask = [], []
# for review in x_train["Review"].values[10:11]:
#     token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review)[:SEQ_LEN-2])
#     token_ids = [101] + token_ids + [102]
#     n_ids = len(token_ids)
#     attention_mask = [1] * n_ids
#     if n_ids < SEQ_LEN:
#         token_ids += [0] * (SEQ_LEN - n_ids)
#         attention_mask += [0] * (SEQ_LEN - n_ids)
#     x.append(token_ids)
#     x_mask.append(attention_mask)
# x, x_mask = torch.LongTensor(x), torch.FloatTensor(x_mask)
#
# with torch.no_grad():
#     feats = model(x, x_mask).numpy()
# print(feats, features_bert_train[10])
# # They are the same, good!

# %% ----------------------------------------- Combine Features --------------------------------------------------------
features_and_target_train = np.hstack((features_behaviour_train, features_bert_train, y_train.values.reshape(-1, 1)))
features_and_target_test = np.hstack((features_behaviour_test, features_bert_test, y_test.values.reshape(-1, 1)))
if "prep_data" not in os.listdir():
    os.mkdir("prep_data")
data_train = pd.DataFrame(
    features_and_target_train,
    columns=list(features_behaviour_train.columns) + ["fBERT{}".format(i) for i in range(N_FEATURES)] + ["Fake"]
)
data_test = pd.DataFrame(
    features_and_target_test,
    columns=list(features_behaviour_test.columns) + ["fBERT{}".format(i) for i in range(N_FEATURES)] + ["Fake"]
)
data_train.to_csv("prep_data/data_train.csv")
data_test.to_csv("prep_data/data_test.csv")
