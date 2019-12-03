import numpy as np
import pandas as pd

DATA_DIR = "YelpChi/"

data_restaurants = pd.read_csv(DATA_DIR + "output_meta_yelpResData_NRYRcleaned.txt", sep=" ", header=None,
                               names=["Date", "ReviewID", "ReviewerID", "ProductID", "Fake", "?", "??", "???", "Rating"])
# print(metadata_restaurants.isna().sum())
data_restaurants.drop(["?", "??", "???"], axis=1, inplace=True)

with open(DATA_DIR + "output_review_yelpResData_NRYRcleaned.txt", "r") as s:
    data_restaurants["Review"] = s.read().split("\n")[:-1]
# The first review can be found here: https://www.yelp.com/menu/alinea-chicago/item/lamb (ctf+f: Willy Wonka)
countR = data_restaurants.groupby('ReviewerID').count()
countred = countR[countR['Date'] != 1]
data_restaurants['mase'] = data_restaurants['ReviewerID'].apply(lambda x: 1 if x in countred.index else 0)
data_restaurants_dropped = data_restaurants[data_restaurants['mase'] == 0]
data_restaurants = data_restaurants[data_restaurants['mase'] == 1]

labels_distrib = np.unique(data_restaurants["Fake"].values, return_counts=True)
print("-" * 82)
print("The distribution of real (N) vs fake (Y) reviews is -------->",
      labels_distrib[0][0], ":", labels_distrib[1][0], "|",
      labels_distrib[0][1], ":", labels_distrib[1][1])
print("-" * 82)
