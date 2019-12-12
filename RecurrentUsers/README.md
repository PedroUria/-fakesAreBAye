# -fakesAreBAye (Recurrent Reviewers)

This folder contains all the code we used for our project with regards to reviewers that had made more than one review.

To reproduce our results, first you will have to ask [Dr. Rayana](http://odds.cs.stonybrook.edu/yelpzip-dataset/) to send you the link to the datasets. Then download `YelpChi.zip` and unzip it in this directory.

Afterwards, the order to run the code is:

1. [`data_eda.py`](data_eda.py): data EDA and cleaning.
2. [`BERT_features_sigmoid.py`](BERT_features_sigmoid.py): trains BERT and saves the model used to extract the text features.
3. [`features.py`](features.py): extracts and saves behavioural and BERT features.
4. [`Run_ME.R`](Run_ME.R): runs MCMC and saved the sampled parameters. The relevant experiment hyperparameters can be found [here](experiments.txt).
5. [`Inference.py`](Inference.py): runs the inference on our test set in order to get the final metrics. 
