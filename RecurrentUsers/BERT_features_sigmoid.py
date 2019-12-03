# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers import AdamW, WarmupLinearSchedule
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
TRAIN, LOAD_MODEL, SAVE_MODEL = False, True, False
FINAL_TEST = True
EXTRACT_FEATURES = True

# %% --------------------------------------- Hyper-Parameters ----------------------------------------------------------
SEQ_LEN = 100
N_LAYERS = 4
N_FEATURES = 3

EPOCHS = 10
LR = 1e-5
EPS = 1e-8
BATCH_SIZE = 32
WARM_UP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1

# %% ----------------------------------------- Helper Functions --------------------------------------------------------
def get_features(p, p_mask, net):
    features = np.zeros((len(p), net.cls.in_features))
    net.extract_features = True
    with torch.no_grad():
        with tqdm(total=len(p) // BATCH_SIZE + 1) as pbar:
            for batch in range(len(p) // BATCH_SIZE + 1):
                inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
                if not inds:
                    break
                features[inds] = net(p[inds].to(device), p_mask[inds].to(device)).cpu().numpy()
                pbar.update(1)
    return features

# %% ----------------------------------------- Model Class -------------------------------------------------------------
class BERTForFeatures(nn.Module):
    def __init__(self, n_bert_layers=N_LAYERS, n_features=N_FEATURES, extract_features=False):
        super(BERTForFeatures, self).__init__()
        self.extract_features = extract_features
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.bert.bert.encoder.layer = self.bert.bert.encoder.layer[:n_bert_layers]
        self.bert.classifier = nn.Linear(768, n_features)
        self.cls = nn.Linear(n_features, 1)

    def forward(self, p, attn_mask):
        features, *_ = self.bert(p, attention_mask=attn_mask)
        if self.extract_features:
            return features
        return self.cls(features)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if "prep_data" not in os.listdir():
    os.mkdir("prep_data")
if "x_{}tok.npy".format(SEQ_LEN) not in os.listdir(os.getcwd() + "/prep_data"):
    from data_eda import data_restaurants, data_restaurants_dropped
    x, x_mask = [], []
    print("Tokenizing the reviews...")
    with tqdm(total=len(data_restaurants)) as pbar:
        for review in data_restaurants["Review"].values:
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review)[:SEQ_LEN-2])
            token_ids = [101] + token_ids + [102]
            n_ids = len(token_ids)
            attention_mask = [1] * n_ids
            if n_ids < SEQ_LEN:
                token_ids += [0] * (SEQ_LEN - n_ids)
                attention_mask += [0] * (SEQ_LEN - n_ids)
            x.append(token_ids)
            x_mask.append(attention_mask)
            pbar.update(1)
    x = np.array(x)
    y = LabelEncoder().fit_transform(data_restaurants["Fake"].values)
    np.save("prep_data/x_{}tok.npy".format(SEQ_LEN), x); np.save("prep_data/x_mask_{}tok.npy".format(SEQ_LEN), x_mask)
    np.save("prep_data/y.npy", y)

    x, x_mask = [], []
    with tqdm(total=len(data_restaurants_dropped)) as pbar:
        for review in data_restaurants_dropped["Review"].values:
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review)[:SEQ_LEN-2])
            token_ids = [101] + token_ids + [102]
            n_ids = len(token_ids)
            attention_mask = [1] * n_ids
            if n_ids < SEQ_LEN:
                token_ids += [0] * (SEQ_LEN - n_ids)
                attention_mask += [0] * (SEQ_LEN - n_ids)
            x.append(token_ids)
            x_mask.append(attention_mask)
            pbar.update(1)
    x_dropped, x_mask_dropped = np.array(x), x_mask
    y_dropped = LabelEncoder().fit_transform(data_restaurants_dropped["Fake"].values)
    np.save("prep_data/x_dropped_{}tok.npy".format(SEQ_LEN), x_dropped); np.save("prep_data/x_mask_dropped_{}tok.npy".format(SEQ_LEN), x_mask_dropped)
    np.save("prep_data/y_dropped.npy", y_dropped)
else:
    os.system("python3 data_eda.py")
    x, x_mask = np.load("prep_data/x_{}tok.npy".format(SEQ_LEN)), np.load("prep_data/x_mask_{}tok.npy".format(SEQ_LEN))
    y = np.load("prep_data/y.npy")
    x_dropped, x_mask_dropped = np.load("prep_data/x_dropped_{}tok.npy".format(SEQ_LEN)), np.load("prep_data/x_mask_dropped_{}tok.npy".format(SEQ_LEN))
    y_dropped = np.load("prep_data/y_dropped.npy")

x_train, x_test, mask_train, mask_test, y_train, y_test = train_test_split(
    x, x_mask, y, random_state=SEED, test_size=0.3, stratify=y)
x_train_, mask_train_ = torch.LongTensor(x_train), torch.FloatTensor(mask_train)
x_train, mask_train, y_train = np.vstack((x_train, x_dropped)), np.vstack((mask_train, x_mask_dropped)), np.hstack((y_train, y_dropped))
x_train, mask_train, y_train = torch.LongTensor(x_train), torch.FloatTensor(mask_train), torch.FloatTensor(y_train)[:, None]
x_test, mask_test, y_test = torch.LongTensor(x_test), torch.FloatTensor(mask_test), torch.FloatTensor(y_test)[:, None]

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = BERTForFeatures().to(device)
if "saved_models_BERT_sigmoid" not in os.listdir():
    os.mkdir("saved_models_BERT_sigmoid")
if LOAD_MODEL:
    try:
        model.load_state_dict(torch.load("saved_models_BERT_sigmoid/BERT_{}layers_{}features_{}len.pt".format(N_LAYERS, N_FEATURES, SEQ_LEN)))
        print("A previous model was loaded successfully!")
    except:
        print("Couldn't load model... Starting from scratch!" if TRAIN else "No model has been found... testing is kinda meaningless!")
if TRAIN:
    optimizer = AdamW(model.parameters(), lr=LR, eps=EPS)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARM_UP_STEPS,
    #                                  t_total=len(x_train) // GRADIENT_ACCUMULATION_STEPS * EPOCHS)
    # criterion = nn.BCEWithLogitsLoss(torch.FloatTensor(CLASS_WEIGHTS).to(device))
    # https://discuss.pytorch.org/t/dealing-with-imbalanced-datasets-in-pytorch/22596/22
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([1.5 * 53400 / 8141]).to(device))
    # criterion.weight = torch.zeros((BATCH_SIZE,))

# %% -------------------------------------- Training Loop ----------------------------------------------------------
if TRAIN:
    recall_test_best = 0
    inds_list = list(range(len(x_train)))
    print("Starting training loop...")
    for epoch in range(EPOCHS):

        random.shuffle(inds_list)
        loss_train, steps_train, labels_pred, labels_real = 0, 0, [], []
        model.train()
        total = len(x_train) // BATCH_SIZE + 1
        with tqdm(total=total, desc="Epoch {}".format(epoch)) as pbar:
            for inds in [inds_list[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE] for batch in
                         range(len(inds_list) // BATCH_SIZE + 1)]:
                if not inds:
                    break
                optimizer.zero_grad()
                logits = model(x_train[inds].to(device), mask_train[inds].to(device))
                # criterion.weight[(y_train[inds] == 0).reshape(-1)] = CLASS_WEIGHTS[0]
                # criterion.weight[(y_train[inds] == 1).reshape(-1)] = CLASS_WEIGHTS[1]
                loss = criterion(logits, y_train[inds].to(device))
                loss.backward()
                optimizer.step()
                # scheduler.step()
                loss_train += loss.cpu().item()
                steps_train += 1
                pbar.update(1)
                labels = torch.sigmoid(logits.detach().cpu()).numpy()
                labels[labels > 0.5], labels[labels <= 0.5] = 1, 0
                labels_pred += list(labels.reshape(-1))
                labels_real += list(y_train[inds].numpy().reshape(-1))
                pbar.set_postfix_str("Training Loss: {:.5f}, Precision: {:.2f}, Recall: {:.2f}".format(
                    loss_train / steps_train, precision_score(labels_real, labels_pred), recall_score(labels_real, labels_pred))
                )
        acc_train = accuracy_score(labels_real, labels_pred)
        prec_train = precision_score(labels_real, labels_pred)
        recall_train = recall_score(labels_real, labels_pred)
        cf_train = confusion_matrix(labels_real, labels_pred)

        loss_test, steps_test, labels_pred, labels_real = 0, 0, [], []
        model.eval()
        total = len(x_test) // BATCH_SIZE + 1
        with torch.no_grad():
            with tqdm(total=total, desc="Epoch {}".format(epoch)) as pbar:
                for batch in range(len(x_test) // BATCH_SIZE + 1):
                    inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
                    logits = model(x_test[inds].to(device), mask_test[inds].to(device))
                    loss = criterion(logits, y_test[inds].to(device))
                    loss_test += loss.cpu().item()
                    steps_test += 1
                    pbar.update(1)
                    labels = torch.sigmoid(logits.detach().cpu()).numpy()
                    labels[labels > 0.5], labels[labels <= 0.5] = 1, 0
                    labels_pred += list(labels.reshape(-1))
                    labels_real += list(y_test[inds].numpy().reshape(-1))
                    pbar.set_postfix_str("Testing Loss: {:.5f}, Precision: {:.2f}, Recall: {:.5f}".format(
                        loss_train / steps_train, precision_score(labels_real, labels_pred), recall_score(labels_real, labels_pred))
                    )
        acc_test = accuracy_score(labels_real, labels_pred)
        prec_test = precision_score(labels_real, labels_pred)
        recall_test = recall_score(labels_real, labels_pred)
        cf_test = confusion_matrix(labels_real, labels_pred)

        print("Epoch {} | Train Loss {:.5f}, Acc {:.2f}, Precision {:.2f}, Recall {:.2f}"
              " - Test Loss {:.5f}, Acc {:.2f}, Precision {:.2f}, Recall {:.2f}".format(
               epoch, loss_train / steps_train, acc_train, prec_train, recall_train, loss_test / steps_test, acc_test, prec_test, recall_test))
        print(cf_train)
        print(cf_test)

        if recall_test > recall_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "saved_models_BERT_sigmoid/BERT_{}layers_{}features_{}len.pt".format(N_LAYERS, N_FEATURES, SEQ_LEN))
            print("A new model has been saved!")
            recall_test_best = recall_test

# %% ----------------------------------------- Final Test --------------------------------------------------------------
if FINAL_TEST:
    print("Computing metrics on test set...")
    labels_pred, labels_real = [], []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(x_test) // BATCH_SIZE + 1) as pbar:
            for batch in range(len(x_test) // BATCH_SIZE + 1):
                inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
                logits = model(x_test[inds].to(device), mask_test[inds].to(device))
                pbar.update(1)
                labels = torch.sigmoid(logits.detach().cpu()).numpy()
                labels[labels > 0.5], labels[labels <= 0.5] = 1, 0
                labels_pred += list(labels.reshape(-1))
                labels_real += list(y_test[inds].numpy().reshape(-1))
    print("Test Acc {:.2f}, Test Recall {:.2f}".format(accuracy_score(labels_real, labels_pred), recall_score(labels_real, labels_pred)))
    print(confusion_matrix(labels_real, labels_pred))

# %% ----------------------------------------- Save Features -----------------------------------------------------------
if EXTRACT_FEATURES:
    model.eval()
    if "saved_features_BERT_sigmoid" not in os.listdir():
        os.mkdir("saved_features_BERT_sigmoid")
    print("Saving BERT last cls weights...")
    with open("saved_models_BERT_sigmoid/BERT_last_weights{}layers_{}features_{}len.txt".format(N_LAYERS, N_FEATURES, SEQ_LEN), "w") as s:
        s.write(str(model.cls.bias.data.cpu().numpy()) + " - " + str(model.cls.weight.data.cpu().numpy()))
    print("Extracting and saving features...")
    features_train, features_test = get_features(x_train_, mask_train_, model), get_features(x_test, mask_test, model)
    np.save("saved_features_BERT_sigmoid/features_train_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_train)
    np.save("saved_features_BERT_sigmoid/features_test_{}layers_{}features_{}len.npy".format(N_LAYERS, N_FEATURES, SEQ_LEN), features_test)
    print("The features have been saved!")
