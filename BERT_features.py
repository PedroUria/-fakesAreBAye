# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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

# %% --------------------------------------- Hyper-Parameters ----------------------------------------------------------
N_LAYERS = 2

EPOCHS = 10
LR = 1e-5
EPS = 1e-8
BATCH_SIZE = 32
WARM_UP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if "prep_data" not in os.listdir():
    from data_eda import data_restaurants
    os.mkdir("prep_data")
    x, x_mask = [], []
    print("Tokenizing the reviews...")
    with tqdm(total=len(data_restaurants)) as pbar:
        for review in data_restaurants["Review"].values:
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review)[:510])
            token_ids = [101] + token_ids + [102]
            n_ids = len(token_ids)
            attention_mask = [1] * n_ids
            if n_ids < 512:
                token_ids += [0] * (512 - n_ids)
                attention_mask += [0] * (512 - n_ids)
            x.append(token_ids)
            x_mask.append(attention_mask)
            pbar.update(1)
    x = np.array(x)
    y = LabelEncoder().fit_transform(data_restaurants["Fake"].values)
    np.save("prep_data/x.npy", x); np.save("prep_data/x_mask.npy", x_mask); np.save("prep_data/y.npy", y)
else:
    print("-"*50)
    os.system("python3 data_eda.py")
    print("-"*50)
    x, x_mask, y = np.load("prep_data/x.npy"), np.load("prep_data/x_mask.npy"), np.load("prep_data/y.npy")

x_train, x_test, mask_train, mask_test, y_train, y_test = train_test_split(
    x, x_mask, y, random_state=SEED, test_size=0.3, stratify=y)
x_train, mask_train, y_train = torch.LongTensor(x_train), torch.FloatTensor(mask_train), torch.LongTensor(y_train)
x_test, mask_test, y_test = torch.LongTensor(x_test), torch.FloatTensor(mask_test), torch.LongTensor(y_test)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_classes=2).to(device)
model.bert.encoder.layer = model.bert.encoder.layer[:N_LAYERS]
optimizer = AdamW(model.parameters(), lr=LR, eps=EPS)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARM_UP_STEPS,
                                 t_total=len(x_train) // GRADIENT_ACCUMULATION_STEPS * EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
acc_test_best = 0
inds_list = list(range(len(x_train)))
print("Starting training loop...")
for epoch in range(10):

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
            logits, *_ = model(x_train[inds].to(device), attention_mask=mask_train[inds].to(device))
            loss = criterion(logits, y_train[inds].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train += loss.cpu().item()
            steps_train += 1
            pbar.update(1)
            pbar.set_postfix_str("Training Loss: {:.5f}".format(loss_train / steps_train))
            labels_pred += list(np.argmax(logits.detach().cpu().numpy(), axis=1).reshape(-1))
            labels_real += list(y_train[inds].numpy().reshape(-1))
    acc_train = accuracy_score(labels_real, labels_pred)

    loss_test, steps_test, labels_pred, labels_real = 0, 0, [], []
    model.eval()
    total = len(x_test) // BATCH_SIZE + 1
    with torch.no_grad():
        with tqdm(total=total, desc="Epoch {}".format(epoch)) as pbar:
            for batch in range(len(x_test) // BATCH_SIZE + 1):
                inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
                logits, *_ = model(x_test[inds].to(device), attention_mask=mask_test[inds].to(device))
                loss = criterion(logits, y_test[inds].to(device))
                loss_test += loss.cpu().item()
                steps_test += 1
                pbar.update(1)
                pbar.set_postfix_str("Training Loss: {:.5f}".format(loss_test / steps_test))
                labels_pred += list(np.argmax(logits.detach().cpu().numpy(), axis=1).reshape(-1))
                labels_real += list(y_test[inds].numpy().reshape(-1))
    acc_test = accuracy_score(labels_real, labels_pred)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train / steps_train, acc_train, loss_test / steps_test, acc_test))

    if acc_test > acc_test_best:
        torch.save(model.state_dict(), "BERT.pt")
        print("A new model has been saved!")
        acc_test_best = acc_test
