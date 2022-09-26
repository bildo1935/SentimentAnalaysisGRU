#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
import gensim
import torchmetrics
import subprocess


# In[3]:


#CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available()==True else "cpu")


# In[10]:


nontest_df = pd.read_csv("training.csv")
nontest_df = nontest_df.sample(frac=1).reset_index(drop=True)


# In[11]:


texts = list(nontest_df['body'])
label = list(nontest_df['label'])
total_data = list(zip(texts, label))

for i in range(len(texts)):
    # convert text to lowercase
    text = texts[i].lower()
    # word tokenizing
    tokens = nltk.tokenize.word_tokenize(text)
    # stemming tokens
    stemmer = nltk.SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    texts[i] = tokens

embedding_model = gensim.models.Word2Vec(sentences=texts, min_count=1, workers=5, window=3, sg=0, vector_size=100)
embedding_model.save('w2v.model')
word2index = {token: token_index for token_index, token in enumerate(embedding_model.wv.index_to_key)}
index2word = {index: token for token, index in enumerate(embedding_model.wv.key_to_index)} 
        
def text_preprocessing(text):
    # convert text to lowercase
    text = text.lower()
    # word tokenizing
    tokens = nltk.tokenize.word_tokenize(text)
    # stemming tokens
    stemmer = nltk.SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [word2index[word] for word in tokens]
    return tokens

def collate_batch(batch):
    label_list, text_list = [], [] 
    for (_text, _label) in batch:
        label_list.append(_label)
        haha = text_preprocessing(_text)
        processed_text = torch.LongTensor(haha)
        text_list.append(processed_text)
    label_list = torch.Tensor(label_list)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    text_list, label_list = text_list.to(device), label_list.to(device)
    return text_list, label_list


# In[12]:


def train_val_split(train_size, val_size):
    train_data = total_data[:train_size]
    val_data = total_data[train_size:]
    return train_data, val_data

train_data, val_data = train_val_split(1950, 450)

train_dataloader = DataLoader(train_data, batch_size=10, collate_fn=collate_batch, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=10, collate_fn=collate_batch, shuffle=False)


# In[13]:


class TextClassifier(nn.Module):
    
    def __init__(self, vocab_size, num_classes, hidden_size, num_layers, batch_first, embedding_size):
        super(TextClassifier, self).__init__()
        self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_model.wv.vectors), padding_idx=0)
        
    def forward(self, x_in):
        x_in = self.embedding(x_in)
        _, y_out = self.RNN(x_in)
        y_out = self.fc(y_out)
        y_out = torch.squeeze(y_out, dim=0)
        y_out = y_out.view(10)
        return y_out


# In[17]:


optimiser = torch.optim.SGD(TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=8, num_layers=1, batch_first=True, embedding_size=100).parameters(), lr=0.1, nesterov=True, momentum=0.9)
loss_func = nn.BCEWithLogitsLoss()
accuracy = torchmetrics.Accuracy(num_classes=1, threshold=0.5).to(device)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)


# In[19]:


#train

epochs = 50

for epoch in range(epochs):
    for batch_text, batch_labels in train_dataloader:
        optimiser.zero_grad()
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=8, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text)
        loss = loss_func(y_pred, batch_labels.float())
        acc = accuracy(preds=y_pred, target=batch_labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100).parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
        optimiser.step()
    print('check')
    print(loss, acc)


# In[20]:


#val

epoch = 25

for epoch in range(epochs):
    for batch_text, batch_labels in val_dataloader:
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text)
        loss = loss_func(y_pred, batch_labels.float())
        acc = accuracy(preds=y_pred, target=batch_labels.long())
    print(loss, acc)


# In[21]:


torch.save(machine, "trained_model.pt")

