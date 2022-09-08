#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
import gensim
import praw
import torchmetrics
import torchtext
import streamlit
import random
import subprocess


# In[2]:


#CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available()==True else "cpu")


# In[3]:


nontest_df = pd.read_csv("training.csv")


# In[4]:


reddit = praw.Reddit(client_id='Z5wGLXCyVogiLcexFuLHsQ', 
                    client_secret='OHOSmhNxBrKaVRgsr2NP1daHJz0fJA',
                    user_agent='ScrapingApp',
                    username='Opening-Method8562',
                    password='acsi2017!')

SGExams_subreddit = reddit.subreddit('SGExams').hot(limit=200)
NationalServiceSG_subreddit = reddit.subreddit('NationalServiceSG').hot(limit=200)
SingaporeRaw_subreddit = reddit.subreddit('SingaporeRaw').hot(limit=200)

dict = {"title":[],
                "subreddit":[],
                "id":[], 
                "url":[], 
                "author": [], 
                "body":[]}

for submission in SGExams_subreddit:
    dict['title'].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict['id'].append(submission.id)
    dict['url'].append(submission.url)
    dict['author'].append(submission.author)
    dict['body'].append(submission.selftext)
    
for submission in NationalServiceSG_subreddit:
    dict['title'].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict['id'].append(submission.id)
    dict['url'].append(submission.url)
    dict['author'].append(submission.author)
    dict['body'].append(submission.selftext)
    
for submission in SingaporeRaw_subreddit:
    dict['title'].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict['id'].append(submission.id)
    dict['url'].append(submission.url)
    dict['author'].append(submission.author)
    dict['body'].append(submission.selftext)
    
df3 = pd.DataFrame.from_dict(dict, orient='columns') 
df3['body'] = df3['title'] + df3['body'] 
test_df = pd.DataFrame(df3['body'])
test_df['label'] = np.NaN 


# In[5]:


df5 = pd.concat([nontest_df, test_df], ignore_index=True)

texts = df5['body'].tolist()
label = df5['label'].tolist()

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
embedding_modelwv = embedding_model.wv
embedding_modelwv.save('vectors.kv')
reloaded_word_vectors = gensim.models.KeyedVectors.load('vectors.kv')
word2index = {token: token_index for token_index, token in enumerate(reloaded_word_vectors.index_to_key)}
index2word = {index: token for token, index in enumerate(reloaded_word_vectors.key_to_index)} 
        
def text_preprocessing(text):
    # convert text to lowercase
    text = text.lower()
    # word tokenizing
    tokens = nltk.tokenize.word_tokenize(text)
    # stemming tokens
    stemmer = nltk.SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [word2index['word'] for word in tokens]
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


# In[6]:


def train_val_test_split(train_size, val_size, test_size):
    train_data = total_data[:(train_size)]
    val_data = total_data[train_size:train_size+val_size]
    test_data = total_data[train_size+val_size:len(total_data)]
    return train_data, val_data, test_data

train_data, val_data, test_data = train_val_test_split(1950, 450, 600)

train_dataloader = DataLoader(train_data, batch_size=10, collate_fn=collate_batch, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=10, collate_fn=collate_batch, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=20, collate_fn=collate_batch, shuffle=False)


# In[7]:


class TextClassifier(nn.Module):
    
    def __init__(self, vocab_size, num_classes, hidden_size, num_layers, batch_first, embedding_size):
        super(TextClassifier, self).__init__()
        self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(reloaded_word_vectors.vectors), padding_idx=0)
        
    def forward(self, x_in, nontest):
        x_in = self.embedding(x_in)
        _, y_out = self.RNN(x_in)
        y_out = self.fc(y_out)
        y_out = torch.squeeze(y_out, dim=0)
        if nontest:
            y_out = y_out.view(10)
        else:
            y_out.view(20)
            softmax = torch.nn.Softmax(dim=1)
            y_out = softmax(y_out)
        return y_out


# In[11]:


optimiser = torch.optim.SGD(TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=8, num_layers=1, batch_first=True, embedding_size=100).parameters(), lr=0.1, nesterov=True, momentum=0.9)
loss_func = nn.HingeEmbeddingLoss()
accuracy = torchmetrics.Accuracy(num_classes=1, threshold=0.5).to(device)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)


# In[12]:


#train

epochs = 50

for epoch in range(epochs):
    for batch_text, batch_labels in train_dataloader:
        optimiser.zero_grad()
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=8, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text, nontest=True)
        loss = loss_func(y_pred, batch_labels.float())
        acc = accuracy(preds=y_pred, target=batch_labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100).parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
        optimiser.step()
    scheduler.step()
    #print(loss, acc)


# In[14]:


#val

epoch = 25

for epoch in range(epochs):
    for batch_text, batch_labels in val_dataloader:
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text, nontest=True)
        loss = loss_func(y_pred, batch_labels.float())
        acc = accuracy(preds=y_pred, target=batch_labels.long())
    #print(loss, acc)


# In[18]:


#test

at_risk_users=[]

epochs = 25
          
for epoch in range(epochs):
    batch_index = 0
    for batch_text, batch_labels in test_dataloader:
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text, nontest=False)
        for i in range(len(y_pred)):
            if y_pred[i]>0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
            if y_pred[i] == 1:
               at_risk_users.append(df3.iloc[(20*batch_index)+i]['author'])
        batch_index += 1

print(f"Flagged User: {random.choice(at_risk_users)}")


# In[ ]:


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

