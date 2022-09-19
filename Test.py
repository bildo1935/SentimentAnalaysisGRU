#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import random


# In[2]:


#CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available()==True else "cpu")


# In[3]:


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


# In[4]:


texts = list(test_df['body'])
label = list(test_df['label'])

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
word2index = {token: token_index for token_index, token in enumerate(embedding_modelwv.index_to_key)}
index2word = {index: token for token, index in enumerate(embedding_modelwv.key_to_index)} 
        
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


# In[5]:


test_dataloader = DataLoader(total_data, batch_size=20, collate_fn=collate_batch, shuffle=False)


# In[9]:


machine = torch.load("trained_model.pt")
optimiser = torch.optim.SGD(machine.parameters(), lr=0.1, nesterov=True, momentum=0.9)
loss_func = nn.HingeEmbeddingLoss()
accuracy = torchmetrics.Accuracy(num_classes=1, threshold=0.5).to(device)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)


# In[12]:


#test

at_risk_users=[]

epochs = 25
          
for epoch in range(epochs):
    batch_index = 0
    for batch_text, batch_labels in test_dataloader:
        machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100)
        machine = machine.to(device)
        y_pred = machine(x_in=batch_text)
        for i in range(len(y_pred)):
            if y_pred[i]>0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
            if y_pred[i] == 1:
               at_risk_users.append(df3.iloc[(20*batch_index)+i]['author'])
        batch_index += 1

print(f"Flagged User: {random.choice(at_risk_users)}")

