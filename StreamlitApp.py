#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
import gensim
import praw
import random

#App

st.title('Welcome to AI4Good.')

st.header('Introduction.')
st.markdown(f'''
Recent volatility and turbulence has caused a marked increase in distress of the Singaporean population. 
It has been found that many who suffer from said mental distress attempt to find relief through (anonymous) participation  
in online communities e.g. Reddit.
''')

st.header('How it works.')
st.markdown(f'''This project mines three subreddits (r/Singapore, r/NationalServiceSG and r/SingaporeRaw) for posts as well as their authors. 
It preprocesses the text and transforms it via Word2Vec embedddings.
It is then fed as input into a trained Recurrent Neural Network with Gated Recurrent Units. 
The output is in the form of two classes (0 or 1 each representing mental stability and a relative lack thereof respectively).
Authors whose texts are classified as representing psychological instability are flagged up.''')

st.markdown(f'''The dataset used for training is a modified version of the training data used in  
[On the State of Social Media Data for Mental Health Research](https://arxiv.org/abs/2011.05233).  
More of such datasets can be found [here](https://github.com/kharrigian/mental-health-datasets).
''')

roller = st.button('Execute Model.')
if roller:
    st.write(generate_user(run=True)

def generate_user(run):
    if run:
        #Retrieve posts
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

        #Text Preprocessing
        nltk.download('punkt')
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

        embedding_model = gensim.models.Word2Vec.load('w2v.model')
        embedding_model.build_vocab(texts, update=True)
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

        #Generate batches
        test_dataloader = DataLoader(total_data, batch_size=20, collate_fn=collate_batch, shuffle=False)

        #Defining Class
        class TextClassifier(nn.Module):

            def __init__(self, vocab_size, num_classes, hidden_size, num_layers, batch_first, embedding_size):
                super(TextClassifier, self).__init__()
                self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
                self.fc = nn.Linear(in_features=hidden_size, out_features=1)
                self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_model.wv.vectors), padding_idx=0)
                self.act = nn.Sigmoid()

            def forward(self, x_in):
                x_in = self.embedding(x_in)
                _, y_out = self.RNN(x_in)
                y_out = self.fc(y_out)
                y_out = torch.squeeze(y_out, dim=0)
                y_out = y_out.view(20)
                #softmax = torch.nn.Softmax(dim=1)
                y_out = self.act(y_out)
                return y_out

        #Loading pre-trained model
        machine = torch.load("trained_model.pt",  map_location = torch.device('cpu'))

        #Running Model

        at_risk_users=[]

        epochs = 25

        for epoch in range(epochs):
            batch_index = 0
            for batch_text, batch_labels in test_dataloader:
                machine = TextClassifier(vocab_size=len(embedding_model.wv), num_classes=2, hidden_size=1, num_layers=1, batch_first=True, embedding_size=100)
                machine = machine.to(device)
                y_pred = machine(x_in=batch_text)
                #activation = nn.Hardtanh(0, 1)
                #y_pred = activation(y_pred)
                for i in range(len(y_pred)):
                    if y_pred[i]>0.5:
                        y_pred[i] = 1
                    else:
                        y_pred[i] = 0
                    if y_pred[i] == 1:
                        at_risk_users.append(df3.iloc[(20*batch_index)+i]['author'])
                batch_index += 1

        flaggeduser = random.choice(at_risk_users)
        return flaggeduser
    
    else:
        None

