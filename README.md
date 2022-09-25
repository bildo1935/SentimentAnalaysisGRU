# Aim
This project aims to perform binary classification of Reddit posts according to the estimated psychological stability of their authors.

# Usage
Simply access the app via https://nvg1356-sentimentanalaysisgru-streamlitapp-mhd6kn.streamlitapp.com/. 

# Methodology
## Main Software and Frameworks Used
1. **JupyterLab** was used as the development environment.
2. **Pandas** and **Numpy** were used for construction of datasets.
3. **PRAW** was used to retrieve real time posts and post metadata from subreddits.
4. **Gensim** was used to convert word tokens to numerical vectors.
5. **NLTK** was used for text preprocessing.
6. **Streamlit** was used to convert the code into an accessible application. 

## Training
1.  The labelled training data was obtained from a paper named **On the State of Social Media Data for Mental Health Research** by Keith Harrigian, Carlos Aguirre and Mark Dredze. The text was labelled either 1 or 0, with 1 representing mental instability and 0 representing a relatively normal mental state.  
> Paper can be found at https://arxiv.org/abs/2011.05233. 
2.  Text from training data is first lowercased then tokenized and stemmed.
3.  Preprocessed text used to generate word embeddings by the **Gensim Word2Vec** model. The trained word embedding model is saved as **w2v.model**.
4. Preprocessed text is split into training and validation sets.
5. Training and validation sets are fed into **Pytorch Dataloader** to generate batches.
6. **TextClassifier** class, which encompasses the **recurrent neural network with gated recurrent units**, is defined.
7. **Stochastic Gradient Descent**, **Binary Cross Entropy Loss with Logits**, **Accuracy from Torchmetrics** and **Exponential Learning Rate Scheduler** are initialised as the optimisation algorithm, loss function, accuracy metric and learning rate adjuster respectively.
8. For each iteration, batches of training data are fed into the initialised model and the model parameters are adjusted according to the loss function every batch with the aim of decreasing the loss. 
9. The trained model is tested on validation data. 
10. The trained classification model is saved as **trained_model.pt**.

Training file can be found in `Train.py`.

## Testing

1.

Testing file can be found in `Test.py`.

## Conversion to App.


