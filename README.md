# SentimentAnalaysisGRU
This project aims to perform binary classification of Reddit posts according to the estimated psychological stability of their authors. 

Methodology
	1. Train.py
		a. A modified training dataset filled with labelled Reddit posts (training.csv) will be converted into a dataframe. 
		b. Each post will be preprocessed via lowercasing, tokenizing and then stemming. 
		c. Using the Word2Vec embedding model from the gensim library, the tokenized sentences form the vocabulary of the model and word embeddings are generated for each word in the preprocessed text. Model is then saved as (w2v.model).
		d. Preprocessed training data is split into train and validation sets.
		e. Batches are generated for both sets.
		f. Model class is defined.
		g. Model, optimiser, loss function, accuracy function and learning rate scheduler are then initialised.
		h. Model is trained and validated.
		f. Model is saved in (trained_model.pt). 
	2. Test.py
		a. Retrieve posts from desired subreddits as well as their metadata and consolidate into dataframe via praw library.
		b. Same as Train.py.
		c. (w2v.model) is loaded and will continue building its vocabulary using the test data. Word embeddings generated for test data. 
		d. Batches generated for test set.
		e. Load trained model from (trained_model.pt).
		f. Model is initialised
		g. Run model on test set.
		h. Username generated. 
	3. StreamlitApp.py
