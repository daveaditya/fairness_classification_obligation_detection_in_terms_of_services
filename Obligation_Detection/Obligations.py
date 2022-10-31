#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:45:04 2022

@author: yoshithaakunuri
"""

# Import statements
import os
import pandas as pd
import re
import contractions
# import nltk
from gensim.models import Word2Vec
import gensim.downloader as api

# import warnings
# warnings.filterwarnings("ignore")

# Load Data
sentences_path = "/Users/yoshithaakunuri/Documents/NLP/Project/Data/ToS/Sentences/"
labels_path = "/Users/yoshithaakunuri/Documents/NLP/Project/Data/ToS/Labels/"

clauses_df = pd.DataFrame(columns = ["Sentences", "Labels"])
for file in os.listdir(sentences_path):
    sent_path = sentences_path + file 
    lab_path = labels_path + file
    
    with open(sent_path, 'r', encoding="utf-8") as sf:
        sentences = sf.readlines()
    sf.close()

    with open(lab_path, 'r', encoding="utf-8") as lf:
        labels = lf.readlines()
    lf.close()

    temp = pd.DataFrame(zip(sentences, labels), columns = ["Sentences", "Labels"])
    clauses_df = pd.concat([clauses_df,temp], axis=0)
    
    # print(len(clauses_df))
    
clauses_df.to_csv("Clauses.csv", index = False)


### Preprocessing

clauses_df = pd.read_csv("Clauses.csv")

clauses_df["Sentences_Processed"] = clauses_df["Sentences"].copy()

# Remove URLs
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply( lambda x:re.sub(r"\S*https?:\S*", " ", x))

# Replace Contractions
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply( lambda x: contractions.fix(x))

## Special Characters
# Remove all special characters and numbers from text
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].replace( r'[^A-Za-z\s]+', ' ', regex = True)
# Remove extra spaces
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].replace( r'\s+', ' ', regex=True)


# Lower case
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].str.lower()

#Split words
clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply( lambda x: x.split())

clauses_df.to_csv("Clauses_Preprocessed.csv", index = False)
# Lemmatization
# clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply( lambda x: [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in x])
# clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply(lambda x: [nltk.stem.WordNetLemmatizer().lemmatize( word, pos = 'a') for word in x])
# clauses_df["Sentences_Processed"] = clauses_df["Sentences_Processed"].apply( lambda x: [nltk.stem.WordNetLemmatizer().lemmatize(word, pos = 'v') for word in x])



## Creating Word2Vec Model from Clauses
 
# Skip-gram model (sg = 1)
size = 300
window = 11 
min_count = 10 
sg = 1
word2vec_model_file = 'word2vec_' + str(size) + '.model' 
stemmed_tokens = pd.Series(clauses_df["Sentences_Processed"]).values # Train the Word2Vec Model
w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, vector_size = size, window = window, sg = sg)
w2v_model.save(word2vec_model_file)

# Load W2V model
sg_w2v_model = Word2Vec.load(word2vec_model_file) 
custom_wv = sg_w2v_model.wv
custom_wv.key_to_index

# Loading Google Word2Vec
google_wv = api.load('word2vec-google-news-300')

# printing Relevant words to Obligation

custom_wv.most_similar(positive=["must"], topn=5)
google_wv.most_similar(positive=["must"], topn=10)
google_wv.most_similar(positive=["oblige"], topn=10)
google_wv.most_similar(positive=["obligatory"], topn=10)


# Identify clauses that are obligatory

keywords = ["must", "should", "ought", "will", "need", "shall", "required", 
            "oblige", "obliged", "obliging", "relent", "obliges", "abide", "compel", 
            "obligatory", "compulsory", "perfunctory", "mandatory"]

def DetectObligations(sentence, keywords):
    if any(word in sentence for word in keywords):
        return 1
    else:
        return 0
    
clauses_df["Obligatory_flag"] = clauses_df["Sentences_Processed"].apply(lambda x: DetectObligations(x, keywords))

clauses_df.to_csv("Obligations_detected.csv", index = False)
