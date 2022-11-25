#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:31:33 2022

@author: yoshithaakunuri
"""

# -------- Import statements --------
import sys
import pandas as pd
import re
import contractions

from gensim.models import Word2Vec
import gensim.downloader as api

import spacy
from spacy import displacy



# -------- Load Data --------
input_path = sys.argv[1]

# input_path = '/Users/yoshithaakunuri/Documents/NLP/Project/Data/ToS/Sentences/9gag.txt'
output_path = "Obligatory_clauses.txt"
# print(input_path)
    
with open(input_path, 'r', encoding="utf-8") as sf:
    sentences = sf.readlines()
sf.close()

dummy = [1 for i in range(len(sentences))]
clauses_df = pd.DataFrame(zip(sentences, dummy), columns = ["Sentences", "Dummy"])


# -------- PRE-PROCESSING --------
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


# -------- Identify clauses that are obligatory --------

keywords = ["must", "should", "ought", "will", "need", "shall", "required", 
            "oblige", "obliged", "obliging", "relent", "obliges", "abide", "compel", 
            "obligatory", "compulsory", "perfunctory", "mandatory"]

def DetectObligations(sentence, keywords):
    if any(word in sentence for word in keywords):
        return 1
    else:
        return 0
    
clauses_df["Obligatory_flag"] = clauses_df["Sentences_Processed"].apply(lambda x: DetectObligations(x, keywords))

# --------- Identifying Obligations to users --------

clauses_df = clauses_df[clauses_df["Obligatory_flag"] == 1]

def check_passive(dependencies):
    if 'agent' in dependencies or 'nsubjpass' in dependencies:
        return 1
    else:
        return 0
    
clauses = list(clauses_df.Sentences_Processed)

# subjects_dict = {}
nlp = spacy.load('en_core_web_sm')

sub_list_to_filter = ["you", "arbitrator", "party", "parties", "users", "user", "they", "anyone", "waiver", "arbitrators", "guests", "guest", "buyer", "players", "owner", "owners", "partner", "partners", "waivers", "buyers", "subscriber", "subscribers"]

obligatory_clauses = []

for clause in clauses:
    doc = nlp(clause)
    
    dependencies = [token.dep_ for token in doc]
    passive = check_passive(dependencies)
    
    for token in doc:
        if passive == 1:
            if token.dep_ == 'pobj':
                # print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
                if str(token.text) in sub_list_to_filter:
                    obligatory_clauses.append(clause)
                    
                # subjects_dict[str(token.text)] = subjects_dict.get(str(token.text), 0) + 1
        else:
            if token.dep_ == 'nsubj':
                # print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
                # subjects_dict[str(token.text)] = subjects_dict.get(str(token.text), 0) + 1
                if str(token.text) in sub_list_to_filter:
                    obligatory_clauses.append(clause)


# -------- Output Obligatory Clauses to user --------

with open(output_path, 'w', encoding='utf-8') as f:
        f.truncate(0)
        f.write("Obligatory Clauses:\n\n")
        for c in obligatory_clauses:
            f.write("- ")
            f.write(f'{c}\n')






