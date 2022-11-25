#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:59:24 2022

@author: yoshithaakunuri
"""
# !pip install spacy1
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download en_core_web_sm

import spacy 
from spacy import displacy 
nlp = spacy.load("en_core_web_sm")
sentence ="you must make alternative communication arrangements to ensure that you can make emergency calls if needed . "
doc = nlp(sentence)
print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")
for token in doc:
    print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
    
# displacy.render(doc, style='dep')


import stanza
# stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp("you must make alternative communication arrangements to ensure that you can make emergency calls if needed . ")
for sent in doc.sentences:
    for word in sent.words:
        print(f'id:{word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}', sep='\n')



import spacy 
from spacy import displacy 
nlp = spacy.load("en_core_web_sm")
sentence ="cat is chased by dog"
doc = nlp(sentence)
print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")

for token in doc:
    if token.dep_ == 'nsubjpass':
        print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
        
       
        
       
#######################################################################################        
        
def check_passive(dependencies):
    if 'agent' in dependencies or 'nsubjpass' in dependencies:
        return 1
    else:
        return 0
    
    
import pandas as pd
clauses_df = pd.read_csv("Obligations_detected.csv")
clauses_df.columns
# clauses_df["Sentences"] = clauses_df["Sentences"].replace('-lrb-', '')
clauses_df = clauses_df[clauses_df["Obligatory_flag"] == 1]

clauses = list(clauses_df.Sentences)

subjects_dict = {}
        
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

# clauses = ['dog is chasing cat', 'cat is chased by dog', 'you must make alternative communication arrangements to ensure that you can make emergency calls if needed . ', 'you may be surprised , but we will refer to all such services , including additional apps published by viber media s.a.r.l , and also third party services which are powered by viber s technology , as the  services  . ']

# print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")

index = 1
for clause in clauses:
    print(index)
    doc = nlp(clause)
    
    dependencies = [token.dep_ for token in doc]
    passive = check_passive(dependencies)
    
    for token in doc:
        if passive == 1:
            if token.dep_ == 'pobj':
                # print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
                subjects_dict[str(token.text)] = subjects_dict.get(str(token.text), 0) + 1
        else:
            if token.dep_ == 'nsubj':
                # print("{:<15} {:^10} {:>15}".format(str(token.head.text), str(token.dep_), str(token.text)))
                subjects_dict[str(token.text)] = subjects_dict.get(str(token.text), 0) + 1
    index += 1

        


# sub_list_to_filter = ["you", "arbitrator", "party", "parties", "users", "user", "they", "anyone", "waiver", "arbitrators", "guests", "guest", "buyer", "players", "owner", "owners", "partner", "partners", "waivers", "buyers", "subscriber", "subscribers"]
