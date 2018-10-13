# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 22:30:00 2018

@author: jayascript
"""

import nltk
import numpy as np
import pandas as pd
import re

from contractions import CONTRACTION_MAP
from nltk.corpus import stopwords
from textblob import Word

filepath = "./data/"
filename = input("Filename: ")

file = filepath + filename

print("Loading {}...".format(file))
data = pd.read_pickle(file)
df = data.copy()

print("A brief look at the data...")
print(df.head())
print("---------------------------")
print(df.tail())

print("Step 1: Removing noise...")

url_pattern = "http[^\s]+\s?…?"
mentions_pattern = '@\s[A-Za-z0-9_]+'

df.loc[:, 'tweet_text'].replace(url_pattern, " ", regex=True, inplace=True)
df.loc[:, 'tweet_text'].replace(mentions_pattern, " ", regex=True, inplace=True)
df.loc[:, 'tweet_text'].replace("#", "", regex=True, inplace=True)

print("Done.")
print(df.head())

print("Step 2: Normalizing text...")

# The contractions.py file used in this section was downloaded from
# DJ Sarkar's repo, and the code was made available from his Guide to NLP
# on Towards Data Science. Links below:
# https://bit.ly/2OOJ0hN <-- Repo
# https://bit.ly/2RIvf2H <-- NLP Guide

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text

df.loc[:, 'tweet_text'].replace("’", "'", regex=True, inplace=True)
df['tweet_text'] = df['tweet_text'].apply(lambda x: expand_contractions(x))
df.loc[:, 'tweet_text'].replace("[^a-zA-Z\s]", " ", regex=True, inplace=True)
df['tweet_text'] = df['tweet_text'].apply(lambda x: x.lower())

print("Done.")
print(df.head())

print("Step 3: Preprocessing text...")

df['tweet_text'].replace(' ', np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True)

df['processed_tweet'] = df['tweet_text'].apply(lambda x: nltk.word_tokenize(x))

stop = stopwords.words("english")
df['processed_tweet'] = df['processed_tweet'].apply(lambda x: [word for word in x if word not in (stop) and len(word) > 1])

df['processed_tweet'] = df['processed_tweet'].apply(lambda x: [Word(word).lemmatize() for word in x])

df['processed_tweet'] = df['processed_tweet'].apply(lambda x: [word for word in x if len(word) > 1])

print("Done.")
print("Your cleaned data:")
print(df.head())
print("---------------------------")
print(df.tail())

filepath = "./data/"
new_filename = "CLEANED_" + input("Save file as: ")

newfile = filepath + new_filename

print("Saving {}...".format(newfile))
df.to_pickle(newfile)
print("Done. End.")
