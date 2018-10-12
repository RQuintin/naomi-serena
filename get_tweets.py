# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 22:30:00 2018

@author: jayascript
"""

import langid
import pandas as pd

print("Loading data...")

naomi_file = "./data/Naomi_Eng.csv"
serena_file = "./data/Serena_Eng.csv"

print("Getting 'naomi osaka' tweets...")
naomi_df = pd.read_csv(naomi_file)
naomi_df = naomi_df.drop(columns=["tweet_loc", "tweet_id"])
naomi_df = naomi_df.drop_duplicates("tweet_text")
naomi_df = naomi_df.dropna()
naomi_df = naomi_df[naomi_df['tweet_text'].apply(lambda x: langid.classify(x)[0] == 'en')]
naomi_df['search query'] = 'naomi osaka'
print("Done.")

print("Getting 'serena williams' tweets...")
serena_df = pd.read_csv(serena_file)
serena_df = serena_df.drop(columns=["tweet_loc", "tweet_id"])
serena_df = serena_df.drop_duplicates("tweet_text")
serena_df = serena_df.dropna()
serena_df = serena_df[serena_df['tweet_text'].apply(lambda x: langid.classify(x)[0] == 'en')]
serena_df['search query'] = 'serena williams'
print("Done.")

print("Combining data...")
naomi_serena_tweets = pd.concat([naomi_df, serena_df])
naomi_serena_tweets = naomi_serena_tweets.drop(columns=["id"])
naomi_serena_tweets.reset_index(drop=True, inplace=True)

print("Here are your tweets:")
print(naomi_serena_tweets.head())
print("***************")
print(naomi_serena_tweets.tail())

filepath = "./data/"
new_filename = "ALL_" + input("Save file as .pkl: ")

newfile = filepath + new_filename

print("Saving {}...".format(newfile))
naomi_serena_tweets.to_pickle(newfile)
print("Done. End.")
