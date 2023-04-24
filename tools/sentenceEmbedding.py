import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# coding: UTF-8
import os
import datetime

import numpy
import pandas as pd
import torch
import numpy as np
import random
import pickle as pkl
from tqdm import tqdm
import math
import time
from datetime import timedelta

def generate_sent_emb(sentence):
    embedding = model.encode(sentence)
    return embedding

def generate_df_emb(series):
    sentence = series["text"]
    embedding = model.encode(sentence)
    return embedding

def build_branch(df, now_id, branch):
    temp_df = df.loc[df['id'] == now_id]
    if len(temp_df) == 0: # error
        return branch
    branch.append(now_id)
    if temp_df['is_post'].iloc[0] == True:
        return branch #is top node
    else:
        next_id = temp_df['replied_to_tweet_id'].iloc[0]
        if len(df.loc[df['id'] == next_id]) > 0:
            #next_round
            branch = build_branch(df, next_id, branch)
        else:
            branch.append(temp_df['conversation_id'].iloc[0])
            return branch
    return branch

def check_error(df, i, branch):
    if  df['replied_to_tweet_id'].iloc[i] != 0:
        parent = df['replied_to_tweet_id'].iloc[i]
        if parent != branch[-2]:
            print("found_error: ", branch[-2], parent)
            df['replied_to_tweet_id'].iloc[i] = branch[-2]
    return df

def get_branch(df):
    print('building branches...')
    for idx in tqdm(range(len(df))):
        branch = []
        now_id = df['id'].iloc[idx]
        branch = build_branch(df, now_id, branch)
        branch.reverse()
        df = check_error(df, idx, branch)
        df['branch'].iloc[idx] = branch
    return df

model = SentenceTransformer('all-MiniLM-L6-v2')

# df = pd.read_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset.csv')
# df = df.drop(columns=["Unnamed: 0"])
# df.to_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset.csv')
# print(df.info())
# print("start ...........................................")
# '''build_branch'''
# df['branch'] = None
# df['replied_to_tweet_id'] = df['replied_to_tweet_id'].fillna(0)
# df.drop_duplicates(subset=['id'],keep='first',inplace=True)
# df = df.loc[df.text.str.len() > 0]
# df = df.loc[df['id'] != df['replied_to_tweet_id']]
# df = df.sort_values(by = 'create_time')
# df.to_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset_1.csv', index= None)
# print("stage 1: ...........................................")
# print(df.info())
#
#
# convs = df.loc[df['label'].notnull()]
# convs.info()
# conv_ids = convs['conversation_id'].unique()
# final_convs = []
# for conv_id in conv_ids:
#     if len(df.loc[df['conversation_id'] == conv_id]) > 1:
#         final_convs.append(conv_id)
#     if len(df.loc[df['conversation_id'] == conv_id]) > 101:
#         temp_df = df.loc[df['conversation_id'] == conv_id]
#         df.loc[df['conversation_id'] == conv_id]  = temp_df.head(101)
#
# data = df.loc[df['conversation_id'].isin(final_convs)]
# data.index = range(len(data))
# data.info()
# data.to_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset_2.csv', index= None)

data = pd.read_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset_2.csv')
print(data.index)
print("stage 2: ...........................................")
data['branch'] = None
data = get_branch(data)
data.info()
data.to_csv(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset_branch.csv', index= None)

print("stage 3: ...........................................")

tqdm.pandas(desc='apply')
data['embedding'] = data.progress_apply(generate_df_emb,axis=1)
data.to_parquet(r'D:\HKU\MM2023\Codes\dataset\twitter\dataset_emb_branch.parquet', index = None)

