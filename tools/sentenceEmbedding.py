import pandas as pd
from tqdm import tqdm
from simcse import SimCSE
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
    embedding = np.array(embedding)
    return embedding

file_list = ['createdebate', 'twitter_test', 'twitter_train']
tqdm.pandas(desc='apply')
for file in file_list:
    path = 'dataset\data_for_encode\\' + file +'.parquet'
    print("getting: ", path)
    data = pd.read_parquet(path)
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    data['embeddingCSE'] = None
    data['embeddingCSE'] = data.progress_apply(generate_df_emb,axis=1)
    data.to_parquet( path + file +'_CSE.parquet', index = None)
