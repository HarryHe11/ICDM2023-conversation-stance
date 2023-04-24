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

model = SentenceTransformer('all-MiniLM-L6-v2')

data = pd.read_csv(r'.\dataset.csv')
tqdm.pandas(desc='apply')
data['embedding'] = data.progress_apply(generate_df_emb,axis=1)
data.to_parquet(r'D.\dataset_emb.parquet', index = None)

