# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
import argparse
from utils import build_dataset, build_iterator, get_time_dif, set_random_state
from models.Transformer import Encoder, Config

parser = argparse.ArgumentParser(description='Conversational Stance Encoder')

args = parser.parse_args()


if __name__ == '__main__':
    dataset = r'/mnt/MM/Codes/dataset/twitter'  # 数据集
    config = Config(dataset)
    set_random_state(1)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    print("iterator built...")
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    model = Encoder().to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)