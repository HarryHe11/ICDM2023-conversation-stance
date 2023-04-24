# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from torch.optim import Adam

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()

    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    batch_per_epoch = len(train_iter)
    output_batch = int(batch_per_epoch / 20)
    print("using device: ", config.device)
    print("epoch: ", config.num_epochs)
    print("batch_per_epoch: ", batch_per_epoch)
    print("output_batch: ", output_batch)
    require_improvement = output_batch * 5
    print("require_improvement: ", require_improvement)
    flag = False  # 记录是否很久没有效果提升
    model.train()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (features, labels) in enumerate(train_iter):

            outputs = model(features)

            y_pred = outputs  # b x len x 2
            y_true = labels  # b x len x 1

            y_pred = torch.flatten(y_pred, end_dim=1)  # (b x len) x2, i.e. 80 x 2
            y_true = torch.flatten(labels, end_dim=1)  # (b x len) x2, i.e. 80 x 1
            y_true = torch.unsqueeze(y_true, 1)
            mask = y_true[:, 0] != 2  # 2 denotes neither class or no labels

            outputs = y_pred[mask, :]
            labels = y_true[mask, :]
            labels = torch.squeeze(labels, 1)

            '''loss computing'''
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels, config.weights)
            loss.backward()
            optimizer.step()

            if total_batch % output_batch == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_f1 = metrics.f1_score(true, predic)
                dev_f1, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train F1: {2:>6.2%},  Val Loss: {3:>5.2},  Val F1: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_f1, dev_loss, dev_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test F1: {1:>6.2%}'
    print(msg.format(test_loss, test_f1))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for features, labels in data_iter:
            outputs = model(features)

            y_pred = outputs  # b x len x 2
            y_true = labels  # b x len x 1

            y_pred = torch.flatten(y_pred, end_dim=1)  # (b x len) x2, i.e. 80 x 2
            y_true = torch.flatten(labels, end_dim=1)  # (b x len) x2, i.e. 80 x 1

            y_true = torch.unsqueeze(y_true, 1)

            mask = y_true[:, 0] != 2  # 2 denotes neither class or no labels
            outputs = y_pred[mask, :]
            labels = y_true[mask, :]
            labels = torch.squeeze(labels, 1)

            loss = F.cross_entropy(outputs, labels, config.weights)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            #print("eval labels: ", labels)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            #print("eval predics: ", predic)

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    f1 = metrics.f1_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return f1, loss_total / len(data_iter), report, confusion
    return f1, loss_total / len(data_iter)