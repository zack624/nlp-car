# -*- coding: utf-8 -*-
import numpy as np
from utils import *
from snownlp import SnowNLP
from sklearn.model_selection import train_test_split
from snownlp import sentiment


def snow_local_test(x_test, y_test):
    neutral = []; neg = []; pos = []
    for i in range(len(x_test)):
        s = SnowNLP(x_test[i])
        if int(y_test[i]) == 0:
            neutral.append(s.sentiments)
        elif int(y_test[i]) == -1:
            neg.append(s.sentiments)
        elif int(y_test[i]) == 1:
            pos.append(s.sentiments)
    print('负面：' + str(neg))
    print('中立：' + str(neutral))
    print('正面：' + str(pos))
    matr = np.ones((len(neutral), 3)) * 2
    matr[0:len(neg), 0] = neg; matr[0:len(neutral), 1] = neutral; matr[0:len(pos), 2] = pos
    np.save('output\\sentiment\\snow_sen1', matr)
    with open('output\\sentiment\\snow_sen1.txt', 'w', encoding='utf-8') as f:
        for line in np.around(matr, decimals=3).tolist():
            f.write(str(line))
            f.write('\n')


def snow_train_disposable(file_post=''):
    sentiment.train(neg_file='data\\neg_' + file_post + '.txt', pos_file='data\\pos_' + file_post + '.txt')
    sentiment.save('data\\sentiment.marshal_' + file_post)


def split_neg_pos(content_train, sentiment_train, post=''):
    neg = []; pos = []
    for i in range(len(sentiment_train)):
        if sentiment_train[i] == '-1':
            neg.append(content_train[i])
        # elif sentiment_train[i] == '1':
        else:
            pos.append(content_train[i])
    with open('data\\neg_' + post + '.txt', 'w', encoding='utf-8') as f:
        for c in neg:
            f.write(c); f.write('\n')
    with open('data\\pos_' + post + '.txt', 'w', encoding='utf-8') as f:
        for c in pos:
            f.write(c); f.write('\n')


def analysis_rate(filename, rate=0.1, compare=-1):
    matr = np.load('output\\sentiment\\snow_sen1.npy')
    neg = neu = pos = 0
    if compare < 0:
        for s in matr.tolist():
            if s[0] < rate:
                neg += 1
            elif s[1] < rate:
                neu += 1
            elif s[2] < rate:
                pos += 1
    else:
        for s in matr.tolist():
            if 2 > s[0] > rate:
                neg += 1
            elif 2 > s[1] > rate:
                neu += 1
            elif 2 > s[2] > rate:
                pos += 1
    neg_count = pos_count = 0
    for s in matr.tolist():
        if s[0] < 2:
            neg_count += 1
        if s[2] < 2:
            pos_count += 1
    print('count: ')
    print('负面：' + str(neg) + ' rate: ' + str(neg/float(neg_count)))
    print('中立：' + str(neu) + ' rate: ' + str(neu/float(np.shape(matr)[0])))
    print('正面：' + str(pos) + ' rate: ' + str(pos/float(pos_count)))
    print('错判中立占比：' + str(neu/float(neg+neu+pos)))
    print('判对负面占比：' + str(neg/float(neg+neu+pos)))
    print('=' * 30)


def sentiment_predict(content_test, rate=0.0005):
    sen_rate = []
    sens = []
    for c in content_test:
        s = SnowNLP(c)
        sen_rate.append(s.sentiments)
    for sen in sen_rate:
        if sen < rate:
            sens.append(-1)
        else:
            sens.append(0)
    return sens


if __name__ == '__main__':
    cid_train, content_train, sub_tr, sentiment_train, sentiment_word_train = load_train()
    x_train, x_test, y_train, y_test = train_test_split(content_train,
                                                        sentiment_train, test_size=0.1)
    # split_neg_pos(content_train, sentiment_train, 'all')
    # snow_train_disposable('all')
    cid_test, content_test = load_test()
    sen_single_test = sentiment_predict(content_test)

    # split_neg_pos(x_train, y_train, 'localtest_1')
    # snow_train_disposable()
    # snow_local_test(x_test, y_test)
    # analysis_rate('output\\sentiment\\snow_sen1.txt', 0.01, compare=-1)
