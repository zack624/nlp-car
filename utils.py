# -*- coding: utf-8 -*-
import jieba


subject = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']


def cut_words(contents):
    return [" ".join(jieba.cut(c)) for c in contents]


def load_stop_words():
    fr = open('data\\chineseStopWords.txt', encoding='gbk')
    stop_words = []
    for line in fr.readlines():
        stop_words.append(line.strip())
    return stop_words


# useless
def load_vocabulary():
    with open('output\\words_changed.txt', 'r', encoding='utf-8') as f:
        return eval(f.read())


def load_train():
    fr = open("data\\train.csv", encoding="utf-8")
    fr.readline()
    cid_train = []
    content_train = []
    subject_train = []
    sentiment_train = []
    sentiment_word_train = []
    for line in fr.readlines():
        cur_line = line.strip().split(',')
        cid_train.append(cur_line[0])
        content_train.append(cur_line[1])
        subject_train.append(subject.index(cur_line[2].strip()))
        sentiment_train.append(cur_line[3])
        sentiment_word_train.append(cur_line[4])
    return cid_train, content_train, subject_train, sentiment_train, sentiment_word_train


def load_test():
    fr = open('data\\test_public.csv', encoding='utf-8')
    fr.readline()
    cid_test = []
    content_test = []
    for line in fr.readlines():
        cur_line = line.strip().split(',')
        cid_test.append(cur_line[0])
        content_test.append(cur_line[1])
    return cid_test, content_test


def split4local_test(cid, x, subjects, sentiments, rate=.9):
    import random
    length = len(x)
    random_indies = list(range(length))
    random.shuffle(random_indies)
    x_train = [x[i] for i in random_indies[0:int(rate * length)]]
    y_sub_train = [subjects[i] for i in random_indies[0:int(rate * length)]]
    y_sen_train = [sentiments[i] for i in random_indies[0:int(rate * length)]]
    cid4local_test = [cid[i] for i in random_indies[int(rate * length):]]
    x4local_test = [x[i] for i in random_indies[int(rate * length):]]
    y4local_sub_test = [subjects[i] for i in random_indies[int(rate * length):]]
    y4local_sen_test = [sentiments[i] for i in random_indies[int(rate * length):]]
    return x_train, y_sub_train, y_sen_train, cid4local_test, x4local_test, y4local_sub_test, y4local_sen_test


def output(file_name, cid_test, subject_test, sentiment_test=0):
    fr = open('output\\submit\\' + file_name, 'w', encoding='utf-8')
    fr.write('content_id,subject,sentiment_value,sentiment_word\n')
    for i in range(len(cid_test)):
        fr.write(cid_test[i])
        fr.write(',')
        fr.write(subject[subject_test[i]])
        fr.write(',')
        if sentiment_test == 0:
            fr.write(str(sentiment_test))
        else:
            fr.write(str(sentiment_test[i]))
        fr.write(',')
        fr.write('\n')
    fr.close()


def save_words(file_name, words):
    with open('output\\words\\' + file_name, 'w', encoding='utf-8') as f:
        f.write(str(words))


def adjust_mulit_y(cid, y):
    y_adjust = []
    labels = []
    for i in range(len(cid)):
        if i == (len(cid) - 1):
            labels.append(y[i])
            y_adjust.append(labels)
            break
        if cid[i+1] == cid[i]:
            labels.append(y[i])
        else:
            labels.append(y[i])
            y_adjust.append(labels)
            labels = []
    return y_adjust


def multi_label_process(y_probability, cid_test, p=0.2):
    multi_label_x_test = []
    multi_label_y_test = []
    for i in range(len(cid_test)):
        max_p = 0
        subject = 0
        labels = set()
        for j in range(10):
            if y_probability[i][j] > max_p:
                max_p = y_probability[i][j]
                subject = j
        multi_label_x_test.append(cid_test[i])
        labels.add(subject)
        for j in range(10):
            if j == subject:
                continue
            if y_probability[i][j] > p:
                multi_label_x_test.append(cid_test[i])
                labels.add(j)
        for k in labels:
            multi_label_y_test.append(k)
    return multi_label_x_test, multi_label_y_test


def multi_label_svc_lr(y_probability, cid_test, y_single, p=0.2):
    multi_label_x_test = []
    multi_label_y_test = []
    for i in range(len(cid_test)):
        subject = y_single[i]
        labels = set()
        multi_label_x_test.append(cid_test[i])
        labels.add(subject)
        for j in range(10):
            if j == subject:
                continue
            if y_probability[i][j] > p:
                multi_label_x_test.append(cid_test[i])
                labels.add(j)
        for k in labels:
            multi_label_y_test.append(k)
    return multi_label_x_test, multi_label_y_test
