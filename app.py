# -*- coding: utf-8 -*-
from utils import *
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_calc(contents):
    x_train = cut_words(contents)
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=95, stop_words=load_stop_words(),
                                 token_pattern=r"(?u)\b[\u4e00-\u9fa5]+\b")
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vectorizer.fit_transform(x_train))
    tfidf_data = tfidf.toarray().tolist()
    words = vectorizer.get_feature_names()
    return tfidf_data, words


def get_x_tfidf(subject_test, words):
    x_test = cut_words(subject_test)
    vectorizer = CountVectorizer(vocabulary=words)
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vectorizer.fit_transform(x_test))
    tfidf_test = tfidf.toarray().tolist()
    return tfidf_test


def my_naive_bayes(x, y):
    # model = nb.BernoulliNB()
    # model = nb.MultinomialNB()
    model = nb.ComplementNB()
    model.fit(x, y)
    return model


def my_svm(x, y):
    # model = svm.SVC(gamma='scale', decision_function_shape='ovo')
    # model = svm.SVR()
    model = svm.LinearSVC()
    model.fit(x, y)
    return model


def local_test(model, x_test, y_test):
    print('mean accuracy: ' + str(model.score(x_test, y_test)))
    print('F1: ' + str(classification_report(y_test, model.predict(x_test))))
    print("=" * 20)


if __name__ == "__main__":
    cid_train, content_train, subject_train, sen_train, sen_word_train = load_train()
    x_train, words = tfidf_calc(content_train)
    x_train, y_sub_train, y_sen_train, x4local_test, y4local_sub_test, y4local_sen_test = \
        split4local_test(x_train, subject_train, sen_train, .7)
    # model = my_naive_bayes(x_train, y_sub_train)
    model = my_svm(x_train, y_sub_train)
    local_test(model, x4local_test, y4local_sub_test)
    print('the length of words: ' + str(len(words)))
    # save_words('words_.txt', words)
    ##############################
    # cid_test, content_test = load_test()
    # x_test = get_x_tfidf(content_test, words)
    # y_test = model.predict(x_test)
    # output('submit_.txt', cid_test, y_test)
    print("succeed")
