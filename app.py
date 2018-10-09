# -*- coding: utf-8 -*-
import sklearn.naive_bayes as nb
from utils import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier


def tfidf_calc(contents, content_test):
    x_train = cut_words(contents)
    x_test = cut_words(content_test)
    x_train.extend(x_test)
    # lr #
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=450, stop_words=load_stop_words(),
                                 token_pattern=r"(?u)\b[\u4e00-\u9fa5]+\b")
    # svc #
    # vectorizer = CountVectorizer(ngram_range=(1, 5), max_features=1000, stop_words=load_stop_words(),
    #                              token_pattern=r"(?u)\b[\u4e00-\u9fa5]+\b")
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vectorizer.fit_transform(x_train))
    tfidf_data = tfidf.toarray().tolist()
    words = vectorizer.get_feature_names()
    return tfidf_data[0:len(contents)], words


def get_x_tfidf(content_test, words):
    x_test = cut_words(content_test)
    vectorizer = CountVectorizer(vocabulary=words)
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vectorizer.fit_transform(x_test))
    tfidf_test = tfidf.toarray().tolist()
    return tfidf_test


# def lsi_calc(tfidf_data, words):
#     lsi_model = models.LsiModel(corpus=tfidf_data, id2word=words, num_topics=10)
#     print(lsi_model.get_topics())
#     # corpus_lsi = [lsi_model[doc] for doc in corpus]
#
#
# def lda_calc(contents, words):
#     x_train = cut_words(contents)
#     vec = CountVectorizer(stop_words=load_stop_words(), vocabulary=words)
#     cntTF = vec.transform(x_train)
#     lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
#                                     learning_method='online',
#                                     learning_offset=50.,
#                                     random_state=0)
#     docres = lda.fit_transform(cntTF).tolist()
#     # print(lda.components_)
#     print(docres)
#     return docres


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


def my_logistic_regression(x, y):
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    # model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(x, y)
    model.fit(x, y)
    return model


def my_nn_MLP(x, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(500, 500), random_state=1)
    clf.fit(x, y)
    return clf


def local_test(model, x_test, y_test):
    print('mean accuracy: ' + str(model.score(x_test, y_test)))
    print('F1: ' + str(classification_report(y_test, model.predict(x_test), target_names=subject)))
    print("=" * 20)


if __name__ == "__main__":
    cid_train, content_train, subject_train, sen_train, sen_word_train = load_train()
    cid_test, content_test = load_test()
    x_train, words = tfidf_calc(content_train, content_test)
    x_train, y_sub_train, y_sen_train, x4local_test, y4local_sub_test, y4local_sen_test = \
        split4local_test(x_train, subject_train, sen_train, .7)
    # normalization #
    normalizer = preprocessing.Normalizer().fit(x_train)
    x_norm_train = normalizer.transform(x_train)
    x_norm_4local_test = normalizer.transform(x4local_test)
    # # ===========================
    # model = my_naive_bayes(x_norm_train, y_sub_train)
    # model = my_svm(x_norm_train, y_sub_train)
    model = my_logistic_regression(x_norm_train, y_sub_train)
    # model = my_nn_MLP(x_norm_train, y_sub_train)
    local_test(model, x_norm_4local_test, y4local_sub_test)
    print('the length of words: ' + str(len(words)))
    save_words('words_add_test_lr_12_450.txt', words)
    # ===========================
    # test and output submit file #
    cid_test, content_test = load_test()
    x_norm_test = normalizer.transform(get_x_tfidf(content_test, words))
    y_test = model.predict(x_norm_test)
    output('submit_add_test_lr_12_450.txt', cid_test, y_test)
    print("succeed")
