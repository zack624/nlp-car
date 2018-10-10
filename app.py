# -*- coding: utf-8 -*-
import sklearn.naive_bayes as nb
from utils import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier


def tfidf_calc(contents, content_test):
    x_train = cut_words(contents)
    x_test = cut_words(content_test)
    x_train.extend(x_test)
    # lr #
    # vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=450, stop_words=load_stop_words(),
    #                              token_pattern=r"(?u)\b[\u4e00-\u9fa5]+\b", min_df=2, max_df=0.95)
    # transform = TfidfTransformer(use_idf=1, smooth_idf=1, sublinear_tf=1)
    # tfidf = transform.fit_transform(vectorizer.fit_transform(x_train))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, lowercase=False,
                          use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfidf = vec.fit_transform(x_train)
    tfidf_data = tfidf.toarray()
    words = vec.get_feature_names()
    return tfidf_data[0:len(contents)], tfidf_data[len(contents):], words


# useless
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
    model = svm.LinearSVC(C=0.1, class_weight='balanced')
    model.fit(x, y)
    return model


def my_logistic_regression(x, y):
    model = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial', C=0.8)
    # model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(x, y)
    model.fit(x, y)
    return model


def my_nn_MLP(x, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(500, 500), random_state=1)
    clf.fit(x, y)
    return clf


def local_test(cid4local_test, model, x_test, y_test):
    print('accuracy: ' + str(model.score(x_test, y_test)))
    print('F1: ' + str(classification_report(y_test, model.predict(x_test), target_names=subject)))


def local_multi_test(cid4local_test, model, x_test, y_test):
    y_true_adjust = adjust_mulit_y(cid4local_test, y_test)
    mul_x, mul_y = multi_label_process(model.predict_proba(x_test).tolist(), cid4local_test, 0.25)
    y_pre_adjust = adjust_mulit_y(mul_x, mul_y)
    mlb = MultiLabelBinarizer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ytrue = mlb.fit_transform(y_true_adjust)
    ypredic = mlb.fit_transform(y_pre_adjust)
    print('accuracy score: ' + str(accuracy_score(ytrue, ypredic)))
    f1 = f1_score(ytrue, ypredic, average=None)
    print(*subject, sep='     ')
    print(*[round(f, 3) for f in f1], sep='    ')
    print("=" * 20)
    print('F1: ' + str(f1_score(ytrue, ypredic, average='macro')))
    print("=" * 20)


if __name__ == "__main__":
    cid_train, content_train, subject_train, sen_train, sen_word_train = load_train()
    cid_test, content_test = load_test()
    # =============================
    x_train, x_idf_test, words = tfidf_calc(content_train, content_test)
    x_train, y_sub_train, y_sen_train, cid4local_test, x4local_test, y4local_sub_test, y4local_sen_test = \
        split4local_test(cid_train, x_train, subject_train, sen_train, .7)
    # normalization #
    normalizer = preprocessing.Normalizer().fit(x_train)
    x_norm_train = normalizer.transform(x_train)
    x_norm_4local_test = normalizer.transform(x4local_test)
    # # ===========================
    # model = my_naive_bayes(x_norm_train, y_sub_train)
    model = my_svm(x_norm_train, y_sub_train)
    # model_lr = my_logistic_regression(x_norm_train, y_sub_train)
    # model = my_nn_MLP(x_norm_train, y_sub_train)
    # local_test(cid4local_test, model, x_norm_4local_test, y4local_sub_test)
    # local_multi_test(cid4local_test, model, x_norm_4local_test, y4local_sub_test)
    print('the length of words: ' + str(len(words)))
    # save_words('words_add_test_lr_12_450.txt', words)
    # =============================
    # test and output submit file #
    x_norm_test = normalizer.transform(x_idf_test)
    y_not_lr = model.predict(x_norm_test)
    output('submit_idf_svc.txt', cid_test, y)
    # =============================
    # y_p_test = model_lr.predict_proba(x_norm_test).tolist()
    # mul_x, mul_y = multi_label_process(y_p_test, cid_test, y_not_lr, 0.25)
    # output('submit_idf20000_svc_lr_multi25.txt', mul_x, mul_y)
    # ==============================
    # with open('output\\probability\\p_mulsub_23.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(y_p_test))
    # ==============================
    print("succeed")
