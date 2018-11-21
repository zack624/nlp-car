import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle as pk
from sklearn import svm

from utils import multi_label_svc_lr,output

path ='datalab/4809/'


def get_data():
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test_public.csv')

    train = train.sample(frac=1)
    train = train.reset_index(drop=True)  # 随机化 1

    data = pd.concat([train, test])

    lbl = LabelEncoder()
    lbl.fit(train['subject'])
    nb_classes = len(list(lbl.classes_))

    pk.dump(lbl, open('label_encoder.sav', 'wb'))

    subject = lbl.transform(train['subject'])

    y = []
    for i in list(train['sentiment_value']):
        y.append(i)

    y1 = []
    for i in subject:
        y1.append(i)

    print(np.array(y).reshape(-1, 1)[:, 0])
    return data, train.shape[0], np.array(y).reshape(-1, 1)[:, 0], test['content_id'], np.array(y1).reshape(-1, 1)[:, 0]

def processing_data(data):
    word = jieba.cut(data)
    return ' '.join(word)


def pre_process():
    data, nrw_train, y, test_id, y1 = get_data()

    data['cut_comment'] = data['content'].map(processing_data)

    print('TfidfVectorizer')
    tf = TfidfVectorizer(ngram_range=(1, 2), analyzer='char')
    # tf = TfidfVectorizer(ngram_range=(1, 2), analyzer='char', min_df=2, max_df=0.95, lowercase=False,
    #                       use_idf=1, smooth_idf=1, sublinear_tf=1) #submit_no1_multi33_forpro_tfapp
    # tf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word',
    #                      token_pattern=r"(?u)\b\w+\b",min_df=2, max_df=0.95, lowercase=False)#submit_no1_multi33_forpro_tfmy
    discuss_tf = tf.fit_transform(data['cut_comment'])

    print('HashingVectorizer')
    ha = HashingVectorizer(ngram_range=(1, 1), lowercase=False)
    discuss_ha = ha.fit_transform(data['cut_comment'])

    data = hstack((discuss_tf, discuss_ha)).tocsr()

    return data[:nrw_train], data[nrw_train:], y, test_id, y1

X,test,y,test_id,y1= pre_process()

N = 10
kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(C=0.5)
clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)

y_train_oofp = np.zeros_like(y, dtype='float64')
y_train_oofp1 = np.zeros_like(y, dtype='float64')

y_test_oofp = np.zeros((test.shape[0], N))
y_test_oofp_1 = np.zeros((test.shape[0], N))

y_test_pro = np.zeros((test.shape[0], N))

def micro_avg_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')


acc = 0
vcc = 0
for i, (train_fold, test_fold) in enumerate(kf):  # 随机化 2
    X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = X[train_fold, :], X[test_fold,
                                                                                                           :], y[
                                                                                             train_fold], y[test_fold], \
                                                                                         y1[train_fold], y1[test_fold]
    clf.fit(X_train, label_train)

    val_ = clf.predict(X_validate) # sentiment
    y_train_oofp[test_fold] = val_
    # print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
    acc += micro_avg_f1(label_validate, val_)
    result = clf.predict(test)
    y_test_oofp[:, i] = result

    clf.fit(X_train, label_1_train) # subject

    val_1 = clf.predict(X_validate)
    y_train_oofp1[test_fold] = val_

    print('topic_value_f1:%f' % micro_avg_f1(label_1_validate, val_1))
    vcc += micro_avg_f1(label_1_validate, val_1)
    result = clf.predict(test)
    y_test_oofp_1[:, i] = result

    # ====================================#
    clf_lr.fit(X_train, label_1_train)
    y_test_pro = y_test_pro + clf_lr.predict_proba(test)

    # ====================================#

print(acc / N)
print(vcc / N)

#====================================#
# clf_lr.fit(X, y1)
# sub_p_test = clf_lr.predict_proba(test)
#====================================#


lbl = pk.load(open('label_encoder.sav','rb'))
res_2 = []
for i in range(y_test_oofp_1.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(int(y_test_oofp_1[i][j]))
    word_counts = Counter(tmp)
    yes = word_counts.most_common(1)
    # res_2.append(lbl.inverse_transform([yes[0][0]])[0])
    res_2.append(yes[0][0])


res = []
for i in range(y_test_oofp.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(y_test_oofp[i][j])
    res.append(int(max(set(tmp), key=tmp.count)))

#====================================#
# mul_x, mul_y_sub, mul_y_sen = multi_label_svc_lr(sub_p_test, test_id.tolist(), res_2, res, p=0.33)
mul_x, mul_y_sub, mul_y_sen = multi_label_svc_lr(y_test_pro/10, test_id.tolist(), res_2, res, p=0.34)
output('submit_no1_multi34_forpro_tfaddapp_2.txt', mul_x, mul_y_sub, sentiment_test=mul_y_sen)
#====================================#


# print(len(res))
# result = pd.DataFrame()
# result['content_id'] = list(mul_x)
#
# result['subject'] = list(mul_y_sub)
# result['subject'] = result['subject']
#
# result['sentiment_value'] = list(mul_y_sen)
# result['sentiment_value'] = result['sentiment_value'].astype(int)
#
# result['sentiment_word'] = ''
# result.to_csv('submit.csv', index=False)