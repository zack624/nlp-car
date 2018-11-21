# -*- coding: utf-8 -*-

import operator
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from pylab import mpl

def load_train():
    fr = open("data\\train.csv", encoding="utf-8")
    fr.readline()
    train_set = []
    for line in fr.readlines():
        train_set.append(line.strip().split(','))
    return train_set


def load_submit(submit_name):
    fr = open('output\\NO1\\' + submit_name, encoding='utf-8')
    fr.readline()
    submit_set = []
    for line in fr.readlines():
        submit_set.append(line.strip().split(','))
    return submit_set


# statistic distribution of data set before processing
'''
[('动力', 2731), ('价格', 1272), ('油耗', 1081), ('操控', 1035), ('舒适性', 930), ('配置', 852), ('安全性', 572), ('内饰', 535), ('外观', 488), ('空间', 441)]
[('0', 6660), ('1', 1669), ('-1', 1615)]
'''
def statistic(data, i):
    import operator
    subjects = {}
    sentiments = {}
    for d in data:
        if d[i] not in subjects:
            subjects[d[i]] = 0
        else:
            subjects[d[i]] += 1
        if d[i+1] not in sentiments:
            sentiments[d[i+1]] = 0
        else:
            sentiments[d[i+1]] += 1
    subs = sorted(subjects.items(), key=operator.itemgetter(1), reverse=True)
    sens = sorted(sentiments.items(), key=operator.itemgetter(1), reverse=True)
    sub_d = {}
    sen_d = {}
    for s in subs:
        sub_d[s[0]] = s[1]/float(len(data))
    for s in sens:
        sen_d[s[0]] = s[1]/float(len(data))
    # len(set([d[0] for d in data]))
    print(sub_d); print(sen_d)
    return sub_d, sen_d


def statistic_plot(sets, i):
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    x=[];y=[]
    if (i==1):
        x = [i for i in sets]
        y = [sets[i] for i in sets]
        plt.bar(x, y, 0.4, color="green")
        plt.xlabel("topic")
        plt.ylabel("ratio")
        plt.title("Topic distribution chart")
    if (i==2):
        x = ["中立","正面","负面"]
        y = [sets[i] for i in sets]
        plt.bar(x, y, 0.2, color="green")
        plt.xlabel("sentiment")
        plt.ylabel("ratio")
        plt.title("Sentiment distribution chart")
    plt.show()
    pass

'''
cid: 9947
unique cid: 8290
[('PKeUvILOs6M3pYBl', 7), ('HBcOg49avix7bfeF', 6), ('vqaIrjURK0c7me1i', 6), ('nJbR1iAFjQPma928', 6), ('bOcxUzdVkeWJDE9I', 5), ('pysamriCg2B3xTIo', 5), ('PflHOdAUnXIjo2WY', 5), ('BaFdXcWgAOQCNimo', 5), 
'''
def statistic_multi_subject(data):
    cid = {}
    for d in data:
        if d[0] not in cid:
            cid[d[0]] = 1
        else:
            cid[d[0]] += 1
    cid = sorted(cid.items(), key=operator.itemgetter(1), reverse=True)
    print('cid: ' + str(len(data)))
    print('unique cid: ' + str(len(cid)))
    print('duplication: ' + str(len(data)-len(cid)))
    print(cid)

    # seg_list = jieba.cut("因为森林人即将换代，这套系统没必要装在一款即将换代的车型上，因为肯定会影响价格。")


def jieba_test():
    seg_list = jieba.cut("唉，这货的价格死硬死硬的，低配版优惠1万据说已经罕有了。")
    print("/".join(seg_list))


def tfidf_test():
    corpus = [" ".join(["我", "的", "也是", "斯巴鲁", "你", "别", "hello"]),
              " ".join(["请教一下", "变速箱油", "差速器", ",", "。"])]
    vectorizer = CountVectorizer()
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vectorizer.fit_transform(corpus))
    # tfidf = tfidf.toarray()
    words = vectorizer.get_feature_names()
    print(tfidf)
    print(words)
    # for ti in tfidf:
    #     print(ti)


def snownlp_test():
    text0 = u'因为森林人即将换代，这套系统没必要装在一款即将换代的车型上，因为肯定会影响价格。'
    textn = u'四驱价格貌似挺高的，高的可以看齐XC60了，看实车前脸有点违和感。不过大众的车应该不会差。'
    textn2 = u'这玩意都是给有钱任性又不懂车的土豪用的，这价格换一次我妹夫EP020可以换三锅了'
    textp = u'优惠可以了！购进吧！买了不会后悔的！时间可鉴！   '
    s = SnowNLP(text0)
    print('中立:')
    print(s.sentiments)
    print(s.keywords(5))
    print('=' * 2)
    print('负面:')
    sn = SnowNLP(textn)
    sn2 = SnowNLP(textn2)
    print(sn.sentiments)
    print(sn.keywords(5))
    print(sn2.sentiments)
    print('=' * 2)
    print('正面:')
    sp = SnowNLP(textp)
    print(sp.sentiments)
    print(sp.keywords(5))


if __name__ == "__main__":
    # snownlp_test()
    subs, sens = statistic(load_train(), 2)
    # statistic_plot(subs,1)
    # statistic_plot(sens,2)
    print('=' * 20)
    jieba_test()
    # tfidf_test()
    statistic_multi_subject(load_train())
    # submit_subs, su_sens = statistic(load_submit('submit_no1_multi30_forpro_tfapp.txt'), 1)
    # submit_subs, su_sens = statistic(load_submit('submit_no1_multi33_forpro_2.txt'), 1)
    print('=' * 20)
    # submit_subs, su_sens = statistic(load_submit('submit_no1_multi33_forpro_tfaddapp.txt'), 1)
    # submit_subs, su_sens = statistic(load_submit('submit_no1_multi34_forpro_tfaddapp_2.txt'), 1)
    print('=' * 20)
    # test_public: 2364
    statistic_multi_subject(load_submit('submit_no1_multi30_forpro_tfapp.txt'))
    # statistic_multi_subject(load_submit('submit_no1_multi33_forpro_2.txt'))
    # statistic_multi_subject(load_submit('submit_no1_multi33_forpro_tfaddapp.txt'))
    # statistic_multi_subject(load_submit('submit_no1_multi34_forpro_tfaddapp_2.txt'))
