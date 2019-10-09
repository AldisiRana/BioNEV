# -*- coding: utf-8 -*-

import numpy
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        print(results)
        return results
        # print('-------------------')

    def predict(self, x, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in x])
        return self.clf.predict(X_, top_k_list=top_k_list)

    def split_train_evaluate(self, x, y, train_precent):
        training_size = int(train_precent * len(x))
        shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
        x_train = [x[shuffle_indices[i]] for i in range(training_size)]
        y_train = [y[shuffle_indices[i]] for i in range(training_size)]
        x_test = [x[shuffle_indices[i]] for i in range(training_size, len(x))]
        y_test = [y[shuffle_indices[i]] for i in range(training_size, len(x))]

        self.train(x_train, y_train, y)
        return self.evaluate(x_test, y_test)


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size + 1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    x = []
    y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        x.append(vec[0])
        y.append(vec[1:])
    fin.close()
    return x, y
