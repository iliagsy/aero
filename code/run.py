# coding=utf-8
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import os
import nltk
import collections

from consts import nClass

from logging import getLogger

logger = getLogger('__main__')
path = '../Data/'


def get_trainX(ldataf, dim=100):
    # dataf: data file path
    # dim: SVD分解的维数
    dictionary = collections.OrderedDict()
    i = 0
    row = []
    col = []
    data = []
    doc_num = 0
    test_index = None
    for dataf in ldataf:
        with open(dataf) as fh:
            for line in fh.readlines():
                line = line.split('~')[1]
                fredist = nltk.FreqDist(line.lower().strip().split())
                for localkey in fredist:
                    if localkey == '_':
                        continue
                    if localkey not in dictionary.keys():
                        dictionary[localkey] = fredist[localkey]
                    # numpy sparse matrix
                    row.append(i)
                    col.append(list(dictionary.keys()).index(localkey))
                    data.append(fredist[localkey])
                i += 1
            if test_index is None: test_index = i
    doc_num = i
    logger.warning("row:{},col:{}".format(i, len(dictionary)))

    arr = sp.coo_matrix((data, (row, col)), shape=(doc_num, len(dictionary)))
    u, s, vt = sl.svds(arr.asfptype(), k=dim)
    logger.warning("svd finish")
    return (u, doc_num, test_index)


def get_trainY(labelf):
    # labelf: label file name
    y = np.loadtxt(labelf, delimiter=',')
    y[y == -1] = 0
    return y


if __name__ == '__main__':
    u, doc_num, test_index = get_trainX([path + 'TrainingData.txt',
                                         path + 'TestData.txt'])
    u, test_u = u[:test_index], u[test_index:]
    y, test_y = (get_trainY(path + 'TrainCategoryMatrix.csv'),
                 get_trainY(path + 'TestTruth.csv'))

    # 留一法
    for tt in range(10):
        # prepare index of data
        test_index = np.array([i for i in range(doc_num) if i % 10 == tt])
        train_index = np.array([i for i in range(doc_num) if i % 10 != tt])

        train_x = u[train_index]
        train_y = y[train_index]

        test_x = u[test_index]
        test_y = y[test_index]

        nn = NN([dim, dim, 9])
        nn.train(train_x, train_y, [2000000, 50])
        res = nn.test(test_x)
        i = 0
        scr = 0
        for line in res:
            predict = line == 1
            real = test_y[i] == 1
            scr += 1 - sum(np.logical_xor(predict, real)) / float(nClass)
            print("predict:{}, real:{}".format(line, test_y[i]))
            i += 1
        logger.warning("accuracy:{}".format(scr/i))
