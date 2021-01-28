import numpy as np
import csv
# import featureList as featureList
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
# from sklearn.externals.six import StringIO
import pandas as pd
from sklearn import model_selection as model

serviceNum = 33


def pretrainning(filename):
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    file = open(filename)
    reader = csv.reader(file)  # 逐行读取信息
    for row in reader:
        X.append(row[1:pow(serviceNum, 2) + 4])
        # print(pow(serviceNum,2))
        Y1.append(row[pow(serviceNum, 2) + 4])
        Y2.append(row[pow(serviceNum, 2) + 5:pow(serviceNum, 2) + 5 + serviceNum])
        Y3.append(row[pow(serviceNum, 2) + 6 + serviceNum:])
    X_scaled = preprocessing.scale(X)
    return [X_scaled, Y1, Y2, Y3]


def trainDicisionTree(csvfileurl):
    # 读取商品信息
    featureList = []
    labelList = []

    # 读取商品信息
    allElectronicsData = open(csvfileurl)
    reader = csv.reader(allElectronicsData)  # 逐行读取信息
    j = 0
    headers = []
    # for i in reader:
    #    j=j+1
    #    headers.append(str(j))  # 读取信息头文件
    # print(headers)

    for row in reader:
        labelList.append(row[len(row) - 1])  # 读取最后一列的目标数据
        rowDict = {}  # 存放特征值的字典
        for i in range(1, len(row) - 1):
            rowDict[i] = row[i]
            # print("rowDict:",rowDict)
        featureList.append(rowDict)

    'Vetorize features:将特征值数值化'
    vec = DictVectorizer()  # 整形数字转化
    dummyX = vec.fit_transform(featureList).toarray()  # 特征值转化是整形数据

    print("dummyX: " + str(dummyX))
    print(vec.get_feature_names())

    print("labelList: " + str(labelList))

    # vectorize class labels
    lb = preprocessing.LabelBinarizer()
    dummyY = lb.fit_transform(labelList)
    print("dummyY: \n" + str(dummyY))

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(dummyX, dummyY)
    print("clf: " + str(clf))

    # Visualize model
    with open("allElectronicInformationGainOri.dot", 'w') as f:
        f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
