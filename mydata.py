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
    S=[]
    Y1 = []
    Y2 = []
    Y3 = []
    F=[]
    file = open(filename)
    reader = csv.reader(file) # 逐行读取信息
    headers = next(reader)  #跳过第一行
    for row in reader:
        F.append(row)
    file.close()
    for row1 in F:
        S.append(row1[1])
        S.append(row1[3])
        X.append(S)
        S=[]
        n=0
        # print(pow(serviceNum,2))

        for row2 in F:
            if row1[4]==row2[2]:
                Y1.append([row2[1]])
                Y2.append([row2[1]])
                Y3.append([row2[1]])
                n=1
                break
        if n==0:
            Y1.append(['None'])
            Y2.append(['None'])
            Y3.append(['None'])
    #X_scaled = preprocessing.scale(X)
    return [X, Y1, Y2, Y3]


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
