# 导入 `pandas` 库

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
# 使用 `read_csv()` 加载数据集
from sklearn.neural_network import MLPClassifier

import mydata
import trainning

faultnum = 10
testnum = 10

if __name__ == '__main__':
	ordata = []
	# ordata=mydata.pretrainning('training.csv')	#数据读取及标准化
	# X=ordata[0].tolist()
	# Y=ordata[1]
	X1 = []
	X1test=[]
	X2 = []
	X2test=[]
	X3 = []
	X3test=[]
	Y1 = []
	Y1test=[]
	Y2 = []
	Y2test = []
	Y3 = []
	Y3test = []
	AllXTrain=[]
	AllX1Test=[]
	AllX2Test = []
	AllX3Test = []
	AllY1Train=[]
	AllY1Test=[]
	AllY2Train = []
	AllY2Test = []
	AllY3Train = []
	AllY3Test = []

	for i in range(1, faultnum + 1):
		ordata=[]
		data1 = []
		data2 = []
		data3 = []
		data4 = []
		for j in range(1, 4):
			ordata.append(mydata.pretrainning('f' + str(i) + '-' + str(j) + 'training.csv'))
			for x in ordata[j-1][0]:
				data1.append(x)
			for x in ordata[j-1][1]:
				data2.append(x)
			for x in ordata[j-1][2]:
				sn = 0
				sm = 0
				alls = 0
				sx = []
				for xx in x:
					sn = sn + 1
					if xx == '1':
						sm=sn
						alls = alls + 1
				if alls==0:
					data3.append('0')
				elif alls==1:
					data3.append(str(sm))
				else:
					data3.append('34')
			for x in ordata[j-1][3]:
				sn = 0
				sm = 0
				alls = 0
				sx = []
				for xx in x:
					sn = sn + 1
					if xx == '1':
						sm = sn
						alls = alls + 1
				if alls == 0:
					data4.append('0')
				elif alls == 1:
					data4.append(str(sm))


		X_train, X_test, y_train, y_test = train_test_split(data1, data2, test_size=0.1, random_state=0)
		X1.append(X_train)
		X1test.append(X_test)
		Y1.append(y_train)
		Y1test.append(y_test)
		X_train, X_test, y_train, y_test = train_test_split(data1, data3, test_size=0.1, random_state=0)
		X2.append(X_train)
		X2test.append(X_test)
		Y2.append(y_train)
		Y2test.append(y_test)
		X_train, X_test, y_train, y_test = train_test_split(data1, data4, test_size=0.1, random_state=0)
		X3.append(X_train)
		X3test.append(X_test)
		Y3.append(y_train)
		Y3test.append(y_test)

	for x in X1:
		for j in x:
			AllXTrain.append(j)
	for x in Y1:
		for j in x:
			AllY1Train.append(j)
	for x in Y2:
		for j in x:
			AllY2Train.append(j)
	for x in Y3:
		for j in x:
			AllY3Train.append(j)
	for x in X1test:
		for j in x:
			AllX1Test.append(j)
	for x in Y1test:
		for j in x:
			AllY1Test.append(j)
	for x in X2test:
		for j in x:
			AllX2Test.append(j)
	for x in Y2test:
		for j in x:
			AllY2Test.append(j)
	for x in X3test:
		for j in x:
			AllX3Test.append(j)
	for x in Y3test:
		for j in x:
			AllY3Test.append(j)

	#trainning.test(AllXTrain, AllX1Test, AllY1Train, AllY1Test)
	#trainning.SVM(AllXTrain, X1[testnum - 1], AllY1Train, Y1[testnum - 1])
	#trainning.SVM(AllXTrain, AllX1Test, AllY1Train, AllY1Test)
	#trainning.SVM(AllXTrain, X2[testnum - 1], AllY2Train, Y2[testnum - 1])
	#trainning.SVM(AllXTrain, X3[testnum - 1], AllY3Train, Y3[testnum - 1])
	#trainning.SVM(AllXTrain, AllX3Test, AllY3Train, AllY3Test)
	#trainning.MLP(AllXTrain, X1[testnum-1], AllY1Train, Y1[testnum-1])
	trainning.MLP(AllXTrain, AllX1Test, AllY1Train, AllY1Test)
	#trainning.mmlp(AllXTrain, X2[testnum-1], AllY2Train, Y2[testnum-1])
	#trainning.mmlp(AllXTrain, AllX2Test, AllY2Train, AllY2Test)
	#trainning.mmlp(X_train, X_test, y_train, y_test)

	#trainning.mmlp()
