import csv

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

