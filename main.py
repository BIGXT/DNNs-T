# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import csv
import training
from sklearn.model_selection import train_test_split



def TxtToCsv(rn,fn):
    import csv
    for i in range(1,rn+1):
        for j in range(1, fn+1):
            inputfile = 'r'+str(i)+'-'+str(j)+'fault.txt'
            outputfile = 'r'+str(i)+'-'+str(j)+'fault.csv'
            row = ['traceID', 'callerServiceName', 'callerSpanID', 'calledServiceName', 'calledSpanID', 'duration',
                   'TOF']
            csvfile = open(outputfile, 'w', newline='')
            writer = csv.writer(csvfile)
            writer.writerow(row)
            lines = open(inputfile, 'r', encoding='utf-8').readlines()
            for line in lines:
                csvfile.write(line)
            csvfile.close()

def CsvToTraining(faultnum):
    import mydata
    data1 = None
    # ordata=mydata.pretrainning('training.csv')	#数据读取及标准化
    # X=ordata[0].tolist()
    # Y=ordata[1]

    for i in range(1, faultnum + 1):
        ordata = []
        for j in range(1, 4):
            ordata.append(mydata.pretrainning('r' + str(i) + '-' + str(j) + 'fault.csv'))

            import csv
            f1 = open('X.csv', 'a', newline='')
            csv_writer1 = csv.writer(f1)
            f2 = open('Y.csv', 'a', newline='')
            csv_writer2 = csv.writer(f2)
            for p in ordata[j - 1][0]:
                csv_writer1.writerow(p)
            for q in ordata[j - 1][1]:
                csv_writer2.writerow(q)
            f1.close()
            f2.close()







# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    #TxtToCsv(10,3)  #数据文件格式转换
    #CsvToTraining(10)   #数据集
    X=[]
    Y=[]
    file = open('X.csv')
    reader = csv.reader(file) # 逐行读取信息
    for row in reader:
        X.append(row)
    file.close()
    file = open('Y.csv')
    reader = csv.reader(file) # 逐行读取信息
    for row in reader:
        Y.append(row)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    #training.l1(X_train, X_test, y_train, y_test)
    training.l2()



