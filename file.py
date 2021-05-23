import pandas as pd


def readcsv(data_file):
    data = pd.read_csv(data_file)


    data = pd.get_dummies(data, dummy_na=True)
    #print(data)
    return data

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
