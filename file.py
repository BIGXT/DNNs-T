import pandas as pd


def readcsv(data_file):
    data = pd.read_csv(data_file)


    data = pd.get_dummies(data, dummy_na=True)
    #print(data)
    return data
