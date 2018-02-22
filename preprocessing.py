import pandas as pd
from sklearn import preprocessing
import numpy as np


def preprocessX(train,flg):
    if flg==0:
        train_x = train.drop(['connection_id', 'target'], axis=1)

    else:
        train_x = train.drop(['connection_id'], axis=1)

    ids = train['connection_id']
    X = np.array(train_x, dtype=np.float64)
    X = preprocessing.scale(X)
    return X,ids

def preprocessY(train):
    train_y = train['target']
    y = []
    for i in train_y:
        trgt = np.zeros(3)
        trgt[i] = 1
        y.append(trgt)
    y = np.array(y)
    return y

def preprocessData(file_name="train_data.csv",flg=0):
    train = pd.read_csv(file_name)
    X,ids= preprocessX(train,flg)
    if flg==0:
        y=preprocessY(train)
        return X,y
    else:
        return X,ids


if __name__=='__main__':
    train_x,train_y=preprocessData()
    test_x,test_ids=preprocessData(file_name="test_data.csv",flg=1)
    print(len(train_x),len(train_y),len(test_x),len(test_ids))
    np.save("train_X.npy",train_x)
    np.save("train_y.npy", train_y)
    np.save("test_X.npy", test_x)
    np.save("ids.npy",test_ids)