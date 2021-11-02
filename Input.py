# cython: language_level=3
# -*- coding: gbk -*-
import numpy as np
import xlrd
def read_excel(name: str, sheet_num: int=0):
    '''
    :param name: The path to the .xlsx file read and its filename
    :param sheet_num:
    '''
    workbook = xlrd.open_workbook(name)  # Open workbook
    data_sheet = workbook.sheets()[sheet_num]
    rowNum = data_sheet.nrows
    # Get the contents of all cells
    datalist = []
    for i in range(rowNum):
        datalist.append(data_sheet.row_values(i))
    return datalist


def input_data(filename, Type='xlsx', y_num=0,null_value=0,ye=0):
    '''
    :param filename:
    :param Type:
    :return:
    '''
    y_num=0
    data_train =read_excel(filename, ye)  # Read the first table
    for i in range(len(data_train)):
        for j in range(len(data_train[0])):
            if not data_train[i][j]:
                data_train[i][j]=null_value
    x_train = []
    y_train = []
    for i in range(len(data_train)):
        if i > 0:
            for j in range(len(data_train[i])):
                if j == y_num:
                    try:
                        y_train.append(float(data_train[i][j]))
                    except:
                        y_train.append(data_train[i][j])
                else:
                    try:
                        x_train.append(float(data_train[i][j]))
                    except:
                        x_train.append(data_train[i][j])
    x_train = np.array(x_train)
    x_train = x_train.reshape(-1, len(data_train[0])-1)
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1, 1)

    data_test = read_excel(filename, ye + 1)  # Read the second table
    for i in range(len(data_test)):
        for j in range(len(data_test[0])):
            if not data_test[i][j]:
                data_test[i][j]=null_value
    x_test = []
    y_test = []
    for i in range(len(data_test)):
        if i > 0:
            for j in range(len(data_test[i])):
                if j == y_num:
                    try:
                        y_test.append(float(data_test[i][j]))
                    except:
                        y_test.append(data_test[i][j])
                else:
                    try:
                        x_test.append(float(data_test[i][j]))
                    except:
                        x_test.append(data_test[i][j])
    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, len(data_test[0])-1)
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1, 1)
    x_all=np.concatenate((x_train, x_test), axis=0)
    y_all=np.concatenate((y_train, y_test), axis=0)
    #导出
    result={}
    result['x_all']=x_all
    result['y_all']=y_all
    result['x_train']=x_train
    result['y_train']=y_train
    result['x_test']=x_test
    result['y_test']=y_test
    return result

if __name__ == '__main__':
    pass
