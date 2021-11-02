# cython: language_level=3
# -*- coding: gbk -*-
import sys
import numpy as np
from decimal import Decimal
from itertools import product
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor # KNN
from sklearn.neural_network import MLPRegressor#
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor  # Adaboost
from sklearn.ensemble import BaggingRegressor  # Bagging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score  # R square
import copy
import warnings
import Input

warnings.filterwarnings(
    "ignore")  # Ignore warnings

def my_StratifiedShuffleSplit(x, y, test_size, random_state):
    '''
    The overall data was divided into three levels: small, medium and large
    :param x:
    :param y:
    :param test_size:
    :param random_state:
    :return:
    '''
    boundary=int(len(y)/3+0.5)
    y1=y.reshape(-1)
    b=np.floor(np.argsort(np.argsort(y1))/boundary)
    b[np.argmax(y1)]=2
    tem_ss=StratifiedShuffleSplit(n_splits=1,test_size=test_size,train_size=1-test_size,random_state=random_state)
    for train_index, test_index in tem_ss.split(x, b):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return x_train,x_test,y_train,y_test

# 计算列表平均值
def mean(data):
    '''Calculate list average'''
    sum1 = 0
    for i in range(len(data)):
        sum1 = sum1 + data[i]
    return sum1 / len(data)

def par_adjust(pars, split1=','):
    '''Suitable adjustment of parameters:
            Dictionary: subject to adjustment such as{'par':'C','beg':10,'end':1000,'ste':{'length':10,'way':"mul"}}
                    That is, C=10,c=100,c=1000; 'way' has 'add' and 'mul' add step and multiply step two ways
            Example:[['kernel',['"rbf"','"linear"']],
                {'par':'gamma','beg':0.5,'end':1,'ste':{'length':0.5,'way':"add"}},
                {'par':'C','beg':10,'end':100,'ste':{'length':10,'way':"mul"}},]
                ->
                ['kernel="rbf",gamma=0.5,C=10', 'kernel="rbf",gamma=0.5,C=100',
                'kernel="rbf",gamma=1.0,C=10', 'kernel="rbf",gamma=1.0,C=100',
                'kernel="linear",gamma=0.5,C=10', 'kernel="linear",gamma=0.5,C=100',
                'kernel="linear",gamma=1.0,C=10', 'kernel="linear",gamma=1.0,C=100']
    '''
    # par1
    result = []
    if isinstance(pars, str):  # Determine if it is a string
        result.append(pars)
        return result
    if isinstance(pars, list):
        for par in pars:
            if isinstance(par, dict):  # Determine if it is a dictionary
                result0 = []
                result0.append(par['par'])
                result1 = []
                par['beg'] = Decimal(str(par['beg']))
                par['end'] = Decimal(str(par['end']))
                par['ste']['length'] = Decimal(str(par['ste']['length']))
                while par['beg'] <= par['end']:
                    result1.append(par['beg'])  ##Record specific figures
                    if par['ste']['way'] == "add":
                        par['beg'] = par['beg'] + par['ste']['length']
                    if par['ste']['way'] == "mul":
                        par['beg'] = par['beg'] * par['ste']['length']
                result0.append(result1)
                result.append(result0)
    # par2
    par = copy.deepcopy(result)
    result = []
    par_lists = []
    par_titles = []
    for term in par:
        par_lists.append(term[1])
        par_titles.append(term[0])

    for par in product(*par_lists):
        par = list(par)
        par_beg = ''
        for i in range(len(par)):
            par[i] = par_titles[i] + '=' + str(par[i]) + split1
            par_beg = par_beg + par[i]
        par_beg = par_beg[:-1]
        result.append(par_beg)
    return result

# RRMSE
def RRMSE(y_true, y_clf):#(%)
    '''Calculate RRMSE (predicted value, true value), note that the true value cannot sum to 0, note that the true value are positive numbers'''
    sum = 0
    for i in range(len(y_true)):
        sum = sum + y_true[i]
        sum = abs(sum)
    rrmse = pow(mean_squared_error(y_true, y_clf), 0.5) * len(y_true) / sum
    return rrmse


# R_square
def R_square(y_true, y_clf):
    return r2_score(y_true, y_clf)

def Calculate_Standard(Standard, y_true, y_clf, negative='on'):
    '''Calculation of various accuracies
    :param Standard: need to be calculated such as RRMSE
    :param y_true: True Value
    :param y_clf: Estimated Value
    :param negative:The smaller the uniform the better For example, take the R-squared as a negative number,
                    use it when building the model, but not when outputting
    :return:
    '''
    result = None
    if Standard == 'RRMSE':
        result = RRMSE(y_true, y_clf)
    elif Standard == 'R_square':
        result = R_square(y_true, y_clf)

    if isinstance(result, np.ndarray) or isinstance(result, list):
        if len(result) == 1:
            result = whether_list(result)

    if negative == 'on':
        if Standard in ['R_square']:
            result = -1 * result

    return result


def whether_list(data):  # Determine if the list is the first element to be returned
    if isinstance(data, np.ndarray) or isinstance(data, list):
        return data[0]
    else:
        return data


def run_one_n(filename='', Type='', par='', data='no', pretreat='no', test_size=0.25, Random_Fen=[1, 1],Random_State='no', Standard='no',
              use_y_clf='off', use_y_test='off', save_clf='off'):
    '''
    filename:File name and its path
    pretreat：
            'no':No treatment
            MinMaxScaler
            StandardScaler
    Type：SVR,KNeighborsRegressor,RandomForestRegressor,MLPRegressor
    Random_state:
            'no':Divided into training and test groups, the training group in the first table of the document,
            the test group in the second table of the document
            Random_state=5  Randomly seeded into a 5-point training and test group
    Random_Fen:#Whether to stratify random sampling into large, small, medium and large
    test_size：Proportion of test groups to the total when grouped
    par：'kernel="rbf",gamma=0.05,C=100'即为kernel="rbf",gamma=0.05,C=100
    '''
    # 导入或继承数据
    if data == 'no':
        if isinstance(filename, dict):  # Determine if it is a dictionary
            data = copy.deepcopy(filename)
        elif isinstance(filename, list):
            data = Input.input_data(filename[0], Type=filename[1], y_num=filename[2])  # sheet1
            filename = filename[0]
        elif filename[-5:] == '.xlsx':
            data = Input.input_data(filename)  # sheet1
    ################
    if pretreat != 'no':
        if pretreat == 'StandardScaler':
            ss = StandardScaler()
        data['x_train'] = ss.fit_transform(data['x_train'])
        data['x_test'] = ss.transform(data['x_test'])
    # Run
    clf = eval(Type + '(' + par + ')')  # Modeling according to type and its parameters
    clf.fit(data['x_train'], data['y_train'])
    y_clf = clf.predict(data['x_test'])

    # Output
    result = {}
    if Standard != 'no':
        result['score'] = Calculate_Standard(Standard, data['y_test'], y_clf)
    if use_y_clf == 'on':
        if isinstance(y_clf[0], np.ndarray):  # Determine if it is a list
            y_clf=y_clf.reshape(-1)
        result['y_clf'] = copy.deepcopy(y_clf)
    if use_y_test == 'on':
        result['y_test'] = copy.deepcopy(data['y_test'])
    if save_clf == 'on':
        result['clf'] = clf
        if pretreat != 'no':
            result['ss'] = ss
    return result
if __name__ == '__main__':
    pass
