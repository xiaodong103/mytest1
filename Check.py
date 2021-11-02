# cython: language_level=3
# -*- coding: gbk -*-
import copy
import sys
import numpy as np
from minepy import MINE  # (MIC)
from scipy.stats import pearsonr  # (Pearson)
from scipy.spatial.distance import pdist, squareform #Distance correlation
from sklearn.ensemble import RandomForestRegressor  # RF
from sklearn.model_selection import train_test_split  # 分train 和 test

import Input
import Output
import Run

def distcorr(X, Y):
    """ Compute the distance correlation function
    # >>> a = [1,2,3,4,5]
    # >>> b = np.array([1,2,9,4,4])
    # >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def check_VIF(data,limit=10):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    for i in range(len(data[0])):
        a=variance_inflation_factor(data,i)
        if a>limit:
            return False
    return True

def calculate_importance(y,x,Type_check,turnT=True):
    '''
    :param turnT:Automatic transposition
    '''
    correlation_x=[]
    if isinstance(Type_check, dict):
        if Type_check['type']=='RandomForest':
            if turnT:#这里翻转过来
                pass
            else:
                x=x.T#每一列分割
            rf = RandomForestRegressor(random_state=Type_check['random_state'])
            #self.Type_check=self.Type_check['type']+'_'+str(self.Type_check['random_state'])
            rf.fit(x,y)
            importances=rf.feature_importances_
            for i in range(len(importances)):
                correlation_x.append([i,importances[i]])
        else:#类似{'type':'DC','comb_x':[0,8]}已经用'DC'选出来的[0,8]，用于重现结果
            return calculate_importance(y,x,Type_check['type'])

    elif isinstance(Type_check, str):
        if turnT:
            x=x.T#每一列分割
        if isinstance(y[0], np.ndarray):  # 判断是不是列表
            y=y.reshape(-1)
        if Type_check=='MIC':
            m = MINE()
            for i in range(len(x)):
                m.compute_score(y, x[i])
                correlation_x.append([i,m.mic()])
        if Type_check=='DC':
            for i in range(len(x)):
                a=distcorr(y, x[i])
                correlation_x.append([i,a])
        if Type_check=='Pearson':
            for i in range(len(x)):
                a=pearsonr(y, x[i])
                correlation_x.append([i,a[0]])

    correlation_x=np.array(correlation_x)

    correlation_x=list(correlation_x)
    for i in range(len(correlation_x)):
        correlation_x[i]=[i,correlation_x[i][1]]
    return correlation_x


class Sklearn_xd(object):
    def __init__(self, filename,base_dir,Type,par,Standard,Standards=[],Random_State=0,
                 Random_Fen=[1,1],test_size=0.25,pretreat='no',show='on',output='off',par_scores='off'):
        '''
        :param filename:
                    Dictionaries: direct data inheritance
                    String: single file or directory path
                    List: [0] individual files or directory paths [1] files within a directory or individual files with expanded names (.xlsx), the

        :param base_dir: Basic operation directory
        :param Type:
        :param par:Pass as a list, with elements as a list or dictionary
        :param Standard:Model selection criteria
        :param Standards:
        :param Random_State: train and test seeds
        :param test_size:Test set ratio
        :param pretreat:Pre-processing method
                    'no':
                    MinMaxScaler：
                    StandardScaler：
        '''
        self.filename = filename
        self.base_dir = base_dir
        self.Type = Type
        self.par = par
        self.Standard = Standard
        self.Standards = Standards
        try:
            self.Random_State=int(Random_State)
        except:
            self.Random_State=Random_State
        self.Random_Fen=Random_Fen
        self.test_size = test_size
        self.pretreat = pretreat
        self.show = show
        self.output = output
        self.par_scores = par_scores

    def import_data(self):##Added data
        if len(self.filename)==2 :
            if len(self.filename)==2:
                self.data=Input.input_data(self.filename[0], y_num=self.filename[1])# sheet1
            if self.filename[1]==0.5 or self.filename[1]<0:
                self.Random_State='no'
                self.test_size=abs(self.test_size)
            self.filename=self.filename[0]
        self.bei_class=copy.deepcopy(self)

    def output_result(self,base_dir=None,order='',enred=False,sta_num=None):
        data1=copy.deepcopy(self.data)
        if hasattr(self,'comb_x'):
            data1['x_all']=self.data['x_all'][:,self.result['comb_x']]
            if 'y_test' in data1:
                data1['x_test']=self.data['x_test'][:,self.result['comb_x']]
                data1['x_train']=self.data['x_train'][:,self.result['comb_x']]
        if hasattr(self,'comb_x'):
            if hasattr(self,'correlation_x_show'):
                correlation_x_show=self.correlation_x_show
            else:
                correlation_x_show='on'
        else:
            correlation_x_show=[]

        return Output.kp_output(yuan=self.bei_class,filename=self.filename,data=data1,result=self.result,
                                correlation_x_show=correlation_x_show)

class Sklearn_xd_GL(object):
    '''
    :param Type_check (1)'off'Not sorted
                      (2)Pearson
                      (3)MIC
                      (4)DC
                      (5)RandomForest  {'type':'RandomForest','random_state':0}
                      (7)Forward Classical forward feature selection method

    '''
    ##correlation_x，correlation_x_show
    def adjust_set(self):
        if isinstance(self.data, list): #Determine if it is a list
            for ye in range(len(self.data)):
                if self.test_size<0:
                    x_all=copy.deepcopy(self.data[ye]['x_all'])#Each column split
                    y_all=copy.deepcopy(self.data[ye]['y_all'])
                else:
                    if not 'y_test' in self.data:
                        if self.Random_Fen[0]:#Whether to stratify random sampling into large, small, medium and large
                            x_all,tem1,y_all,tem2 = Run.my_StratifiedShuffleSplit(self.data[ye]['x_all'],self.data[ye]['y_all'],
                                                                                  test_size=abs(self.test_size),random_state=self.Random_State)
                        else:
                            x_all,tem1,y_all,tem2 = train_test_split(self.data[ye]['x_all'],self.data[ye]['y_all'],
                                                                     test_size=abs(self.test_size),random_state=self.Random_State)
                correlation_x=calculate_importance(y=y_all,x=x_all,Type_check=self.Type_check)
                for i in range(len(self.data[ye]['x_all'])):
                    self.data[ye]['x_all'][i]=self.data[ye]['x_all'][i]*abs(correlation_x[i][1])
            tem_data={}
            tem_data['y_all']=copy.deepcopy(self.data[0]['y_all'])
            tem_data['x_all']=copy.deepcopy(self.data[0]['x_all'])
            for ye in range(1,len(self.data)):
                tem_data['x_all']=tem_data['x_all']+self.data[ye]['x_all']
            self.data=tem_data
            del tem_data
        else:
            self.adjust()

        if isinstance(self.Type_check, list):
            if self.Type_check[0]=='file':
                tem_data=Input.excel_to_list_dictl(self.Type_check[1])[self.Type_check[2]]
                self.Type_check=tem_data['title']
                correlation_x=[]
                for i in range(len(tem_data['data'])):
                    correlation_x.append([i,tem_data['data'][i]])
                zong=0
                for i in range(len(correlation_x)):
                    zong=zong+abs(correlation_x[i][1])
                for i in range(len(correlation_x)):
                    correlation_x[i][1]=abs(correlation_x[i][1])/zong
                correlation_x=sorted(correlation_x,reverse=True, key=lambda x:abs(x[1]))
                self.correlation_x=correlation_x
            else:
                self.correlation_x=self.Type_check

        elif self.Type_check=='Forward':
             self.split_beg='all'
             self.split_end='all'
             self.VIF='off'
             self.correlation_x=[[_x,0] for _x in range(len(self.data['x_all'][0]))]
        elif self.Type_check=='off':
            self.correlation_x=[[_x,0] for _x in range(len(self.data['x_all'][0]))]
        else:
            #转化为列表
            if self.test_size<0:
                x_all=copy.deepcopy(self.data['x_all'])#Each column split
                y_all=copy.deepcopy(self.data['y_all'])
            else:
                if not 'y_test' in self.data:
                    if self.Random_Fen[0]:##Whether to stratify random sampling into large, small, medium and large
                        x_all,tem1,y_all,tem2 = Run.my_StratifiedShuffleSplit(self.data['x_all'],self.data['y_all'],
                                                                            test_size=abs(self.test_size),random_state=self.Random_State)
                    else:
                        x_all,tem1,y_all,tem2 = train_test_split(self.data['x_all'],self.data['y_all'],
                                                                test_size=abs(self.test_size),random_state=self.Random_State)
                else:
                    x_all=copy.deepcopy(self.data['x_train'])
                    y_all=copy.deepcopy(self.data['y_train'])
            self.correlation_x=calculate_importance(y=y_all,x=x_all,Type_check=self.Type_check)
            self.correlation_x=sorted(self.correlation_x,reverse=True, key=lambda x:abs(x[1]))
            del x_all
            del y_all
        self.correlation_x_show=copy.deepcopy(self.correlation_x)

        if isinstance(self.correlation_x[0],list):
            for i in range(len(self.correlation_x)):
                self.correlation_x[i]=self.correlation_x[i][0]

        #Adjust self.split1_beg and self.split_end
        if isinstance(self.split_end, str): #Determine if it is a string
            if self.split_end=='all':
                self.split_end=len(self.correlation_x)
        elif self.split_end>len(self.correlation_x):
            self.split_end=len(self.correlation_x)

        if isinstance(self.split_beg, str): #判断是不是字符串
            if self.split_beg=='all':
                self.split_beg=len(self.correlation_x)
        elif self.split_beg>len(self.correlation_x):
            self.split_beg=len(self.correlation_x)


    def run_data(self,data,pars,Standard,show='on'):
        if not hasattr(self,'result'):
            self.result = None#Initialization

        if self.VIF!='off':
            if self.test_size<0:
                x_all=copy.deepcopy(self.data['x_all'])#Each column split
                y_all=copy.deepcopy(self.data['y_all'])
            else:
                if not 'y_test' in self.data:

                    if self.Random_Fen[0]:#Whether to stratify random sampling into large, small, medium and large
                        x_all,tem1,y_all,tem2 = Run.my_StratifiedShuffleSplit(self.data['x_all'],self.data['y_all'],
                                                                              test_size=abs(self.test_size),random_state=self.Random_State)
                    else:
                        x_all,tem1,y_all,tem2 = train_test_split(self.data['x_all'],self.data['y_all'],
                                                                 test_size=abs(self.test_size),random_state=self.Random_State)
                else:
                    x_all=copy.deepcopy(self.data['x_train'])#Each column split
                    y_all=copy.deepcopy(self.data['y_train'])

        result_end=None#Initialization
        for par1 in pars:
            result2=None
            result1=None
            comb_x=[]
            for cor_i in range(len(self.correlation_x)):
                comb_x.append(self.correlation_x[cor_i])
                if self.VIF!='off':
                    if len(comb_x)>1:
                        if check_VIF(x_all[:,comb_x],self.VIF):
                            pass
                        else:
                            comb_x.pop()
                            continue
                if len(comb_x)<self.split_beg:
                    continue
                if isinstance(self.split_end, int):
                    if len(comb_x)>self.split_end:
                        break
                self.comb_x=copy.deepcopy(comb_x)
                data1={}
                data1['x_all']=data['x_all'][:,comb_x]
                data1['y_all']=copy.deepcopy(data['y_all'])
                if 'y_test' in data:
                    data1['x_test']=data['x_test'][:,comb_x]
                    data1['y_test']=copy.deepcopy(data['y_test'])
                    data1['x_train']=data['x_train'][:,comb_x]
                    data1['y_train']=copy.deepcopy(data['y_train'])
                if show=='on':
                    result1= copy.deepcopy(self.run_it(data=data1, pars=[par1], Standard=Standard, show='on', call_upd_tqd='on'))
                else:
                    result1= copy.deepcopy(self.run_it(data=data1, pars=[par1], Standard=Standard, show='off'))
                self.result=None##Removal of underlying records
                result1['comb_x']=copy.deepcopy(comb_x)#####
                tem_better=False
                if result2 is None:
                    result2={}
                    result2['score']=abs(result1['score'])*2+1
                if result1['score']<result2['score']:#Note that only the optimal value can be taken here
                    tem_better=True

                if tem_better:
                    result2=copy.deepcopy(result1)
                else:
                    if self.split_end=='yes':
                        if len(comb_x)>1:
                            comb_x.pop()

            if result_end is None:
                result_end={}
                result_end['score']=abs(result2['score'])*2+1
            if result2['score']<result_end['score']:#Note that only the optimal value can be taken here
                result_end=copy.deepcopy(result2)

        self.result=copy.deepcopy(result_end)
        return self.result

if __name__ == '__main__':
    pass



