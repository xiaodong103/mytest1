# cython: language_level=3
# -*- coding: gbk -*-
import copy
import sys
from sklearn.model_selection import train_test_split  # 分train 和 test
import Run
from Check import Sklearn_xd,Sklearn_xd_GL
import secrets


class Liup(Sklearn_xd):
    def __init__(self,filename,base_dir,Type,par,Standard,Standards,Random_State,Random_Fen,
                 test_size,pretreat,show,output,par_scores):
        super(Liup, self).__init__(filename,base_dir,Type,par,Standard,Standards,Random_State,Random_Fen,test_size,
                                   pretreat,show,output,par_scores)
        self.bei_class=copy.deepcopy(self)

    def adjust(self):
        self.pars=Run.par_adjust(self.par)

    def adjust_set(self):#
        self.adjust()

    def run_it(self, data, pars, Standard, show='on', use_y='on',call_upd_tqd='off'):
        if not hasattr(self,'result'):
            self.result = None#Initialization
        if not 'y_test' in data:
            if self.Random_Fen[0]:#Whether to stratify random sampling into large, small, medium and large
                data['x_train'],data['x_test'],data['y_train'],data['y_test'] = Run.my_StratifiedShuffleSplit(data['x_all'],data['y_all'],
                                                                                                              test_size=abs(self.test_size),random_state=self.Random_State)
            else:
                data['x_train'],data['x_test'],data['y_train'],data['y_test'] = train_test_split(data['x_all'],data['y_all'],
                                                                                                 test_size=abs(self.test_size),random_state=self.Random_State)

        if self.test_size>0:
            split_sym='train'
        else:
            split_sym='all'

        for par1 in pars:
            score1=[]
            for Random_state in range(100):
                Random_state=secrets.randbelow(10000)#Randomly split training and validation sets
                data1={}
                if self.Random_Fen[1]:#Whether to stratify random sampling into large, small, medium and large
                    data1['x_train'],data1['x_test'],data1['y_train'],data1['y_test'] = Run.my_StratifiedShuffleSplit(data['x_'+split_sym],data['y_'+split_sym],
                                                                                                                      test_size=abs(self.test_size),random_state=Random_state)
                else:
                    data1['x_train'],data1['x_test'],data1['y_train'],data1['y_test'] = train_test_split(data['x_'+split_sym],data['y_'+split_sym],
                                                                                                         test_size=abs(self.test_size),random_state=Random_state)
                if use_y=='on':
                    result1= Run.run_one_n(Type=self.Type, par=par1, data=data1, pretreat=self.pretreat,
                                           Standard=Standard, use_y_clf='on')
                else:
                    result1= Run.run_one_n(Type=self.Type, par=par1, data=data1, pretreat=self.pretreat,
                                           Standard=Standard)
                score1.append(result1['score'])
                result1={}#Free memory

            result1['score']=Run.mean(score1)

            if self.result is None:
                self.result={}
                self.result['score']=abs(result1['score'])*2

            if result1['score']<self.result['score']:
                self.result['score']=result1['score']
                self.result['par']=copy.deepcopy(par1)
                self.result['Random_state']=Random_state
                if hasattr(self,'comb_x'):
                    self.result['comb_x']=self.comb_x

            if not hasattr(self,'result_compare'):
                self.result_compare={}
                self.result_compare['score']=result1['score']+1

            if result1['score']<self.result_compare['score']:
                self.result_compare['score']=result1['score']

                if show=='on':
                    if self.Standard in ['R_square']:
                        tem_score=-1*self.result['score']
                    else:
                        tem_score=self.result['score']

                    if hasattr(self,'comb_x'):
                        par_show='\33[1;36m'+'score='+str(round(tem_score,6))+','+self.result['par']+ \
                                 ',comb_x='+str(self.result['comb_x'])+'\33[0m'
                    else:
                        par_show='\33[1;36m'+'score='+str(round(tem_score,6))+','+self.result['par']+'\33[0m'
                    tem=' '*150
                    sys.stdout.write("\r%s"% tem) #Clear a row
                    sys.stdout.write("\r%s\n"% par_show)
                    sys.stdout.flush()
        return self.result


    def run_data(self,data,pars,Standard,show='on',call_upd_tqd='on'):
        self.result=copy.deepcopy(self.run_it(data=data,pars=pars,Standard=Standard,show=show,call_upd_tqd=call_upd_tqd))


class Liup_GL(Sklearn_xd_GL,Liup):#self.result,self.correlation_x_show,self.correlation_x
    def __init__(self,Type_check,VIF,split_beg,split_end,filename,base_dir,Type,par,Standard,Standards=[],Random_State=0,
                 Random_Fen=[1,1],test_size=0.25,pretreat='no',show='on',output='off',
                 par_scores='off'):
        '''
        : param Type_check (1)'off'Not sorted
                      (2)Pearson
                      (3)MIC
                      (4)DC
                      (5)RandomForest  {'type':'RandomForest','random_state':0}
                      (7)Forward Classical forward feature selection method
                        
        :param split_beg: Minimum number of sets of independent variables to start running
        :param split_end：Maximum number of sets of independent variables to end the run
        '''
        self.Type_check=Type_check
        self.VIF=VIF
        self.split_beg=split_beg
        self.split_end=split_end
        super(Liup_GL, self).__init__(filename,base_dir,Type,par,Standard,Standards,Random_State,Random_Fen,
                                      test_size,pretreat,show,output,par_scores)
        self.bei_class=copy.deepcopy(self)


def run_cir_n_liup_GL(filename,Type,Type_check,par,Standard,Standards=[],VIF='off',split_beg=1,split_end=1,Random_State=0,
                    Random_Fen=[1,1],test_size=0.25,pretreat='no',show='on',output='off',
                      par_scores='off',base_dir=None,order=''):

    liup=Liup_GL(filename=filename,base_dir=base_dir,Type=Type,Type_check=Type_check,VIF=VIF,par=par,Standard=Standard,pretreat=pretreat,Standards=Standards,
                split_beg=split_beg,split_end=split_end,Random_State=Random_State,
                Random_Fen=Random_Fen,show=show,output=output,test_size=test_size,par_scores=par_scores)
    liup.import_data()
    liup.adjust_set()
    liup.run_data(data=liup.data, pars=liup.pars, Standard=Standard)
    liup.output_result(base_dir=base_dir,order=order)


#######
if __name__ == '__main__':
    pass
