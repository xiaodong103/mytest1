# mytest1
A combined strategy of improved variable selection and ensemble algorithm
The algorithms involved in the paper were implemented in Python (3.7.2). The main python third-party libraries used include: 
Numpy(1.17.3) , Scipy(1.3.1), Joblib(0.14.0) , scikit-learn(0.21.3) and Sklearn(0.0) 


This is stripped from the system code, and all many parameters are not called to. The main use is the Check_P.run_cir_n_liup_GL function. 
filename, Type, Type_check, par parameters are the four parameters that we need to adjust according to the data of the actual operation.

Implementation of variable selection criteria (mainly adjusting filename, Type, Type_check, par parameters):
Check_P.run_cir_n_liup_GL(filename=[r'E:\study\raw_data.xlsx',0.5],pretreat='StandardScaler',Type='SVR',
                          Standard='RRMSE',Standards=['RRMSE','R_square'],
                          Random_State=0,Random_Fen=[1,1],
                          test_size=0.3333,output='all',
                          Type_check='DC',
                          split_beg=1,split_end='yes',VIF=10,
                          par=[['kernel',["'rbf'"]],
	         {'par':'gamma','beg':0.1,'end':1,'ste':{'length':0.1,'way':'add'}},
                         {'par':'C','beg':10,'end':100000,'ste':{'length':10,'way':'mul'}},])

The results of the base learner are reproduced in the following way (mainly adjusting the filename, type, and par parameters):
Check_P.run_cir_n_liup_GL(filename=[r'E:\study\single learner16\2021-05-28_15-04-38.399131\Fitted_data_re.xlsx',0.5],pretreat='StandardScaler',Type='SVR',
                      Standard='RRMSE',Standards=['RRMSE','R_square'],
                      Random_State=0,Random_Fen=[1,1],
                      test_size=0.3333,output='all',
                      Type_check='off',
                      split_beg='all',split_end='all',
                    par=[['kernel',["'rbf'"]],
                     {'par':'gamma','beg':0.4,'end':0.4,'ste':{'length':0.1,'way':'add'}},
                      {'par':'C','beg':1000,'end':1000,'ste':{'length':10,'way':'mul'}},])


The way the results of the first ensemble model are reproduced (mainly adjusting the filename, type, par parameters):
Check_P.run_cir_n_liup_GL(filename=[r'E:\study\first ensemble32\RF\141_2021-05-29_23-39-14.403410\Fitted_data_re.xlsx',0.5],
                  pretreat='StandardScaler',Type='BaggingRegressor',
                  Standard='RRMSE',Standards=['RRMSE','R_square'],
                  Random_State=0,Random_Fen=[1,1],
                  test_size=0.3333,output='all',
                  Type_check='off',
                  split_beg='all',split_end='all',
                par='base_estimator=SVR(kernel="rbf",gamma=0.9,C=100),n_estimators=10,random_state=45')
