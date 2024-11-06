# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:11:55 2023

@author: jinyu
"""

import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt

import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
# "error", "ignore", "always", "default", "module" or "once"


"""---------------------------------------------------------
Define optimization function (Bayesian)
---------------------------------------------------------"""

#list different solver for different penalty
solvers_l1 = ['liblinear','saga']
solvers_l2 = ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']

def l1_opt(C,solver): #penalty=l1
    params_lr = {}
    
    params_lr['C'] = int(C)
    
    scores = cross_val_score(LogisticRegression(penalty="l1", C=C,
                                                solver=solvers_l1[1 if solver>0.5 else 0],
                                                random_state=1122,max_iter=1000), 
                             train_data, train_label,scoring='recall',cv=5).mean()
    return scores
def l2_opt(C,solver): #penalty=l2
    params_lr = {}
    params_lr['C'] = int(C)
    
    scores = cross_val_score(LogisticRegression(penalty="l2", C=C,
                                                solver=solvers_l2[5 if solver>0.85 
                                                               else (4 if solver>0.68 
                                                                     else (3 if solver>0.51
                                                                           else (2 if solver>0.34
                                                                                 else (1 if solver>0.17
                                                                                       else 0))))],
                                                random_state=1122,max_iter=1000), 
                             train_data, train_label,scoring='recall',cv=5).mean()
    return scores
def elastic_opt(C, l1_ratio): #penalty=elasticnet
    params_lr_elastic = {}
    params_lr_elastic['C'] = int(C)
    params_lr_elastic['l1_ratio'] = int(l1_ratio)
    
    scores = cross_val_score(LogisticRegression(penalty="elasticnet", C=C,
                                                l1_ratio=l1_ratio, solver="saga", 
                                                random_state=1122,max_iter=1000), 
                             train_data, train_label,scoring='roc_auc',cv=5).mean()
    return scores

"""---------------------------------------------------------
Dataset loading
---------------------------------------------------------"""

def type(s):
    it = {b'Non-response':0, b'Response':1}
    return it[s]

basepath = "D:/pathological/machine_learning_gene/response/dataset/selected/"

selection = "knn5"

trainfile = selection + "_training.txt"
testfile = selection + "_test.txt"
idfile = selection + "_test.csv"

saveDir = "D:/pathological/machine_learning_gene/response/result/selected/" + selection + "/"

pathtrain = basepath + trainfile
pathtest = basepath + testfile
pathresult = basepath + idfile     #for reading test patient ids

datatrain = np.loadtxt(pathtrain, dtype=float, delimiter='\t', converters={0:type})
datatest = np.loadtxt(pathtest, dtype=float, delimiter='\t', converters={0:type})

result = pd.read_csv(pathresult, header=0, index_col=0)
test_patientid = result.index.values

train_label_nosmo, train_data_nosmo = np.split(datatrain,indices_or_sections=(1,),axis=1)
smo = SMOTE(random_state=23,k_neighbors=5)
train_data, train_label = smo.fit_resample(train_data_nosmo, train_label_nosmo)
test_label, test_data = np.split(datatest,indices_or_sections=(1,),axis=1)

#check labels
label ={0:"Non-response", 1:"Response"}

current_time = time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime())


"""---------------------------------------------------------
optimize the LR model and do machine learning
---------------------------------------------------------"""

#######define penalty and iter time for optimization
my_penalty = "l1"  #l1 l2 elasticnet
my_iter = 200

########define parameter
params_lr = {'C': (300,450), 'solver': (0,1)}
params_lr_elastic = {'C': (0.1,100),'l1_ratio': (0,1)}

########prepare 5-fold cross validation
my_cv = KFold(n_splits=5, shuffle=True, random_state=233)


print("-----------------------------Start Optimization---------------------------------")

if my_penalty == "l1":
    print("Choosing penalty:", my_penalty)
    lr_bo = BayesianOptimization(l1_opt,params_lr,  random_state=1122,allow_duplicate_points=True)
    lr_bo.maximize(init_points=10, n_iter=my_iter) #n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"C": lr_bo.max["params"]["C"],
                                                "solver": solvers_l1[1 if lr_bo.max["params"]["solver"] > 0.5 else 0]}))
    print("-----------------------------End Optimization---------------------------------")
    
    C = lr_bo.max["params"]["C"]
    solver = solvers_l1[1 if lr_bo.max["params"]["solver"] > 0.5 else 0]
    
    best_lr = LogisticRegression(penalty=my_penalty, C=C, solver = solver, random_state=1122)
    best_lr.fit(train_data, train_label.ravel())

    pred_label = best_lr.predict(test_data)
    pred_label_train = best_lr.predict(train_data)
    

    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:",new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label,pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Test Specificity : ', specificity1 )
    sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Test Sensitivity : ', sensitivity1)
    
    
    

    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label,pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2=sum(sum(cm2))
    accuracy2=(cm2[0,0]+cm2[1,1])/total2
    print ('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])
    print('Trianing pecificity : ', specificity2)
    sensitivity2 = cm2[1,1]/(cm2[1,0]+cm2[1,1])
    print('Trianing Sensitivity : ', sensitivity2)
    
    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))

    #####from decision_function() get roc_curve()
    score_test = best_lr.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr,tpr,threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr,tpr) # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")
    plt.figure(figsize=(4,4),dpi=800)
    out=plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out=plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out=plt.xlim([0.0, 1.0])
    out=plt.ylim([0.0, 1.05])
    out=plt.xlabel('False Positive Rate',fontsize=10)
    out=plt.ylabel('True Positive Rate',fontsize=10)
    out=plt.title('ROC and AUC (Test)',fontsize=12)
    out=plt.legend(loc="lower right")
    plt.show()
    #output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=",auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_lr.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1,tpr1,threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1,tpr1)
    print("trainingAUC=",auc(fpr1, tpr1))


    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    eva_name = ['TrainingAUC','Accuracy','Specificity','Sensitivity','TestAUC','Accuracy','Specificity','Sensitivity']
    eva_value = [auc(fpr1, tpr1),accuracy2,specificity2,sensitivity2,auc(fpr, tpr),accuracy1,specificity1,sensitivity1]
    
    name = np.hstack((test_patientid,eva_name))
    val = np.hstack((new_list,eva_value))
    
    test_label_df = pd.DataFrame({"patient": name,
                                  "pred_label": val})
    
    savefilename_test = saveDir + 'Test_result_l1_' + str(current_time) + '.csv'
    test_label_df.to_csv(savefilename_test)
    
    params_label = ['penalty','C','solver']
    params_bestvalue = [my_penalty, C, solver]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + 'BestParameters_LR_l1_' + str(current_time) + '.csv'
    best_params.to_csv(savefilename_bestparams)
    
    
    print("File saved")

elif my_penalty == "l2":
    print("Choosing penalty:", my_penalty)
    
    lr_bo = BayesianOptimization(l2_opt,params_lr,  random_state=1122)
    lr_bo.maximize(init_points=10, n_iter=my_iter) #n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"C": lr_bo.max["params"]["C"],
                                                "solver": solvers_l2[5 if lr_bo.max["params"]["solver"] > 0.85 
                                                                   else (4 if lr_bo.max["params"]["solver"] >0.68 
                                                                        else (3 if lr_bo.max["params"]["solver"] >0.51
                                                                              else (2 if lr_bo.max["params"]["solver"] >0.34
                                                                                    else (1 if lr_bo.max["params"]["solver"] >0.17
                                                                                          else 0))))]}))
    print("-----------------------------End Optimization---------------------------------")
    
    C = lr_bo.max["params"]["C"]
    solver = solvers_l2[5 if lr_bo.max["params"]["solver"] > 0.85 
                       else (4 if lr_bo.max["params"]["solver"] >0.68 
                            else (3 if lr_bo.max["params"]["solver"] >0.51
                                  else (2 if lr_bo.max["params"]["solver"] >0.34
                                        else (1 if lr_bo.max["params"]["solver"] >0.17
                                              else 0))))]
    
    best_lr = LogisticRegression(penalty=my_penalty, C=C, solver = solver, random_state=1122)
    best_lr.fit(train_data, train_label.ravel())

    pred_label = best_lr.predict(test_data)
    pred_label_train = best_lr.predict(train_data)
    

    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:",new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label,pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Test Specificity : ', specificity1 )
    sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Test Sensitivity : ', sensitivity1)
    


    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label,pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2=sum(sum(cm2))
    accuracy2=(cm2[0,0]+cm2[1,1])/total2
    print ('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])
    print('Trianing pecificity : ', specificity2)
    sensitivity2 = cm2[1,1]/(cm2[1,0]+cm2[1,1])
    print('Trianing Sensitivity : ', sensitivity2)
    
    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))

    #####from decision_function() get roc_curve()
    score_test = best_lr.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr,tpr,threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr,tpr) # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")

    plt.figure(figsize=(4,4),dpi=800)
    out=plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out=plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out=plt.xlim([0.0, 1.0])
    out=plt.ylim([0.0, 1.05])
    out=plt.xlabel('False Positive Rate',fontsize=10)
    out=plt.ylabel('True Positive Rate',fontsize=10)
    out=plt.title('ROC and AUC (Test)',fontsize=12)
    out=plt.legend(loc="lower right")
    plt.show()

    #output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=",auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_lr.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1,tpr1,threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1,tpr1)
    print("trainingAUC=",auc(fpr1, tpr1))


    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    eva_name = ['TrainingAUC','Accuracy','Specificity','Sensitivity','TestAUC','Accuracy','Specificity','Sensitivity']
    eva_value = [auc(fpr1, tpr1),accuracy2,specificity2,sensitivity2,auc(fpr, tpr),accuracy1,specificity1,sensitivity1]
    
    name = np.hstack((test_patientid,eva_name))
    val = np.hstack((new_list,eva_value))
    
    test_label_df = pd.DataFrame({"patient": name,
                                  "pred_label": val})

    savefilename_test = saveDir + 'Test_result_l2_' + str(current_time) + '.csv'
    test_label_df.to_csv(savefilename_test)
    
    params_label = ['penalty','C','solver']
    params_bestvalue = [my_penalty, C, solver]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + 'BestParameters_LR_l2_' + str(current_time) + '.csv'
    best_params.to_csv(savefilename_bestparams)
    
    print("File saved")
    
    
else: 
    print("Choosing penalty:", my_penalty)
    
    lr_bo = BayesianOptimization(elastic_opt,params_lr_elastic,  random_state=1122,allow_duplicate_points=True)
    lr_bo.maximize(init_points=10, n_iter=my_iter) #n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"C": lr_bo.max["params"]["C"],"l1_ratio": lr_bo.max["params"]["l1_ratio"]}))
    print("-----------------------------End Optimization---------------------------------")
    
    C = lr_bo.max["params"]["C"]
    l1_ratio = lr_bo.max["params"]["l1_ratio"]
    solver = "saga"
    
    best_lr = LogisticRegression(penalty=my_penalty, C=C, l1_ratio=l1_ratio, solver = solver, random_state=1122)
    best_lr.fit(train_data, train_label.ravel())

    pred_label = best_lr.predict(test_data)
    pred_label_train = best_lr.predict(train_data)


    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:",new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label,pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Test Specificity : ', specificity1 )
    sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Test Sensitivity : ', sensitivity1)
    


    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label,pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2=sum(sum(cm2))
    accuracy2=(cm2[0,0]+cm2[1,1])/total2
    print ('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])
    print('Trianing pecificity : ', specificity2)
    sensitivity2 = cm2[1,1]/(cm2[1,0]+cm2[1,1])
    print('Trianing Sensitivity : ', sensitivity2)
    
    
    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))


    #####from decision_function() get roc_curve()
    score_test = best_lr.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr,tpr,threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr,tpr) # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")

    plt.figure(figsize=(4,4),dpi=800)
    out=plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out=plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out=plt.xlim([0.0, 1.0])
    out=plt.ylim([0.0, 1.05])
    out=plt.xlabel('False Positive Rate',fontsize=10)
    out=plt.ylabel('True Positive Rate',fontsize=10)
    out=plt.title('ROC and AUC (Test)',fontsize=12)
    out=plt.legend(loc="lower right")
    plt.show()

    #output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=",auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_lr.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1,tpr1,threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1,tpr1)
    print("trainingAUC=",auc(fpr1, tpr1))


    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    eva_name = ['TrainingAUC','Accuracy','Specificity','Sensitivity','TestAUC','Accuracy','Specificity','Sensitivity']
    eva_value = [auc(fpr1, tpr1),accuracy2,specificity2,sensitivity2,auc(fpr, tpr),accuracy1,specificity1,sensitivity1]
    
    name = np.hstack((test_patientid,eva_name))
    val = np.hstack((new_list,eva_value))
    
    test_label_df = pd.DataFrame({"patient": name,
                                  "pred_label": val})

    savefilename_test = saveDir + 'Test_result_elasticnet_' + str(current_time) + '.csv'
    test_label_df.to_csv(savefilename_test)
    
    params_label = ['penalty','C','l1_ratio','solver']
    params_bestvalue = [my_penalty, C, l1_ratio, solver]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + 'BestParameters_LR_elasticnet_' + str(current_time) + '.csv'
    best_params.to_csv(savefilename_bestparams)
    
    print("File saved")
    



