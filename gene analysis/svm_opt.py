"""
Created by JINYU
2023/09/11
"""

import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt

import time

from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
# "error", "ignore", "always", "default", "module" or "once"

"""---------------------------------------------------------
Dataset loading
---------------------------------------------------------"""

def type(s):
    it = {b'Non-response':0, b'Response':1}
    return it[s]

basepath = "D:/pathological/machine_learning_gene/response/dataset/selected/"

selection = "missf"

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
smo = SMOTE(random_state=23,k_neighbors=3)
train_data, train_label = smo.fit_resample(train_data_nosmo, train_label_nosmo)
test_label, test_data = np.split(datatest,indices_or_sections=(1,),axis=1)

#check labels
label ={0:"Non-response", 1:"Response"}

current_time = time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime())


"""---------------------------------------------------------
Define optimization function (Bayesian)
---------------------------------------------------------"""

degrees = [1,2,3,4,5]

def rbf_opt(c, gamma):   #also for "sigmoid"
    params_svm_linear = {}

    params_svm_linear['c'] = int(c)
    params_svm_linear['gamma'] = int(gamma)

    scores = cross_val_score(svm.SVC(kernel="rbf", C=c, gamma=gamma, decision_function_shape='ovo',probability=True, random_state=1122),
                             train_data, train_label, scoring='roc_auc', cv=5).mean()
    return scores

def linear_opt(c):
    params_svm = {}

    params_svm['c'] = int(c)

    scores = cross_val_score(svm.SVC(kernel="linear", C=c, decision_function_shape='ovo',probability=True, random_state=1122),
                                     train_data, train_label, scoring='roc_auc', cv=5).mean()
    return scores

def poly_opt(c, gamma, degree):
    params_svm_poly = {}

    params_svm_poly['c'] = int(c)
    params_svm_poly['gamma'] = int(gamma)
    params_svm_poly['degree'] = int(degree*10)

    scores = cross_val_score(svm.SVC(kernel="poly", C=c, gamma=gamma,                                      
                                     degree=degrees[5 if degree>0.85 
                                                    else (4 if degree>0.68 
                                                          else (3 if degree>0.51
                                                                else (2 if degree>0.34
                                                                      else (1 if degree>0.17
                                                                            else 0))))],                                     
                                     decision_function_shape='ovo',probability=True, random_state=1122),
                                     train_data, train_label, scoring='accuracy', cv=5).mean()
    return scores


"""---------------------------------------------------------
optimize the SVM and do machine learning
---------------------------------------------------------"""

#######define kernel and iter time for optimization
my_kernel = "rbf"
my_iter = 200

########define parameter
params_svm_linear = {'c': (0.001, 500)}
params_svm = {'c': (0.1, 100), 'gamma': (0.001, 10)}  #rbf and sigmoid
params_svm_poly = {'c': (0.1, 100), 'gamma': (0.01, 10), 'degree':(0.1,0.3)}


########prepare 5-fold cross validation
my_cv = KFold(n_splits=5, shuffle=True, random_state=233)

print("-----------------------------Start Optimization---------------------------------")

if my_kernel == "linear":
    print("Choosing kernel:", my_kernel)
    svm_bo = BayesianOptimization(linear_opt, params_svm_linear, random_state=1122)
    svm_bo.maximize(init_points=10, n_iter=my_iter)  # n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"C": svm_bo.max["params"]["c"]}))
    print("-----------------------------End Optimization---------------------------------")

    C = svm_bo.max["params"]["c"]

    best_svm = svm.SVC(kernel=my_kernel, C=C, random_state=1122, decision_function_shape='ovo', probability=True)
    best_svm.fit(train_data, train_label.ravel())

    pred_label = best_svm.predict(test_data)
    pred_label_train = best_svm.predict(train_data)

    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:", new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label, pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Test Specificity : ', specificity1)
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Test Sensitivity : ', sensitivity1)

    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label, pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2 = sum(sum(cm2))
    accuracy2 = (cm2[0, 0] + cm2[1, 1]) / total2
    print('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0, 0] / (cm2[0, 0] + cm2[0, 1])
    print('Trianing pecificity : ', specificity2)
    sensitivity2 = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])
    print('Trianing Sensitivity : ', sensitivity2)

    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))

    #####from decision_function() get roc_curve()
    score_test = best_svm.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr, tpr, threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr, tpr)  # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")

    plt.figure(figsize=(4, 4),dpi=800)
    out = plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out = plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out = plt.xlim([0.0, 1.0])
    out = plt.ylim([0.0, 1.05])
    out = plt.xlabel('False Positive Rate', fontsize=10)
    out = plt.ylabel('True Positive Rate', fontsize=10)
    out = plt.title('ROC and AUC (Test)', fontsize=12)
    out = plt.legend(loc="lower right")
    plt.show()

    # output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=", auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_svm.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1, tpr1, threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1, tpr1)
    print("trainingAUC=", auc(fpr1, tpr1))

    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    eva_name = ['TrainingAUC','Accuracy','Specificity','Sensitivity','TestAUC','Accuracy','Specificity','Sensitivity']
    eva_value = [auc(fpr1, tpr1),accuracy2,specificity2,sensitivity2,auc(fpr, tpr),accuracy1,specificity1,sensitivity1]
    
    name = np.hstack((test_patientid,eva_name))
    val = np.hstack((new_list,eva_value))
    
    test_label_df = pd.DataFrame({"patient": name,
                                  "pred_label": val})

    savefilename_test = saveDir + my_kernel + '_Test_result_' + str(current_time) + '.csv'
    test_label_df.to_csv(savefilename_test)

    params_label = ['kernel', 'C']
    params_bestvalue = [my_kernel, C]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + my_kernel + 'BestParameters_SVM_' + str(current_time) + '.csv'
    best_params.to_csv(savefilename_bestparams)

    print("File saved")

elif my_kernel == "rbf" or my_kernel == "sigmoid":
    
    print("Choosing kernel:", my_kernel)
    svm_bo = BayesianOptimization(rbf_opt, params_svm, random_state=1122)
    svm_bo.maximize(init_points=10, n_iter=my_iter)  # n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"c": svm_bo.max["params"]["c"], "gamma":svm_bo.max["params"]["gamma"]}))
    print("-----------------------------End Optimization---------------------------------")

    C = svm_bo.max["params"]["c"]
    gamma = svm_bo.max["params"]["gamma"]

    best_svm = svm.SVC(kernel=my_kernel, C=C, gamma=gamma, random_state=1122, decision_function_shape='ovo', probability=True)
    best_svm.fit(train_data, train_label.ravel())

    pred_label = best_svm.predict(test_data)
    pred_label_train = best_svm.predict(train_data)

    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:", new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label, pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Test Specificity : ', specificity1)
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Test Sensitivity : ', sensitivity1)

    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label, pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2 = sum(sum(cm2))
    accuracy2 = (cm2[0, 0] + cm2[1, 1]) / total2
    print('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0, 0] / (cm2[0, 0] + cm2[0, 1])
    print('Trianing Specificity : ', specificity2)
    sensitivity2 = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])
    print('Trianing Sensitivity : ', sensitivity2)

    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))

    #####from decision_function() get roc_curve()
    score_test = best_svm.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr, tpr, threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr, tpr)  # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")

    plt.figure(figsize=(4, 4),dpi=800)
    out = plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out = plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out = plt.xlim([0.0, 1.0])
    out = plt.ylim([0.0, 1.05])
    out = plt.xlabel('False Positive Rate', fontsize=10)
    out = plt.ylabel('True Positive Rate', fontsize=10)
    out = plt.title('ROC and AUC (Test)', fontsize=12)
    out = plt.legend(loc="lower right")
    plt.show()

    # output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=", auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_svm.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1, tpr1, threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1, tpr1)
    print("trainingAUC=", auc(fpr1, tpr1))

    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    eva_name = ['TrainingAUC','Accuracy','Specificity','Sensitivity','TestAUC','Accuracy','Specificity','Sensitivity']
    eva_value = [auc(fpr1, tpr1),accuracy2,specificity2,sensitivity2,auc(fpr, tpr),accuracy1,specificity1,sensitivity1]
    
    name = np.hstack((test_patientid,eva_name))
    val = np.hstack((new_list,eva_value))
    
    test_label_df = pd.DataFrame({"patient": name,
                                  "pred_label": val})

    savefilename_test = saveDir + my_kernel + '_Test_result_' + str(current_time) + '.csv'
    test_label_df.to_csv(savefilename_test)

    params_label = ['kernel', 'C','gamma']
    params_bestvalue = [my_kernel, C, gamma]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + my_kernel + 'BestParameters_SVM_' + str(current_time) + '.csv'
    best_params.to_csv(savefilename_bestparams)

    print("File saved")
        
else: #poly
    print("Choosing penalty:", my_kernel)
    svm_bo = BayesianOptimization(poly_opt, params_svm_poly, random_state=1122)
    svm_bo.maximize(init_points=10, n_iter=my_iter)  # n_iter: controls the times of optimization
    print("Best Parameter Setting : {}".format({"c": svm_bo.max["params"]["c"], 
                                                "gamma":svm_bo.max["params"]["gamma"], 
                                                "degree":degrees[5 if svm_bo.max["params"]["degree"] > 0.85 
                                                                   else (4 if svm_bo.max["params"]["degree"] >0.68 
                                                                        else (3 if svm_bo.max["params"]["degree"] >0.51
                                                                              else (2 if svm_bo.max["params"]["degree"] >0.34
                                                                                    else (1 if svm_bo.max["params"]["degree"] >0.17
                                                                                          else 0))))]}))
    print("-----------------------------End Optimization---------------------------------")
    
    

    C = svm_bo.max["params"]["c"]
    gamma = svm_bo.max["params"]["gamma"]
    degree = degrees[5 if svm_bo.max["params"]["degree"] > 0.85 
                     else (4 if svm_bo.max["params"]["degree"] >0.68 
                           else (3 if svm_bo.max["params"]["degree"] >0.51
                                 else (2 if svm_bo.max["params"]["degree"] >0.34
                                       else (1 if svm_bo.max["params"]["degree"] >0.17
                                             else 0))))]

    best_svm = svm.SVC(kernel=my_kernel, C=C, gamma=gamma, degree=degree, random_state=1122, decision_function_shape='ovo', probability=True)
    best_svm.fit(train_data, train_label.ravel())

    pred_label = best_svm.predict(test_data)
    pred_label_train = best_svm.predict(train_data)

    print("predicted label:", pred_label)
    new_list = [label[item] for item in pred_label]
    print("label name:", new_list)

    print(classification_report(test_label, pred_label))
    ##1-recall：sensitivity
    ##0-recall：specificity

    #####from confusion matrix calculate accuracy -- test
    print("------------------Test Confusion Matrix------------------")
    cm1 = confusion_matrix(test_label, pred_label)
    print('Test Confusion Matrix : \n', cm1)
    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Test Accuracy : ', accuracy1)
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Test Specificity : ', specificity1)
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Test Sensitivity : ', sensitivity1)

    #####from confusion matrix calculate accuracy -- training
    print("----------------Training Confusion Matrix-----------------")
    cm2 = confusion_matrix(train_label, pred_label_train)
    print('Trianing Confusion Matrix : \n', cm2)
    total2 = sum(sum(cm2))
    accuracy2 = (cm2[0, 0] + cm2[1, 1]) / total2
    print('Trianing Accuracy : ', accuracy2)
    specificity2 = cm2[0, 0] / (cm2[0, 0] + cm2[0, 1])
    print('Trianing pecificity : ', specificity2)
    sensitivity2 = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])
    print('Trianing Sensitivity : ', sensitivity2)

    # print("5-Fold Cross Validation:")
    # accuracy_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='accuracy', cv=my_cv)
    # auc_5cv_train = cross_val_score(best_lr, train_data, train_label, scoring='roc_auc', cv=my_cv)
    # print("Accuracy:", accuracy_5cv_train)
    # print('Mean Accuracy (std): %.3f (%.3f)' % (mean(accuracy_5cv_train), std(accuracy_5cv_train)))
    # print("AUC:",auc_5cv_train)
    # print('Mean Auc (std): %.3f (%.3f)' % (mean(auc_5cv_train), std(auc_5cv_train)))

    #####from decision_function() get roc_curve()
    score_test = best_svm.fit(train_data, train_label.ravel()).decision_function(test_data)
    fpr, tpr, threshold = roc_curve(test_label, score_test)
    roc_auc = auc(fpr, tpr)  # Compute ROC curve and ROC area for each class

    #######################Test AUC#######################
    print("---------------------Test ROC and AUC--------------------")

    plt.figure(figsize=(4, 4),dpi=800)
    out = plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    out = plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    out = plt.xlim([0.0, 1.0])
    out = plt.ylim([0.0, 1.05])
    out = plt.xlabel('False Positive Rate', fontsize=10)
    out = plt.ylabel('True Positive Rate', fontsize=10)
    out = plt.title('ROC and AUC (Test)', fontsize=12)
    out = plt.legend(loc="lower right")
    plt.show()

    # output.savefig('C:/Users/ariken/Desktop/AUC_TEST.png',dpi=800,bbox_inches='tight')
    print("testAUC=", auc(fpr, tpr))

    #######################Training AUC#######################
    print("-------------------Training ROC and AUC-------------------")
    score_train = best_svm.fit(train_data, train_label.ravel()).decision_function(train_data)
    fpr1, tpr1, threshold1 = roc_curve(train_label, score_train)
    roc_auc_train = auc(fpr1, tpr1)
    print("trainingAUC=", auc(fpr1, tpr1))

    """---------------------------------------------------------
    Save predicted label
    ---------------------------------------------------------"""
    test_label_df = pd.DataFrame({"patient": test_patientid,
                                  "pred_label": new_list})

    savefilename_test = saveDir + my_kernel + '_Test_result' + '.csv'
    test_label_df.to_csv(savefilename_test)

    params_label = ['kernel', 'C','gamma','degree']
    params_bestvalue = [my_kernel, C, gamma,degree]
    best_params = pd.DataFrame([params_label, params_bestvalue])
    savefilename_bestparams = saveDir + 'BestParameters_SVM_'+ my_kernel + '.csv'
    best_params.to_csv(savefilename_bestparams)

    print("File saved")