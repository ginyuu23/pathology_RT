# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:38:50 2023

@author: ariken
"""

import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt

import time

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
# "error", "ignore", "always", "default", "module" or "once"

def type(s):
    it = {b'Non-response':0, b'Response':1}
    return it[s]

basepath = "/response/dataset/selected/"

selection = "missf"

trainfile = selection + "_training.txt"
testfile = selection + "_test.txt"
idfile = selection + "_test.csv"

saveDir = "response/result/selected/" + selection + "/"

pathtrain = basepath + trainfile
pathtest = basepath + testfile
pathresult = basepath + idfile     #for reading test patient ids

datatrain = np.loadtxt(pathtrain, dtype=float, delimiter='\t', converters={0:type})
datatest = np.loadtxt(pathtest, dtype=float, delimiter='\t', converters={0:type})

result = pd.read_csv(pathresult, header=0, index_col=0)
test_patientid = result.index.values

train_label_nosmo, train_data_nosmo = np.split(datatrain,indices_or_sections=(1,),axis=1)
smo = SMOTE(random_state=233,k_neighbors=5)
train_data, train_label = smo.fit_resample(train_data_nosmo, train_label_nosmo)
test_label, test_data = np.split(datatest,indices_or_sections=(1,),axis=1)

#check labels
label ={0:"Non-response", 1:"Response"}

current_time = time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime())

my_iter = 200

method = "roc_auc"

"""---------------------------------------------------------
Define optimization function (Bayesian)
---------------------------------------------------------"""

def linear_opt(c):
    params_svm_linear = {}

    params_svm_linear['c'] = int(c)

    scores = cross_val_score(LinearSVC(loss="squared_hinge", C=c, dual=False, max_iter=1000, penalty="l2", random_state=0),
                                     train_data, train_label, scoring=method, cv=5).mean()
    return scores

"""---------------------------------------------------------
optimize the SVM and do machine learning
---------------------------------------------------------"""

########define parameter
params_svm_linear = {'c': (0,500)}

print("-----------------------------Start Optimization---------------------------------")
svm_bo = BayesianOptimization(linear_opt, params_svm_linear, random_state=0,allow_duplicate_points=True)
svm_bo.maximize(init_points=15, n_iter=my_iter)  # n_iter: controls the times of optimization
print("Best Parameter Setting : {}".format({"C": svm_bo.max["params"]["c"]}))
print("-----------------------------End Optimization---------------------------------")

C = svm_bo.max["params"]["c"]

best_svm = LinearSVC(loss="squared_hinge", C=C, dual=False, max_iter=1000, penalty="l2", random_state=0)
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

savefilename_test = saveDir + 'lSVM_Test_result_' + str(current_time) + '.csv'
test_label_df.to_csv(savefilename_test)

params_label = ['C']
params_bestvalue = [C]
best_params = pd.DataFrame([params_label, params_bestvalue])
savefilename_bestparams = saveDir + 'BestParameters_lSVM_' + str(current_time) + '.csv'
best_params.to_csv(savefilename_bestparams)

print("File saved")
