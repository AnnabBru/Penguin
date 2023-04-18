# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
#import tensorflow
#tensorflow.random.set_seed(1234)
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
import pickle
import joblib
from sklearn import svm
classifier =svm.SVC(gamma='scale',C=1,probability=True)
import plot_learning_curves as plc
from sklearn.preprocessing import MinMaxScaler #For feature normalization
scaler = MinMaxScaler()

df1 = pd.read_csv("Pseu_Modification_coors.txt",sep=' ',skiprows=(0),header=(0))
df2 = pd.read_csv("eventalign_train.txt",sep='\t',skiprows=(0),header=(0))
df3 = pd.read_csv("Pseu_Modification_coors.txt",sep=' ',skiprows=(0),header=(0))
df4 = pd.read_csv("eventalign_test.txt",sep='\t',skiprows=(0),header=(0))

print(df2.shape)
print("&&&&&&&&")
print(df1.head())
print("***********************")
print(df2.head())
print("######################")
print(df3.head())
print("######################")
print(df4.head())

model_kmer_list=list(df2.iloc[:, 9]) #10 for model-kmer that
model_kmer_list_test=list(df4.iloc[:, 9]) #10 for model-kmer that
print("333333333333333333", type(model_kmer_list))
print("333333333333333333", type(model_kmer_list_test))
print(model_kmer_list[5])
print(model_kmer_list[5][2])
print(model_kmer_list_test[5])
print(model_kmer_list_test[5][2])

def filter_df(df1,df2,chrom):
    df_pseu = pd.DataFrame()
    df_U = pd.DataFrame()
    for el in chrom :
        df1_sub = df1[df1.iloc[:,0] == el]
        df2_sub = df2[df2["contig"] == el]
        x_sub = list(set(df1_sub.iloc[:,1]-2).intersection(set(df2_sub.iloc[:,1])) )
        df_pseu_sub = df2_sub[df2_sub['position'].isin(x_sub)]
        df_pseu = df_pseu.append(df_pseu_sub)
        df_U_sub = df2_sub[~df2_sub['position'].isin(x_sub)]
        df_U = df_U.append(df_U_sub)

    return df_pseu, df_U

chrom_train = list(set(df1.iloc[:,0]))
chrom_test = list(set(df3.iloc[:,0]))

U_kmer_list=[]
for i in model_kmer_list:
    #print(i)
    if i[2]=='T':
        U_kmer_list.append(i)

U_kmer_list_test=[]
for i in model_kmer_list_test:
    #print(i)
    if i[2]=='T' :
        U_kmer_list_test.append(i)

df=df2[df2['model_kmer'].isin(U_kmer_list)]
df_test = df4[df4['model_kmer'].isin(U_kmer_list_test)]

df_pseu,df_U = filter_df(df1,df,chrom_train)
df_pseu_test,df_U_test = filter_df(df3,df_test,chrom_test)

#df_pseu = df_pseu.sample(n=2000, replace=False)#For Arabidopsis

listofones = [1] * len(df_pseu.index)
df_pseu.insert(13, "label", listofones, True)

listofones = [1] * len(df_pseu_test.index)
df_pseu_test.insert(13, "label", listofones, True)

listofzeros=[0]*len(df_U.index)
df_U.insert(13, "label", listofzeros, True)

listofzeros=[0]*len(df_U_test.index)
df_U_test.insert(13, "label", listofzeros, True)

##########prepare training datast  
   
df_U = df_U.sample(n=len(df_pseu), replace=False) #try replace=false
# Create DataFrame from positive and negative examples
dataset = df_U.append(df_pseu, ignore_index=True)
print('DATASET :',dataset.head())
#dataset['label'] = dataset['label'].astype('category')
columns=['event_level_mean','event_stdv','event_length']
#columns=['event_level_mean']
#columns=['event_stdv']
#columns=['event_length']

##################################
#prepare testing datast       
####df_U_test = df_U_test.sample(n=len(df_pseu_test), replace=False) #try replace=false

#np.savetxt('pseu_samples.txt', df_pseu_test,fmt='%s')
#np.savetxt('U_samples.txt', df_U_test,fmt='%s')

# Create DataFrame from positive and negative examples
dataset_test = df_U_test.append(df_pseu_test, ignore_index=True)

#dataset_test['label'] = dataset_test['label'].astype('category')

#shuffle the test and train datasets
from sklearn.utils import shuffle
dataset = shuffle(dataset)
dataset_test.to_csv("PreShuffle.csv",index=False)
dataset_test=shuffle(dataset_test)

#combine onehot_encoding of train and test
union_reference_kmer_set=set(dataset.iloc[:, 2]).union(set(dataset_test.iloc[:, 2]))
union=list(union_reference_kmer_set)
print(len(union))
dataset['reference_kmer']=pd.Categorical(dataset['reference_kmer'], categories=list(union))
dataset_test['reference_kmer']=pd.Categorical(dataset_test['reference_kmer'], categories=list(union))


X_train = dataset[columns]
#insert onehot encoding of reference-kmer in train data
Onehot=pd.get_dummies(dataset['reference_kmer'], prefix='reference_kmer')
X_train= pd.concat([X_train,Onehot],axis=1)
#X_train=Onehot
print("#############",X_train.shape)
print(X_train.head())

#scale training data
X_train= scaler.fit_transform(X_train)
y_train = dataset['label'] 
print(",,,,,,,,",X_train.shape)

X_test = dataset_test[columns]

#insert onehot encoding of reference-kmer in test data
Onehot=pd.get_dummies(dataset_test['reference_kmer'], prefix='reference_kmer')
X_test= pd.concat([X_test,Onehot],axis=1)
#X_test= Onehot

print("#############",X_test.shape)
print(X_test.head())

#scale training data
X_test= scaler.fit_transform(X_test)
y_test = dataset_test['label'] 

print(",,,,,,,,",X_test.shape)

###################################

from sklearn.model_selection import train_test_split

#train, test = train_test_split(df, test_size=0.2)   
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3) for unblanced dataset

#clf = classifier.fit(X_train,y_train)
clf = classifier.fit(X_train,y_train.ravel())

# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix

'''
#save the ML model to test on unseen dataset
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
'''
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob = y_prob[:,1]

def fill_tp_fp_fn(l,dataset):
    df = list()
    dico = {}

    for el in l :
        df.extend([dataset.iloc[el][0],str(dataset.iloc[el][1]+2)])

    df = ['_'.join(df[i:i+2]) for i in range(0,len(df),2)]
    for el in df :
        
        if el not in dico :
            dico[el] = 1
        else : 
            dico[el] = dico[el] + 1
    df = list(set(df))
    return df, dico

l_fp,l_tp,l_fn,l_tn = list(),list(),list(),list()
for i in range(len(y_pred)):
    if y_pred[i] == 1 and y_test.iloc[i] == 1 :
        l_tp.append(i)
    elif y_pred[i] == 1 and y_test.iloc[i] == 0 :
        l_fp.append(i)
    elif y_pred[i] == 0 and y_test.iloc[i] == 1 :
        l_fn.append(i)
    else : 
        l_tn.append(i)

tp_pos,d_tp = fill_tp_fp_fn(l_tp,dataset_test)
fp_pos,d_fp = fill_tp_fp_fn(l_fp,dataset_test)
fn_pos,d_fn = fill_tp_fp_fn(l_fn,dataset_test)
tn_pos, d_tn = fill_tp_fp_fn(l_tn,dataset_test)



print('TP_freq', d_tp)
print('#############')

print('FP_freq', d_fp)
print('#############')

print('FN_freq', d_fn)
print('#############')

#print('TN_freq', d_tn)
#print('#############')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)
#true negatives c00, false negatives C10, true positives C11, and false positives C01 
#tn c00, fpC01, fnC10, tpC11 
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_test, y_pred)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9


print('TN=',l.item((0, 0)))
print('FP=',l.item((0, 1)))
print('FN=',l.item((1, 0)))
print('TP=',l.item((1, 1)))

print("////////////////////")
print("Error Rate TRAIN")

y_pred_train = classifier.predict(X_train)
y_prob_train = classifier.predict_proba(X_train)
y_prob_train = y_prob_train[:,1]


l_fp,l_tp,l_fn,l_tn = list(),list(),list(),list()
for i in range(len(y_pred_train)):
    if y_pred_train[i] == 1 and y_train.iloc[i] == 1 :
        l_tp.append(i)
    elif y_pred_train[i] == 1 and y_train.iloc[i] == 0 :
        l_fp.append(i)
    elif y_pred_train[i] == 0 and y_train.iloc[i] == 1 :
        l_fn.append(i)
    else : 
        l_tn.append(i)


tp_pos = fill_tp_fp_fn(l_tp,dataset)
fp_pos = fill_tp_fp_fn(l_fp,dataset)
fn_pos = fill_tp_fp_fn(l_fn,dataset)
tn_pos = fill_tp_fp_fn(l_tn,dataset)

print('TP : ', tp_pos)
print('FP : ', fp_pos)
print('FN : ', fn_pos)
print('TN : ', tn_pos)

print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train)*100)
 
print(classification_report(y_train, y_pred_train))
auc=roc_auc_score(y_train.round(),y_pred_train)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)

#true negatives c00, false negatives C10, true positives C11, and false positives C01 

#tn c00, fpC01, fnC10, tpC11 
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_train, y_pred_train)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
print('TN=',l.item((0, 0)))
print('FP=',l.item((0, 1)))
print('FN=',l.item((1, 0)))
print('TP=',l.item((1, 1)))


#plot learning curve: works with all classifier and all features except x(padded signal) as it leads to error with SVM 
#References:https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53

#############################################
#old code to plot learning curve: works only with RandomForest
#Reference: https://www.dataquest.io/blog/learning-curves-machine-learning/
##################
