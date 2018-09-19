#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:33:49 2018

@author: root
"""

import pandas as pd
import datetime
import csv 
import numpy as np
import re
#import math
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

#/////////////Processing data/////////////////////////

def processing_books():
    Books = {'ISBN':[],'BookTitle':[],'BookAuthor':[],'YearOfPublication':[],'Publisher':[]}
    with open('BX-Books.csv',encoding='latin1',newline='') as file:
        reader = csv.reader(file)
        count=0
        z=0
        for raw_data in reader:
            if count==0:
                count+=1
                continue
            temp=''.join(i for i in raw_data)
            temp=temp.replace('&amp;','')
            temp=temp.replace('"','')
            temp=temp.replace("\\",'')
            data=temp.split(";")
            z+=1
            for i in range(1,len(data)):
                if data[i].isdigit():
                    if(i-1 >1):
                        data[1:i-1]=[''.join(data[1:i-1])]
                    if 4<len(data)-3:
                        data[4:len(data)-3]=[''.join(data[4:len(data)-3])]
                    break
            if len(data)>=5:
                Books['ISBN'].append(data[0])
                Books['BookTitle'].append(data[1])
                Books['BookAuthor'].append(data[2].replace("/"," "))
                if data[3].isdigit():
                    Books['YearOfPublication'].append(data[3])
                    Books['Publisher'].append(data[4])
                elif data[4].isdigit():
                    Books['YearOfPublication'].append(data[4])
                    Books['Publisher'].append(data[3])
                else:
                    pass
            else:
                pass
                if len(data)>=5:
                    print(data)
    file.close()
    books=pd.DataFrame(data=Books)
    books.YearOfPublication=pd.to_numeric(books.YearOfPublication,errors='coerce')
    now = datetime.datetime.now()
    books.loc[(books.YearOfPublication>now.year),'YearOfPublication']=now.year
    books.loc[(books.YearOfPublication==0),'YearOfPublication']=np.NAN
    year_mean=round(books.YearOfPublication.mean())
    books.YearOfPublication.fillna(year_mean,inplace=True)
    books.YearOfPublication = books.YearOfPublication.astype(np.int32)
    return books

def processing_users():
    title = 0
    User_data = {'User_ID':[],'City':[],'State':[],'Country':[],'Age':[]}
    with open('BX-Users.csv',encoding = 'latin1',newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            location = data[1].split(",")
            location[-1] = re.sub(r'\.','',location[-1])
            if len(location) == 1:
                location = ['','','']
            if len(location) == 2:
                location.append('usa')
            User_data['User_ID'].append(data[0])
            User_data['City'].append(location[0])
            User_data['State'].append(location[1])
            User_data['Country'].append(location[2])
            User_data['Age'].append(data[2])

    file.close()
    df = pd.DataFrame(data = User_data)
    df.Age=pd.to_numeric(df.Age,errors='coerce')
    df.loc[(df.Age>116),'Age']=116
    df.loc[(df.Age == 0 ),'Age'] = np.NAN
    df.loc[(df.Age < 3 ),'Age'] = np.NAN
    age_mean=round(df.Age.mean())
    df.Age.fillna(age_mean,inplace=True)
    df.Age = df.Age.astype(np.int32)
    return df

def processing_rating():
    title = 0
    User_Book_Rating = {'User_ID':[],'ISBN':[],'Rating':[]}
    with open('BX-Book-Ratings.csv',encoding = 'latin1',newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            User_Book_Rating['User_ID'].append(data[0])
            User_Book_Rating['ISBN'].append(data[1])
            User_Book_Rating['Rating'].append(data[2])
    file.close()
    df = pd.DataFrame(data = User_Book_Rating)
    df.Rating = df.Rating.astype(np.int32)
    df.User_ID = df.User_ID.astype(np.int32)
    return df

#/////////////User base////////////////////
    
def recommand_based_on_topk_user (k, user, rating_matrix,user_sim,users_mean_rating):
    user_index=user_sim.index.get_loc(user)
    user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
    prediction_matrix = np.zeros([1,rating_matrix.shape[1]])
    for item in np.where(rating_matrix.values[user_index,:]==0)[0]:
        temp = user_id[[rating_matrix.values[:,item][user_id]>0]]
        if(temp.shape[0]==0):
            prediction_matrix[0, item] = users_mean_rating[user_index]
        else:
            if(temp.shape[0]>k):
                top_k_user_id=temp[:k]
            else:
                top_k_user_id=temp
            denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
            numerator = user_sim.values[user_index,:][top_k_user_id].dot(rating_matrix.values[:,item][top_k_user_id]-users_mean_rating[top_k_user_id])
            if denominator==0:
                prediction_matrix[0, item] = users_mean_rating[user_index]
            else:
                temp=round(users_mean_rating[user_index]+ numerator/denominator)
                if temp>10:
                    prediction_matrix[0, item] = 10
                elif temp<1:
                    prediction_matrix[0, item] = 1
                else:
                    prediction_matrix[0, item] = temp
    rl=np.argsort(prediction_matrix)[0][::-1]
    return rl[:10]

#/////////////Item base////////////////////
    
def recommand_based_on_topk_item (k, user, rating_matrix,item_sim,items_mean_rating):
    user_index=rating_matrix.index.get_loc(user)
    prediction_matrix = np.zeros([1,rating_matrix.shape[1]])
    for item in np.where(rating_matrix.values[user_index,:]==0)[0]:
        items_id = np.array([np.argsort(item_sim.values[:,item])][0][::-1])
        temp = items_id[[rating_matrix.values[user_index,:][items_id]>0]]
        if(temp.shape[0]==0):
            prediction_matrix[0, item] = items_mean_rating[item]
        else:
            if(temp.shape[0]>k):
                top_k_item_id=temp[:k]
            else:
                top_k_item_id=temp
            denominator = np.sum(item_sim.values[item,:][top_k_item_id])
            numerator = item_sim.values[item,:][top_k_item_id].dot(rating_matrix.values[user_index,:][top_k_item_id]-items_mean_rating[top_k_item_id])
            if denominator==0:
                prediction_matrix[0, item] = items_mean_rating[item]
            else:
                temp=round(items_mean_rating[item]+ numerator/denominator)
                if temp>10:
                    prediction_matrix[0, item] = 10
                elif temp<1:
                    prediction_matrix[0, item] = 1
                else:
                    prediction_matrix[0, item] = temp
    rl=np.argsort(prediction_matrix)[0][::-1]
    return rl[:10]

def recommand_based_on_model_based (result_matrix,rating_matrix,user):
    user_index=rating_matrix.index.get_loc(user)
    prediction_matrix = result_matrix[user_index]
    rl=np.argsort(prediction_matrix)[::-1]
    return rl[:10]

def predict_based_on_topk_users_a (k, user_sim, training_set, testing_set):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = 0
            else:
                if(temp.shape[0]>k):
                    top_k_user_id=temp[:k]
                else:
                    top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = 0
                else:
                    prediction_matrix[user, item] = round(numerator/denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    return RMSE

# ///////////////User base//////////////////////////
    
def predict_based_on_topk_users_p (k, user_sim, training_set, testing_set,users_mean_rating):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = users_mean_rating[user]
            else:
                if(temp.shape[0]>k):
                    top_k_user_id=temp[:k]
                else:
                    top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id]-users_mean_rating[top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = users_mean_rating[user]
                else:
                    temp=round(users_mean_rating[user]+ numerator/denominator)
                    if temp>10:
                        prediction_matrix[user, item] = 10
                    elif temp<1:
                        prediction_matrix[user, item] = 1
                    else:
                        prediction_matrix[user, item] = temp
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    return RMSE

# ///////////////Item base//////////////////////////

def predict_based_on_topk_items_p (k, item_sim, training_set, testing_set,items_mean_rating):
    prediction_matrix=np.zeros(testing_set.shape)
    for item in range(len(prediction_matrix)):
        item_index=item_sim.index.get_loc(testing_set.index[item])
        items_id = np.array([np.argsort(item_sim.values[:,item_index])][0][::-1])
        for user in testing_set.values[item,:].nonzero()[0]:
            temp = items_id[[training_set.values[:,user][items_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[item, user] = items_mean_rating[item]
            else:
                if(temp.shape[0]>k):
                    top_k_item_id=temp[:k]
                else:
                    top_k_item_id=temp
                denominator = np.sum(item_sim.values[item_index,:][top_k_item_id])
                numerator = item_sim.values[item_index,:][top_k_item_id].dot(training_set.values[:,user][top_k_item_id]-items_mean_rating[top_k_item_id])
                if denominator==0:
                    prediction_matrix[item, user] = items_mean_rating[item]
                else:
                    temp=round(items_mean_rating[item]+ numerator/denominator)
                    if temp>10:
                        prediction_matrix[item, user] = 10
                    elif temp<1:
                        prediction_matrix[item, user] = 1
                    else:
                        prediction_matrix[item, user] = temp
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    return RMSE

def SVD_model_fit_and_predict(k,training_matrix,testing_matrix):
    training_matrix_nan=training_matrix.copy()
    training_matrix_nan[training_matrix_nan==0]=np.nan
    tm = training_matrix_nan.values
    tm_mean=np.nanmean(tm,axis=0,keepdims=True)
    tm=tm-tm_mean
    tm[np.isnan(tm)]=0
    ts = csc_matrix(tm).asfptype()
    u, s, vt = svds(ts, k)
    s_diag_matrix=np.diag(s)
    X_pred = np.around(np.dot(np.dot(u, s_diag_matrix), vt)+tm_mean)
    nz=testing_matrix.values.nonzero()
    tv = testing_matrix.values[nz[0],nz[1]]
    pv = X_pred[nz[0],nz[1]]
    mse=((pv-tv) ** 2).mean(axis=0)
    nnz=nnan_rating_matrix.values.nonzero()
    ttv = training_matrix.values[nnz]
    ppv=X_pred[nnz[0],nnz[1]]
    mmse=((ppv-ttv) ** 2).mean(axis=0)
    train_rmse = round(sqrt(mmse),3)
    RMSE = round(sqrt(mse),3)
    print(train_rmse,RMSE)
    return X_pred

#/////////////////main part//////////////////
#%%
books=processing_books()

users=processing_users()

rating=processing_rating()
#%%
new_rating=rating[rating.ISBN.isin(books.ISBN)]

new_rating=new_rating[new_rating.User_ID.isin(users.User_ID)]

rating_exp = new_rating[new_rating.Rating!=0]
#%%
# build user-item matrix, only consider  users who have rated at least 10 books and books which have at least 10 ratings. 

previous_shape=np.inf
while (rating_exp.shape[0]<previous_shape):
    previous_shape = rating_exp.shape[0]
    counts=rating_exp['ISBN'].value_counts()
    
    rating_exp=rating_exp[rating_exp['ISBN'].isin(counts[counts>=7].index)]
    
    counts1=rating_exp['User_ID'].value_counts()
    
    rating_exp=rating_exp[rating_exp['User_ID'].isin(counts1[counts1>=7].index)]

#%%
rating_exp=rating_exp.sample(frac=1).reset_index(drop=True)
rating_matrix=rating_exp.pivot(index='User_ID',columns='ISBN',values='Rating')
rating_matrix_item = rating_exp.pivot(index='ISBN',columns='User_ID',values='Rating')

#%%
kf = RepeatedKFold(n_splits=10,n_repeats=1)

nnan_rating_matrix = rating_matrix.fillna(0)
nnan_rating_matrix_item = rating_matrix_item.fillna(0)

#%%
# Test Part to find the appropriate k in Item-base model, user-base model

#pearson,euclidean,cosine,mean_centered_cos,mean_centered_cos1=[],[],[],[],[]
#pearson_i,euclidean_i,cosine_i,mean_centered_cos_i,mean_centered_cos_i1=[],[],[],[],[]
#
#performance_k=[]
#k_list = [i for i in range(5,100,5)]
#for train_index, test_index in kf.split(rating_exp):
#    training_set=rating_exp.iloc[train_index]
#    testing_set= rating_exp.iloc[test_index]
#    n_users = rating_matrix.shape[0]
#
#    n_items = rating_matrix.shape[1]
#    
#    testing_matrix = pd.DataFrame(np.zeros((n_users, n_items)),index=rating_matrix.index,columns=rating_matrix.columns)
#    training_matrix=nnan_rating_matrix.copy()
#    
#    testing_matrix_item = pd.DataFrame(np.zeros((n_items,n_users)),index=rating_matrix_item.index,columns=rating_matrix_item.columns)
#    training_matrix_item = nnan_rating_matrix_item.copy()
#    
#    user_sim = pd.DataFrame(data=1-pairwise_distances(training_matrix, metric="correlation"),index=rating_matrix.index,columns=rating_matrix.index)
#
#    user_sim_item = pd.DataFrame(data=1-pairwise_distances(training_matrix_item, metric="correlation"),index=rating_matrix_item.index,columns=rating_matrix_item.index)
#    
#    user_sim1 = pd.DataFrame(data=1/(1+pairwise_distances(training_matrix, metric="euclidean")),index=rating_matrix.index,columns=rating_matrix.index)
#
#    user_sim1_item = pd.DataFrame(data=1/(1+pairwise_distances(training_matrix_item, metric="euclidean")),index=rating_matrix_item.index,columns=rating_matrix_item.index)
#   
#    user_sim2 = pd.DataFrame(data=cosine_similarity(training_matrix),index=rating_matrix.index,columns=rating_matrix.index)
#   
#    user_sim2_item = pd.DataFrame(data=cosine_similarity(training_matrix_item),index=rating_matrix_item.index,columns=rating_matrix_item.index)
#   
#    temp_rating=training_matrix.values.astype(float)
#    temp_rating_item=training_matrix_item.values.astype(float)
#
#    temp_rating[temp_rating==0]=np.nan
#    temp_rating_item[temp_rating_item==0]=np.nan
#    
#    users_mean_rating = np.nanmean(temp_rating, axis=1)
#    users_mean_rating_item = np.nanmean(temp_rating_item, axis=1)
#    
#    for index,row in testing_set.iterrows():
#        training_matrix.loc[row['User_ID'],row['ISBN']]=0
#        testing_matrix.loc[row['User_ID'],row['ISBN']]=nnan_rating_matrix.loc[row['User_ID'],row['ISBN']]
#
#        training_matrix_item.loc[row['ISBN'],row['User_ID']]=0
#        testing_matrix_item.loc[row['ISBN'],row['User_ID']]=nnan_rating_matrix_item.loc[row['ISBN'],row['User_ID']]
#
#    temp=[]
#    for k in k_list:
#        temp.append(predict_based_on_topk_users_p(k, user_sim,training_matrix,testing_matrix,users_mean_rating))
#    performance_k.append(temp)
#colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
#for i in range(len(colors)):
#    plt.plot(k_list,performance_k[i],color=colors[i],label=i)
#plt.show()

#///////////formula_1 vs formula_2 testing part/////////

#u_1=[]
#for i in k_list:
#    u_1.append(predict_based_on_topk_users_a(i,user_sim,training_matrix,testing_matrix))
#u_2=[]
#for i in k_list:
#    u_2.append(predict_based_on_topk_users_p(i,user_sim,training_matrix,testing_matrix,users_mean_rating))
#plt.plot(k_cos_user,u_1,color='C1',label='F1')
#plt.plot(k_cos_user,u_2,color='C2',label='F2')
#plt.show()

#/////////////compare 3 different similarity/////////////////
###user_base
#s_1=[]
#s_2=[]
#s_3=[]
#for i in k_list:
#    s_1.append(predict_based_on_topk_users_p(i,user_sim,training_matrix,testing_matrix,users_mean_rating))
#    
#for i in k_list:
#    s_2.append(predict_based_on_topk_users_p(i,user_sim1,training_matrix,testing_matrix,users_mean_rating))

#for i in k_list:
#    s_3.append(predict_based_on_topk_users_p(i,user_sim2,training_matrix,testing_matrix,users_mean_rating))
#    
#plt.plot(k_cos_user,s_1,color='C1',label='correlation')
#plt.plot(k_cos_user,s_2,color='C2',label='euclidean')
#plt.plot(k_cos_user,s_3,color='C3',label='cosine')
#plt.show()

###item base




# Test Part to find the appropriate k in SVD Model
#rmse=[]
#data=[]
#for i in range(1,200,20):
#    result=[]
#    for train_index, test_index in kf.split(rating_exp):
#        training_set=rating_exp.iloc[train_index]
#        testing_set= rating_exp.iloc[test_index]
#        n_users = rating_matrix.shape[0]
#    
#        n_items = rating_matrix.shape[1]
#        
#        testing_matrix = pd.DataFrame(np.zeros((n_users, n_items)),index=rating_matrix.index,columns=rating_matrix.columns)
#        training_matrix=nnan_rating_matrix.copy()
#        
#          
#        for index,row in testing_set.iterrows():
#            training_matrix.loc[row['User_ID'],row['ISBN']]=0
#            testing_matrix.loc[row['User_ID'],row['ISBN']]=nnan_rating_matrix.loc[row['User_ID'],row['ISBN']]
#        
#    
#        result.append(SVD_model_fit_and_predict(i,training_matrix,testing_matrix))
#    data.append((i,np.mean([x[0] for x in result]),np.mean([x[1] for x in result])))
##%%
#plt.plot([x[0] for x in data],[x[1] for x in data],label='training_set_RMSE')
#plt.plot([x[0] for x in data],[x[2] for x in data],label='testing_set_RMSE')
#plt.ylabel('rmse')
#plt.xlabel('Number of singular values')
#plt.legend()
#plt.show()

#### recommend books
temp_rating=nnan_rating_matrix.values.astype(float)
temp_rating_item=nnan_rating_matrix_item.values.astype(float)

temp_rating[temp_rating==0]=np.nan
temp_rating_item[temp_rating_item==0]=np.nan
user_sim2 = pd.DataFrame(data=cosine_similarity(nnan_rating_matrix),index=rating_matrix.index,columns=rating_matrix.index)
user_sim_item = pd.DataFrame(data=1-pairwise_distances(nnan_rating_matrix_item, metric="correlation"),index=rating_matrix_item.index,columns=rating_matrix_item.index)

users_mean_rating = np.nanmean(temp_rating, axis=1)
users_mean_rating_item = np.nanmean(temp_rating_item, axis=1)

user_id = input("Please input user id, we can recommend some books for this user")
try:
   val = int(user_id)
   if val not in nnan_rating_matrix.index:
       raise ValueError
except ValueError:
   print("That's not valid user!")
rl_1 = recommand_based_on_topk_user(100,val,nnan_rating_matrix,user_sim2,users_mean_rating)
rl_2 = recommand_based_on_topk_item(25,val,nnan_rating_matrix,user_sim_item,users_mean_rating_item)
result_matrix=SVD_model_fit_and_predict(100,nnan_rating_matrix,nnan_rating_matrix)
rl_3 = recommand_based_on_model_based(result_matrix,nnan_rating_matrix,val)
temp=list(set(rl_1).union(set(rl_2)))
result = list(set(temp).union(set(rl_3)))
print(nnan_rating_matrix.columns[result].values)