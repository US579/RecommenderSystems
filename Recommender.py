import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100,'display.max_columns', 1000,"display.max_colwidth",1000,'display.width',1000)
header = ['user_id','item_id','rating','timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=header)
df2 = pd.read_csv('data/ml-100k/u.item', sep='|',
                  encoding='latin-1',usecols=range(5),names=m_cols)



df = pd.merge(df,df2,on="item_id")
print(df.head())
# print(df.describe())


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

ratings['rating'].hist(bins=50)
# plt.show()
ratings['number_of_ratings'].hist(bins=60)
# plt.show()

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# plt.show()

movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
print(movie_matrix.head())

# n_users = df.user_id.unique().shape[0]
# n_items = df.item_id.unique().shape[0]
# print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
#
#
# print(df2)
# X_train,x_test = cv.train_test_split(df,test_size=0.2)
#
# train_data_matrix = np.zeros((n_users,n_items))
# for line in X_train.itertuples():
#     train_data_matrix[line[1]-1, line[2]-1] = line[3]
# test_data_matrix = np.zeros((n_users, n_items))
# for line in x_test.itertuples():
#     test_data_matrix[line[1]-1, line[2]-1] = line[3]
#
# ## 通过余弦相似度计算用户和物品的相似度
# user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
# item_similarity = pairwise_distances(train_data_matrix.T, metric = "cosine")
#
#
# def rmse(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return sqrt(mean_squared_error(prediction, ground_truth))
#
# # print ('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
# # print ('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))
# # print ('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
# # print ('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))
#
#
# # 计算矩阵的稀疏度
# sparsity = round(1.0 - len(df) / float(n_users*n_items),3)
# print('The sparsity level of MovieLen100K is ' + str(sparsity * 100) + '%')
#
#
#
# u, s, vt = svds(train_data_matrix, k = 20)
# s_diag_matrix = np.diag(s)
# x_pred = np.dot(np.dot(u,s_diag_matrix),vt)
# print('User-based CF MSE: ' + str(rmse(x_pred, test_data_matrix)))
#
#
#
#
#





















