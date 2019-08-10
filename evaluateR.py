import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import warnings
import math
from operator import itemgetter
import sys


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',100,'display.max_columns', 10000,"display.max_colwidth",10000,'display.width',10000)
header = ['user_id','item_id','rating','timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
u_cols = ['user_id', 'age', 'gender', 'occupation']

#读取文档，分为测试集和训练集
df_all = pd.read_csv('./data/ml-100k/u.data',sep='\t',names=header)
df_train = pd.read_csv('./data/ml-100k/u5.base',sep='\t',names=header)
df_test = pd.read_csv('./data/ml-100k/u5.test',sep='\t',names=header)
df_item = pd.read_csv('./data/ml-100k/u.item', sep='|',encoding='latin-1',usecols=range(5),names=m_cols)
rating = df_train
#print(df_item)

#合并item和data
df_train = pd.merge(df_train,df_item,on="item_id")
n_users = df_all.user_id.unique().shape[0]
n_items = df_all.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


ratings = pd.DataFrame(df_train.groupby('title')['rating'].mean())
#print(ratings)
ratings['number_of_ratings'] = df_train.groupby('title')['rating'].count()
#print(ratings.head())
movie_matrix = df_train.pivot_table(index='user_id', columns='title', values='rating')

#计算相似度
#print(df_train)
train_data_matrix = np.zeros((n_users,n_items))
for row in df_train.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in df_test.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric = "cosine")

#统计流行度
item_popular = {}
for i in range(n_items):
    if np.sum(train_data_matrix[:,i]) != 0:
        item_popular[i] = np.sum(train_data_matrix[:,i] != 0)
#print(item_popular)
item_count = len(item_popular)
#print(item_count)




#统计综合打分
#相似度考虑到每个人打分的习惯和电影的流行度进行补偿
rate = train_data_matrix.mean(axis = 1)
rate2 = (train_data_matrix - rate[:, np.newaxis])
pred_user = rate[:, np.newaxis] + user_similarity.dot(rate2) / np.array([np.abs(user_similarity).sum(axis = 1)]).T
print('pred_user', pred_user)
pred_item = train_data_matrix.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis = 1)])
#print('pred_item', pred_item)

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
# print('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))
# print('Item based CF RMSE: ' + str(rmse(pred_item, test_data_matrix)))
# print('User based CF RMSE: ' + str(rmse(pred_user, test_data_matrix)))



def evaluate(train, prediction, item_popular, name):
    hit = 0
    rec_count = 0
    test_count = 0
    popular_sum = 0
    hit_pred = set()
    #print(train_data_matrix)
    for u_index in range(n_users):
        items = np.where(train_data_matrix[u_index, :] == 0)[0]
        #print('item', items)
        pre_items = sorted(
            dict(zip(items, prediction[u_index, items])).items(),
            key=itemgetter(1),
            reverse=True)[:20]
        print('oinginal suggestion：', items)
        print('recommend suggestion：', [key for key, value in pre_items])
        #print('item',pre_items)
        test_items = np.where(test_data_matrix[u_index, :] > 0)[0]
        #print('test',test_items)
        # 对比测试集和推荐集的差异 item, w
        for item, _ in pre_items:
            if item in test_items:
                hit += 1
            #elif int(train[u_index, item]) < 3:
                #hit += 1
            hit_pred.add(item)
            # 计算用户对应的电影出现次数log值的sum加和
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])
        rec_count += len(pre_items)
        test_count += len(test_items)
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(hit_pred) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    #print('name\tprecision\t\trecall\t\tcover\t\tpopular')
    #print(name, precision, recall, coverage, popularity)
    return precision, recall, coverage, popularity


def svd(train_data_matrix, n_users, n_items, test_data_matrix):
    #计算稀疏度
    spar = round(1.0 - len(df_train) / float(n_users * n_items), 3)
    #进行奇异值分解
    svd_x = []
    svd_y = []
    precision_y = []
    for i in range(1,50,2):
        svd_x.append(i)
        u, s, v = svds(train_data_matrix, k = i) #s为分解的奇异值 u[1650, 1650] s[1650, 940] v[940, 940]
        s_matrix = np.diag(s)
        pred_svd = np.dot(np.dot(u, s_matrix), v)
        svd_y.append(rmse(pred_svd, test_data_matrix))
        print('svd-rmse',rmse(pred_svd, test_data_matrix))
        print('user-rmse',rmse(pred_user, test_data_matrix))
        print('item-rmse',rmse(pred_item, test_data_matrix))
        svd_precision, recall, coverage, popularity = evaluate(train_data_matrix, pred_svd, item_popular, 'svd')
        precision_y.append(svd_precision)
        item_precision, item_recall, item_coverage, item_popularity = evaluate(train_data_matrix, pred_item, item_popular, 'item')
        user_precision, user_recall, user_coverage, user_popularity = evaluate(train_data_matrix, pred_user, item_popular, 'user')
    return svd_x, svd_y, precision_y

#讨论k取值和rmse的变化

svd_x, svd_y, precision_y = svd(train_data_matrix, n_users, n_items, test_data_matrix)
svd_rmse = plt.figure()
a = svd_rmse.add_subplot(111)
b = svd_rmse.add_subplot(222)
a.set(xlim = [0,50], ylim = [2.6,3.5], title = 'k-rmse', xlabel = 'value of k', ylabel = 'RMSE')
a.plot(svd_x, svd_y, color = 'red')
#plt.show()
b.set(xlim = [0,50], ylim = [0.1,0.3], title = 'k-rmse', xlabel = 'value of k', ylabel = 'Precision')
b.plot(svd_x, precision_y, color = 'red')
plt.show()




#加入gender等属性构建模型
df_users = pd.read_csv('./data/ml-100k/u.user', sep='|',usecols=range(4),names=u_cols)
print(df_users)
df_occu = open('./data/ml-100k/u.occupation')
df_occu = df_occu.readlines()
#print(df_occu)
occupation = {}
count = 0
for i in df_occu:
    occupation[i.strip('\n')] = count
    count += 1
#print(occupation)
gender = {0:'F', 30:'M'}

#替换string
for i in occupation.keys():
    df_users = df_users.replace(i, occupation[i])
df_users = df_users.replace('M', 30)
df_users = df_users.replace('F', 0)

#构建矩阵
user_data_matrix = df_users.as_matrix()
user_similarity2 = pairwise_distances(user_data_matrix, metric = "cosine")

#合并两个相似度并计算
weight = []
user_y = []
precision_user = []
for i in range(0,10,1):
    weight.append(i/10)
    new_users_similar = i/10 * user_similarity + (1-i)/10 * user_similarity2
    rate = train_data_matrix.mean(axis = 1)
    rate2 = (train_data_matrix - rate[:, np.newaxis])
    new_pred_user = rate[:, np.newaxis] + new_users_similar.dot(rate2) / np.array([np.abs(new_users_similar).sum(axis=1)]).T
    user_precision, user_recall, user_coverage, user_popularity = evaluate(train_data_matrix, new_pred_user, item_popular,'user')
    user_y.append(rmse(new_pred_user, test_data_matrix))
    #print(user_precision)
    #print(rmse(new_pred_user, test_data_matrix))
    precision_user.append(user_precision)
    # print('pred_user', pred_user)

user_rmse = plt.figure()
c = user_rmse.add_subplot(111)
d = user_rmse.add_subplot(222)
c.set(xlim = [0,1], ylim = [3.0,3.2], title = 'weight-rmse', xlabel = 'value of weight', ylabel = 'RMSE')
c.plot(weight, user_y)
d.set(xlim = [0,1], ylim = [0.138,0.14], title = 'weight-precision', xlabel = 'value of weight', ylabel = 'Precision')
d.plot(weight, precision_user)
plt.show()
print("result is weight should be 0.1, 0.1是指电影评分占的权重")



#加入类别等元素构建similar
cols = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
df_items = pd.read_csv('./data/ml-100k/u.item', sep='|',encoding='latin-1',names=cols)

df_items.drop(df_items.columns[1:5],axis=1,inplace=True)

item_data_matrix = df_items.as_matrix()

item_similarity2 = pairwise_distances(item_data_matrix, metric = "cosine")

#合并两个相似度并计算
weight = []
item_y = []
precision_item = []
for i in range(0,10,1):
    weight.append(i/10)
    new_item_similar = item_similarity
    #new_item_similar = i/10 * item_similarity + (1-i)/10 * item_similarity2
    rate = train_data_matrix.mean(axis = 1)
    rate2 = (train_data_matrix - rate[:, np.newaxis])
    new_pred_item = train_data_matrix.dot(new_item_similar) / np.array([np.abs(new_item_similar).sum(axis=1)])
    item_precision, item_recall, item_coverage, item_popularity = evaluate(train_data_matrix, new_pred_item, item_popular,'item')
    item_y.append(rmse(new_pred_item, test_data_matrix))
    precision_item.append(item_precision)


item_rmse = plt.figure()
e = item_rmse.add_subplot(111)
f = item_rmse.add_subplot(222)
#e.set(xlim = [0,1], ylim = [3.0,3.2], title = 'weight-rmse', xlabel = 'value of weight', ylabel = 'RMSE')
e.plot(weight, item_y)
#f.set(xlim = [0,1], ylim = [0.138,0.14], title = 'weight-precision', xlabel = 'value of weight', ylabel = 'Precision')
f.plot(weight, precision_item)
plt.show()
print("result is weight should be 0.0, 0.0是指电影评分占的权重")



