import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import warnings
import math
from operator import itemgetter
import sys




#Calculate RMSE
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


#Calculate precision, recall, coverage and popularity
def evaluate(train, prediction, item_popular, name):
    hit = 0
    rec_count = 0
    test_count = 0
    popular_sum = 0
    hit_pred = set()
    for u_index in range(n_users):
        items = np.where(train_data_matrix[u_index, :] == 0)[0]
        pre_items = sorted(
            dict(zip(items, prediction[u_index, items])).items(),
            key=itemgetter(1),
            reverse=True)[:20]
        test_items = np.where(test_data_matrix[u_index, :] > 0)[0]
        for item, _ in pre_items:
            if item in test_items:
                hit += 1
            hit_pred.add(item)
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])
        rec_count += len(pre_items)
        test_count += len(test_items)
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(hit_pred) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    return precision, recall, coverage, popularity

#Calculate RMSE for svd item_base and user_base
def svd(train_data_matrix, n_users, n_items, test_data_matrix):
    svd_x = []
    svd_y = []
    precision_y = []
    for i in range(1,50,2):
        svd_x.append(i)
        u, s, v = svds(train_data_matrix, k = i) #s为分解的奇异值 u[1650, 1650] s[1650, 940] v[940, 940]
        s_matrix = np.diag(s)
        pred_svd = np.dot(np.dot(u, s_matrix), v)
        svd_y.append(rmse(pred_svd, test_data_matrix))
        svd_precision, recall, coverage, popularity = evaluate(train_data_matrix, pred_svd, item_popular, 'svd')
        precision_y.append(svd_precision)
    return svd_x, svd_y, precision_y



#Define label
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',100,'display.max_columns', 10000,"display.max_colwidth",10000,'display.width',10000)
header = ['user_id','item_id','rating','timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
u_cols = ['user_id', 'age', 'gender', 'occupation']

#Read the data and splite training data and test data
df_all = pd.read_csv('ml-100k/u.data',sep='\t',names=header)
df_train = pd.read_csv('ml-100k/u5.base',sep='\t',names=header)
df_test = pd.read_csv('ml-100k/u5.test',sep='\t',names=header)
df_item = pd.read_csv('ml-100k/u.item', sep='|',encoding='latin-1',usecols=range(5),names=m_cols)
rating = df_train

#merge item and user
df_train = pd.merge(df_train,df_item,on="item_id")
n_users = df_all.user_id.unique().shape[0]
n_items = df_all.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


ratings = pd.DataFrame(df_train.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = df_train.groupby('title')['rating'].count()
movie_matrix = df_train.pivot_table(index='user_id', columns='title', values='rating')

#Calculate similarity
train_data_matrix = np.zeros((n_users,n_items))
for row in df_train.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in df_test.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric = "cosine")

#Calculate popularity
item_popular = {}
for i in range(n_items):
    if np.sum(train_data_matrix[:,i]) != 0:
        item_popular[i] = np.sum(train_data_matrix[:,i] != 0)
item_count = len(item_popular)


#Similarity compensates for each person's rating habits and the popularity of movies
rate = train_data_matrix.mean(axis = 1)
rate2 = (train_data_matrix - rate[:, np.newaxis])
pred_user = rate[:, np.newaxis] + user_similarity.dot(rate2) / np.array([np.abs(user_similarity).sum(axis = 1)]).T
pred_item = train_data_matrix.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis = 1)])




#Show the effect of K change on RMSE
def plot_rmse(train_data_matrix, n_users, n_items, test_data_matrix):
    svd_x, svd_y, precision_y = svd(train_data_matrix, n_users, n_items, test_data_matrix)
    svd_rmse = plt.figure()
    a = svd_rmse.add_subplot(111)
    b = svd_rmse.add_subplot(222)
    a.set(xlim = [0,50], ylim = [2.6,3.5], title = 'k-rmse', xlabel = 'value of k', ylabel = 'RMSE')
    a.plot(svd_x, svd_y, color = 'red')
    b.set(xlim = [0,50], ylim = [0.1,0.3], title = 'k-rmse', xlabel = 'value of k', ylabel = 'Precision')
    b.plot(svd_x, precision_y, color = 'red')
    plt.show()




#add attributes of u.user and calculate the similarity
df_users = pd.read_csv('ml-100k/u.user', sep='|',usecols=range(4),names=u_cols)
df_occu = open('ml-100k/u.occupation')
df_occu = df_occu.readlines()
occupation = {}
count = 0
for i in df_occu:
    occupation[i.strip('\n')] = count
    count += 1
gender = {0:'F', 30:'M'}

#replace string
for i in occupation.keys():
    df_users = df_users.replace(i, occupation[i])
df_users = df_users.replace('M', 30)
df_users = df_users.replace('F', 0)

user_data_matrix = df_users.as_matrix()
user_similarity2 = pairwise_distances(user_data_matrix, metric = "cosine")

#merge two similarity and calculate
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
    precision_user.append(user_precision)


#Show the effect of weight change on precision and RMSE
def plot_user(weight, precision_user):
    user_rmse = plt.figure()
    c = user_rmse.add_subplot(111)
    d = user_rmse.add_subplot(222)
    c.set(xlim = [0,1], ylim = [3.0,3.2], title = 'weight-rmse', xlabel = 'value of weight', ylabel = 'RMSE')
    c.plot(weight, user_y)
    d.set(xlim = [0,1], ylim = [0.138,0.14], title = 'weight-precision', xlabel = 'value of weight', ylabel = 'Precision')
    d.plot(weight, precision_user)
    plt.show()


#add classification of movie to calculate the similarity
cols = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
df_items = pd.read_csv('ml-100k/u.item', sep='|',encoding='latin-1',names=cols)

df_items.drop(df_items.columns[1:5],axis=1,inplace=True)

item_data_matrix = df_items.as_matrix()

item_similarity2 = pairwise_distances(item_data_matrix, metric = "cosine")

#merge two similarity and calculate
weight2 = []
item_y = []
precision_item = []
for i in range(0,10,1):
    weight2.append(i/10)
    new_item_similar = i/10 * item_similarity + (1-i)/10 * item_similarity2
    rate = train_data_matrix.mean(axis = 1)
    rate2 = (train_data_matrix - rate[:, np.newaxis])
    new_pred_item = train_data_matrix.dot(new_item_similar) / np.array([np.abs(new_item_similar).sum(axis=1)])
    item_precision, item_recall, item_coverage, item_popularity = evaluate(train_data_matrix, new_pred_item, item_popular,'item')
    item_y.append(rmse(new_pred_item, test_data_matrix))
    precision_item.append(item_precision)


#Show the effect of weight change on precision and RMSE
def plot_item(weight, precision_item):
    item_rmse = plt.figure()
    e = item_rmse.add_subplot(111)
    f = item_rmse.add_subplot(222)
    e.set(title = 'weight-rmse', xlabel = 'value of weight', ylabel = 'RMSE')
    e.plot(weight, item_y)
    f.set(title = 'weight-precision', xlabel = 'value of weight', ylabel = 'Precision')
    f.plot(weight, precision_item)
    plt.show()


#evaluate
plot_rmse(train_data_matrix, n_users, n_items, test_data_matrix)
plot_item(weight2, precision_item)
plot_user(weight, precision_user)

