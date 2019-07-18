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
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',100,'display.max_columns', 10000,"display.max_colwidth",10000,'display.width',10000)

header = ['user_id','item_id','rating','timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']

df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=header)
df_item = pd.read_csv('data/ml-100k/u.item', sep='|',encoding='latin-1',usecols=range(5),names=m_cols)
rating = df

df = pd.merge(df,df_item,on="item_id")
print(df.user_id)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))



ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

# ratings['rating'].hist(bins=50)
# plt.show()
# ratings['number_of_ratings'].hist(bins=60)
# plt.show()

# import seaborn as sns
# sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# # plt.show()

movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
# print(movie_matrix.head(10))
X_train,x_test = train_test_split(df,test_size=0.2)


train_data_matrix = np.zeros((n_users,n_items))
for row in X_train.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]
for row in x_test.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]

train_data_matrix1 = np.zeros((n_users,n_items))
for row in df.itertuples():
    train_data_matrix1[row[1]-1,row[2]-1] = row[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in x_test.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]



user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
item_similarity = cosine_similarity(train_data_matrix.T, dense_output=True)

# print(user_similarity)

def KNN(items, ratings, item_similarity, keywords, k):
    movie_list = [] 
    movie_id = list(items[items['title'].str.contains(keywords)].item_id)[0] 
    movie_similarity = item_similarity[movie_id - 1]
    movie_similarity_index = np.argsort(-movie_similarity)[1:k + 1]
    for index in movie_similarity_index:
        move_list = list(set(list(items[items['item_id'] == index + 1].title)))
        move_list.append(movie_similarity[index])  
        move_list.append(ratings[ratings['item_id'] == index + 1].rating.mean()) 
        # move_list.append(len(ratings[ratings['item_id'] == index + 1]))
        movie_list.append(move_list)
    return movie_list

print(KNN(df,rating,item_similarity,'Toy Story ',10))


def predict(rating, similarity, type = 'user'):
    if type == 'user':
        mean_user_rating = rating.mean(axis = 1)
        rating_diff = (rating - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
def predict_rate(rating, similarity):
    pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def predict_rate1(rating, similarity,user_id,title,):
    # item_id = list(df[df['title'].str.contains(title)].item_id)[0]
    print(similarity)
    similarity = similarity[title]
    print(rating)
    print(similarity)
    rating = rating[user_id]
    print(rating)
    print("************************")
    pred = rating.dot(similarity) / np.array([np.abs(similarity).sum()])
    return pred

# print(predict_rate(train_data_matrix, item_similarity))
# print(predict_rate1(train_data_matrix1,item_similarity,0,196))
item_prediction = predict_rate(train_data_matrix, item_similarity)
# print('item_prediction')
# print(item_prediction)


user_prediction = predict(train_data_matrix, user_similarity, type = 'user')
print(user_prediction)

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))


import scipy.sparse as sp

from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k = 20)

s_diag_matrix = np.diag(s)

x_pred = np.dot(np.dot(u,s_diag_matrix),vt)

print('User-based CF MSE: ' + str(rmse(x_pred, test_data_matrix)))