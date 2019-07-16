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

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))



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



















