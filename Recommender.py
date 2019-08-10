import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',100,'display.max_columns', 10000,"display.max_colwidth",10000,'display.width',10000)

header = ['user_id','item_id','rating','timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']

df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=header)
df_item = pd.read_csv('data/ml-100k/u.item', sep='|',encoding='latin-1',usecols=range(5),names=m_cols)
rating = df

df = pd.merge(df,df_item,on="item_id")

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('users number = ' + str(n_users) + '    movies number = ' + str(n_items))
print()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()

ratings['rating'].hist(bins=50)
plt.show()
ratings['number_of_ratings'].hist(bins=60)
plt.show()

sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
plt.show()
movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
X_train,x_test = train_test_split(df,test_size=0.2)


# initialize the matrix
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

# caculate the similarity matrix
item_similarity = cosine_similarity(train_data_matrix.T, dense_output=True)


# predict according to "toy story" with top-12 with similarity and mean

def KNN(items, ratings, item_similarity, keywords, k):
    '''
    :param items: movelens pandas table
    :param ratings: ratings of movie
    :param item_similarity: similarity matrix
    :param keywords: movie name
    :param k: top-K
    :return: list contains 1 moveName 2 movieSimilarity 3 mean
    '''
    moveList = []
    movie_id = list(items[items['title'].str.contains(keywords)].item_id)[0] 
    movie_similarity = item_similarity[movie_id - 1]
    movie_similarity_index = np.argsort(-movie_similarity)[1:k + 1]
    for i in movie_similarity_index:
        list_mv = list(set(list(items[items['item_id'] == i + 1].title)))
        list_mv.append(movie_similarity[i])
        list_mv.append(ratings[ratings['item_id'] == i + 1].rating.mean())
        moveList.append(list_mv)
    return moveList

print(KNN(df,rating,item_similarity,'Toy Story ',12))
