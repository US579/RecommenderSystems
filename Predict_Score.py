import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import warnings

# Read Data and Cleaning

Udata_header = ['user_id', 'item_id', 'rating', 'timestamp']
m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']

Udata_df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=Udata_header)
# Uitem_header = ['item_id', 'movie_title', 'release_date', 'video_release_date',
#                 'IMDb_URL', 'unknown', 'Action', 'Adventure' ,'Animation', 
#               'Childrens','Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
#               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
#               'Thriller', 'War', 'Western']
Uitem_df = pd.read_csv('data/ml-100k/u.item',sep='|',names=m_cols,encoding='latin1',usecols=range(5))

Ugenre_header = ['Type','id']
Ugenre_df = pd.read_csv('data/ml-100k/u.genre',sep='|',names=Ugenre_header)

Uuser_header = ['user_id', 'age', 'gender', 'occupation' ,'zip_code']
Uuser_df = pd.read_csv('data/ml-100k/u.user',sep='|',names=Uuser_header)

Total_df = pd.merge(Udata_df,Uitem_df,on = "item_id") 
Total_df = pd.merge(Total_df,Uuser_df,on = "user_id")
SortByUser=Total_df.sort_values(by = ["user_id"])


# Modelling
df = Total_df
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# classifying the movie according to the type
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')

X_train,X_test = train_test_split(df,test_size=0.2)

# Create Data matrix
train_data_matrix = np.zeros((n_users,n_items))
for row in X_train.itertuples():
	train_data_matrix[row[1]-1,row[2]-1] = row[3]
for row in X_test.itertuples():
	train_data_matrix[row[1]-1,row[2]-1] = row[3]

train_data_matrix1 = np.zeros((n_users,n_items))
for row in df.itertuples():
	train_data_matrix1[row[1]-1,row[2]-1] = row[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in X_test.itertuples():
	test_data_matrix[line[1]-1, line[2]-1] = line[3]

# Similarity
user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
#user_similarity = pairwise_distances(train_data_matrix, metric = "euclidean")
item_similarity = cosine_similarity(train_data_matrix.T, dense_output=True)


###############################
#							  #
#	Predict Movie Score Part  #
# 						      #
############################### 

# get similarity of testUser with allUser
def get_similarity(testUser, allUser):
	return pairwise_distances(testUser,allUser, metric = "cosine")

# get matrix of topK similarity User
def get_topK(matrix,similarity,k):
	similarity = similarity[0]
	topK_data_matrix = []
	i = len(similarity)
	for j in range(i):
		# 有问题
		arr = similarity.argsort()[-k:]
		arr_index = arr
	for m in arr_index:
		topK_data_matrix.append(matrix[m])
	# top k mean similarity
	topK_data_matrix = np.asarray(topK_data_matrix)
	return topK_data_matrix

# Through User based to predict score
# The function and formula with previous one is different
def user_base_predict(testUser, topKUser):
	# similarity again:
	sim = pairwise_distances(testUser,topKUser, metric = "cosine")
	sim2 = pairwise_distances(testUser,topKUser, metric = "cosine")
	#print(sim)
	for i in range(len(sim)):
		for j in range(len(sim[0])):
			sim[i][j] = 1/(sim[i][j]+1)
	sim_avg = sim.mean(axis = 1)
	pred = sim_avg * (np.dot(sim2,topKUser))

	return pred

def user_base_predict2(testUser, topKUser):
	r1 = topKUser.mean(axis =1)

	sim = pairwise_distances(testUser,topKUser, metric = "cosine")
	sim2 = pairwise_distances(testUser,topKUser, metric = "cosine")
	for i in range(len(sim)):
		for j in range(len(sim[0])):
			sim[i][j] = 1/(sim[i][j]+1)
	sim_avg = sim.mean(axis = 1)

	r2 = sim_avg* (np.dot(sim2,topKUser))
	diff = topKUser - r1[:,np.newaxis]
	pred = r1[:,np.newaxis] + sim_avg* (np.dot(sim2,diff))
	
	return pred 
	
	
# predict all user's score
def predict_all(train_data_matrix,topK):
	predict = []
	for i in range(len(train_data_matrix)):
		testUser = [train_data_matrix[i]]
		if i == 0:
			allUser = train_data_matrix[i+1:]
		elif i == (len(train_data_matrix) -1):
			allUser = train_data_matrix[:i]
		else:
			allUp = train_data_matrix[:i]
			allDown = train_data_matrix[i+1:]
			allUser = np.concatenate((allUp,allDown))
		s = get_similarity(testUser,allUser)
		topKUser = get_topK(train_data_matrix,s,topK)
		prediction = user_base_predict(testUser,topKUser)
		predict.append(prediction)
	
	return np.asarray(predict)

y_predict = predict_all(train_data_matrix,10)

def predict_userMovieScore(predictall, userID):
	return predictall[userID-1]
	
## Useing MSE to test the result:
#from sklearn.metrics import mean_squared_error
#y_true = train_data_matrix
#y_predict = np.squeeze(y_predict, axis=1)
#mean_squared_error(y_true, y_predict)




# RUN: if we want to predict the 1st user's score:
predict_userMovieScore(y_predict,1)



