# RecommenderSystems
Recommender system using collaborative filtering





# Background

### content-based
*Content-based approach requires a good amount of information of items’ own features, rather than using users’ interactions and feedbacks.


### Collaborative Filtering(协同过滤)
* Collaborative Filtering, on the other hand, doesn’t need anything else except users’ historical preference on a set of items. Because it’s based on historical data, the core assumption here is that the users who have agreed in the past tend to also agree in the future.
* 主要根据用户的历史信息
( Core: 这个用户以前喜欢什么现在也会喜欢什么  which means 很专一 )

#### This can be expressed by two categories:
	- Explicit Rating
	   1. 5 stars for Titanic, explicitly show how much they like this kinda movie
	- Implicit Rating
	   2. Suggests users preference indirectly, such as page views, clicks, purchase records, whether or not listen to a music track, and so on.

### The standard method of Collaborative Filtering 

#### KNN

	- user-based CF 
	- item-based CF. 

1. Initialisation

	- build n × m matrix that store the relation between user and ratings 

- Parameter breakdown:
```
n × m matrix of ratings

u_i : represent users
p_j : represent item
```

2. Caculation and predication

* User-based CF

The main process is to caculate the similarity between `target user` and `all other users`, select the top X similar users,and take the weighted average of ratings from these X users with similarity as weights
<div align=center><img width="250" height="75" src="https://github.com/US579/RecommenderSystems/blob/master/image/formula1.png"/></div>

while different people have different baseline when giving ratings,some generally give full scores but some pretty strict. so,to aviod bias,we can substact `each user's average rating of all item-based` when caculating weighted average,and add it back to the target user as below:
<div align=center><img width="250" height="75" src="https://github.com/US579/RecommenderSystems/blob/master/image/formula2.png"/></div>

there are two ways to caculate similarity:
<div align=center><img width="390" height="225" src="https://github.com/US579/RecommenderSystems/blob/master/image/formula3.png"/></div>

# baseline

### 1. initialize the feature matrix

``` python
for row in X_train.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]
for row in x_test.itertuples():
    train_data_matrix[row[1]-1,row[2]-1] = row[3]

```

In our project we using cosine similarity as distance function 

```python
user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric = "cosine")
```

to predicte top-K movie

```python 
def KNN(items, ratings, item_similarity, keywords, k):
    movie_list = [] 
    movie_id = list(items[items['title'].str.contains(keywords)].item_id)[0] 
    movie_similarity = item_similarity[movie_id - 1]
    movie_similarity_index = np.argsort(-movie_similarity)[1:k + 1]
    for index in movie_similarity_index:
        movie_list = list(set(list(items[items['item_id'] == index + 1].title)))
        movie_list.append(movie_similarity[index])  
        movie_list.append(ratings[ratings['item_id'] == index + 1].rating.mean()) 
        movie_list.append(move_list)
    return movie_list
 ```

### 2 Prediction

to perdicte the score according to the user base and item base

```python
def predict(rating, similarity, base = 'user'):
    if base == 'user':
        mean_user_rating = rating.mean(axis = 1)
        rating_diff = (rating - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif base == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def predict_rate(rating, similarity):
    pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
```


### 3. Evaluation 



