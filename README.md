# RecommenderSystems
Recommender system using collaborative filtering





# Background

### content-based
*Content-based approach requires a good amount of information of items’ own features, rather than using users’ interactions and feedbacks.


### Collaborative Filtering(协同过滤)
* Collaborative Filtering, on the other hand, doesn’t need anything else except users’ historical preference on a set of items. Because it’s based on historical data, the core assumption here is that the users who have agreed in the past tend to also agree in the future.
* mainly depend on the history of user
( Core: people perfer the things they get use to it )

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

## How to Run

1. part 1

``` Python
# predict  TOP-12 recommended movies
python3 Recommender.py
```
2。 part 2
```python
python3 evaluate.py
```
Output 3 pictures
They are K-RMSE and K-Precision, Weight-RMSE and Weight-Precision of user base, Weight-RMSE and Weight-Precision of item base

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
To calculate the RMSE and return it
```python
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
```

To calculate precision, recall, coverage and popularity and return them
```python
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
```
Use svd and calculate its RMSE and precision
```python
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
```
Show the effect of K change on RMSE
```python
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
```
Show the effect of weight change on precision and RMSE
```python
def plot_item(weight, precision_item):
    item_rmse = plt.figure()
    e = item_rmse.add_subplot(111)
    f = item_rmse.add_subplot(222)
    e.set(title = 'weight-rmse', xlabel = 'value of weight', ylabel = 'RMSE')
    e.plot(weight, item_y)
    f.set(title = 'weight-precision', xlabel = 'value of weight', ylabel = 'Precision')
    f.plot(weight, precision_item)
    plt.show()
```










