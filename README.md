# RecommenderSystems
Recommender system using collaborative filtering





# Background

### Collaborative Filtering(协同过滤)
* Collaborative Filtering, on the other hand, doesn’t need anything else except users’ historical preference on a set of items. Because it’s based on historical data, the core assumption here is that the users who have agreed in the past tend to also agree in the future.

* 主要根据用户的历史信息
( Core: 这个用户以前喜欢什么现在也会喜欢什么  which means 很专一 )

#### This can be expressed by two categories:
	- Explicit Rating
	   1. 5 stars for Titanic, explicitly show how much they like this kinda movie
	- Implicit Rating
	   2. Suggests users preference indirectly, such as page views, clicks, purchase records, whether or not listen to a music track, and so on.

### The standard method of Collaborative Filtering is known as `Nearest Neighborhood algorithm`. 
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

the main process is to caculate the similarity between `target user` and `all other users`, select the top X similar users,and take the weighted average of ratings from these X users with similarity as weights