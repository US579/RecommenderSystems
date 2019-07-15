import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')



header = ['user_id','item_id','rating','timestamp']
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=header)
df2 = pd.read_csv('data/ml-100k/u.item', sep='|',
                  encoding='latin-1',usecols=range(5),names=m_cols)
print(df.head())





















