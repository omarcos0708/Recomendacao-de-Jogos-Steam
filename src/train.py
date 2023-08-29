import pickle
import numpy as np
import pandas as pd
from utils import get_steam_data, get_rattings, ItemBasedRecommender

# Get data

df = get_steam_data(r'assets\steam-200k.csv')

#get implicit ratings
df_ratings = get_rattings(df)

# instantiate recommender

recommender = ItemBasedRecommender(
    data = df_ratings,
    item_col = 'item_id',
    user_col = 'user_id',
    score_col = 'rating',
    aggfunc = np.sum
)

# train recomender
recommender.fit()

# save recommender

with open('models/recommender.pkl', "wb") as model_file:
    pickle.dump(recommender, model_file)