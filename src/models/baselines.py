import numpy as np
import pandas as pd

class PopularityBaseline:
    def __init__(self, method="count"):
        self.method = method
        self.popularity = None

    def fit(self, train_df):
        if self.method == "count":
            self.popularity = train_df.groupby("movieId")["userId"].count()
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")
        self.popularity = self.popularity.sort_values(ascending=False)

    def recommend(self, n=10):
        return list(self.popularity.index[:n])

class RandomBaseline:
    def __init__(self, seed=None):
        self.seed = seed
        self.movies = None
        if seed is not None:
            np.random.seed(seed)

    def fit(self, train_df):
        self.movies = train_df["movieId"].unique()

    def recommend(self, userId, n=10):
        return list(np.random.choice(self.movies, size=n, replace=False))
