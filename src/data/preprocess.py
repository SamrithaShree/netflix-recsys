import pandas as pd
import os

def load_ratings(ratings_path="data/raw/ratings.dat"):
    return pd.read_csv(
        ratings_path, sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

def load_movies(movies_path="data/raw/movies.dat"):
    return pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1"
    )

def filter_users_items(df, min_ratings_user=20, min_ratings_movie=20):
    df = df.groupby("userId").filter(lambda x: len(x) >= min_ratings_user)
    df = df.groupby("movieId").filter(lambda x: len(x) >= min_ratings_movie)
    return df

def train_test_split_by_time(df, test_ratio=0.2, user_col="userId", time_col="timestamp"):
    train_frames, test_frames = [], []
    for uid, group in df.groupby(user_col):
        group = group.sort_values(time_col)
        cutoff = int((1.0 - test_ratio) * len(group))
        train_frames.append(group.iloc[:cutoff])
        test_frames.append(group.iloc[cutoff:])
    train = pd.concat(train_frames).reset_index(drop=True)
    test = pd.concat(test_frames).reset_index(drop=True)
    return train, test

def preprocess(
    ratings_path="data/raw/ratings.dat",
    movies_path="data/raw/movies.dat",
    min_ratings_user=20,
    min_ratings_movie=20,
    processed_out="data/processed/processed.csv",
    train_out="data/processed/train.csv",
    test_out="data/processed/test.csv",
    test_ratio=0.2
):
    ratings = load_ratings(ratings_path)
    movies = load_movies(movies_path)
    df = ratings.merge(movies, on="movieId")
    df = filter_users_items(df, min_ratings_user, min_ratings_movie)

    os.makedirs(os.path.dirname(processed_out), exist_ok=True)
    df.to_csv(processed_out, index=False)
    print(f"Processed data saved at {processed_out}")

    train, test = train_test_split_by_time(df, test_ratio=test_ratio, user_col="userId", time_col="timestamp")
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    print(f"Train and test data saved at {train_out}, {test_out}")

if __name__ == "__main__":
    preprocess()
