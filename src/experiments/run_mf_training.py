import pandas as pd
from surprise import Reader, Dataset
from src.models.mf_model import SVDRecommender
from src.utils.metrics import batch_eval

def prepare_surprise_data(df):
    reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

train_data = prepare_surprise_data(train)
trainset = train_data.build_full_trainset()

algo = SVDRecommender()
algo.fit(trainset)
algo.save_model()

users = test["userId"].unique()
test_dict = test.groupby("userId")["movieId"].apply(list).to_dict()

top_n = {}
for uid in users:
    top_n[uid] = algo.recommend(uid, train, n=10)

prec, recall, ndcg = batch_eval(users, test_dict, top_n, k=10)
print(f"SVD model: Precision@10={prec:.4f} Recall@10={recall:.4f} NDCG@10={ndcg:.4f}")
