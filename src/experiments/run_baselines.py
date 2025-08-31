import pandas as pd
from src.models.baselines import PopularityBaseline, RandomBaseline
from src.utils.metrics import batch_eval

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

test_dict = test.groupby("userId")["movieId"].apply(list).to_dict()
users = list(test_dict.keys())

pop_model = PopularityBaseline()
pop_model.fit(train)
pop_recs = {uid: pop_model.recommend(n=10) for uid in users}

rand_model = RandomBaseline(seed=42)
rand_model.fit(train)
rand_recs = {uid: rand_model.recommend(userId=uid, n=10) for uid in users}

for name, recs in [("Popularity", pop_recs), ("Random", rand_recs)]:
    prec, recall, ndcg = batch_eval(users, test_dict, recs, k=10)
    print(f"{name} baseline: Precision@10={prec:.4f} Recall@10={recall:.4f} NDCG@10={ndcg:.4f}")
