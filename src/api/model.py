import pickle
import pandas as pd
import torch
import traceback
from src.models.mf_model import SVDRecommender
from src.models.ncf_model import NCF
from src.data.ncf_data import create_id_mappings


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_svd = None
        self.mf_model = None
        self.ncf_model = None
        self.user2idx = None
        self.item2idx = None
        self.train_df = None
        
        self.load_models()

    def load_models(self):
        try:
            self.loaded_svd = SVDRecommender.load_model("model_artifacts/svd_model.pkl")
            print("Surprise SVD model loaded.")
        except Exception as e:
            self.loaded_svd = None
            print("MF model load failed:", e)
            traceback.print_exc()

        self.mf_model = SVDRecommender()
        self.mf_model.model = self.loaded_svd

        try:
            self.train_df = pd.read_csv("data/sample/train.csv")
            self.user2idx, self.item2idx, _, _ = create_id_mappings(self.train_df)
            n_users = len(self.user2idx)
            n_items = len(self.item2idx)
            self.ncf_model = NCF(n_users, n_items, embedding_dim=64, hidden_layers=[128, 64, 32])
            self.ncf_model.load_state_dict(torch.load("model_artifacts/ncf_best.pth", map_location=self.device))
            self.ncf_model.to(self.device)
            self.ncf_model.eval()
            print("NCF model loaded.")
        except Exception as e:
            print("NCF model load failed:", e)
            traceback.print_exc()

    def recommend_mf(self, user, n=10):
        if self.mf_model is None or self.mf_model.model is None or self.train_df is None:
            raise Exception("MF Model or training data not loaded")
        recs = self.mf_model.recommend(user, self.train_df, n)
        return [int(x) for x in recs]

    def recommend_ncf(self, user, n=10):
        if self.ncf_model is None or self.user2idx is None or self.item2idx is None or self.train_df is None:
            raise Exception("NCF Model or mappings not loaded")

        if user not in self.user2idx:
            return []

        all_items_set = set(self.item2idx.values())
        rated_items = set(self.train_df[self.train_df['userId'] == user]['movieId'].map(self.item2idx))
        to_predict = list(all_items_set - rated_items)
        if not to_predict:
            return []

        user_tensor = torch.tensor([self.user2idx[user]] * len(to_predict), dtype=torch.long).to(self.device)
        items_tensor = torch.tensor(to_predict, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.ncf_model(user_tensor, items_tensor)
        top_indices = torch.topk(scores, n).indices

        inv_item2idx = {v: k for k, v in self.item2idx.items()}
        top_items = [int(inv_item2idx[to_predict[i.item()]]) for i in top_indices]
        return top_items
