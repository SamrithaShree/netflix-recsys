import torch
from torch.utils.data import Dataset

def create_id_mappings(df, user_col="userId", item_col="movieId"):
    user_ids = df[user_col].unique()
    item_ids = df[item_col].unique()
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}
    idx2item = {idx: iid for iid, idx in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item

class NetflixDataset(Dataset):
    def __init__(self, df, user2idx, item2idx, user_col="userId", item_col="movieId", rating_col="rating"):
        self.users = torch.tensor(df[user_col].map(user2idx).values, dtype=torch.long)
        self.items = torch.tensor(df[item_col].map(item2idx).values, dtype=torch.long)
        self.ratings = torch.tensor(df[rating_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
