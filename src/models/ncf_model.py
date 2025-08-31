import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_layers=[64,32], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        layers = []
        input_size = embedding_dim * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = h
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)
        
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp(x)
        out = self.output_layer(x)
        return out.squeeze()
