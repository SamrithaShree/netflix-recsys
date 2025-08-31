import unittest
import pandas as pd
import torch
from src.data.ncf_data import create_id_mappings, NetflixDataset
from src.models.ncf_model import NCF

class TestNCFTraining(unittest.TestCase):
    def setUp(self):
        # Minimal synthetic data for testing
        data = {
            'userId': [1, 2, 1, 3],
            'movieId': [10, 11, 12, 13],
            'rating': [4.0, 5.0, 3.0, 2.0]
        }
        self.df = pd.DataFrame(data)
        self.user2idx, self.item2idx, _, _ = create_id_mappings(self.df)
        self.dataset = NetflixDataset(self.df, self.user2idx, self.item2idx)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 4)

    def test_model_forward_pass(self):
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        model = NCF(n_users, n_items, embedding_dim=8, hidden_layers=[16, 8])
        model.eval()

        users = torch.tensor([self.user2idx[1], self.user2idx[2]])
        items = torch.tensor([self.item2idx[10], self.item2idx[11]])

        with torch.no_grad():
            outputs = model(users, items)
        self.assertEqual(outputs.shape[0], 2)

    def test_training_step(self):
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        model = NCF(n_users, n_items, embedding_dim=8, hidden_layers=[16, 8])
        model.train()

        users = torch.tensor([self.user2idx[1], self.user2idx[2]])
        items = torch.tensor([self.item2idx[10], self.item2idx[11]])
        ratings = torch.tensor([4.0, 5.0])

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        self.assertIsNotNone(loss.item())

if __name__ == "__main__":
    unittest.main()
