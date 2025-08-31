import unittest
import pandas as pd
from src.data.ncf_data import create_id_mappings, NetflixDataset

class TestData(unittest.TestCase):
    def setUp(self):
        data = {'userId': [1, 2], 'movieId': [10, 20], 'rating': [5, 3]}
        self.df = pd.DataFrame(data)
        self.user2idx, self.item2idx, _, _ = create_id_mappings(self.df)

    def test_create_mappings(self):
        self.assertEqual(len(self.user2idx), 2)
        self.assertEqual(len(self.item2idx), 2)

    def test_dataset_length(self):
        dataset = NetflixDataset(self.df, self.user2idx, self.item2idx)
        self.assertEqual(len(dataset), 2)

if __name__ == "__main__":
    unittest.main()
