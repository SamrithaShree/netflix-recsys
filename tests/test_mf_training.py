import unittest
import pandas as pd
from surprise import Reader, Dataset
from src.models.mf_model import SVDRecommender

class TestMFTraining(unittest.TestCase):
    def setUp(self):
        data = {'userId': [1, 2, 1], 'movieId': [10, 20, 30], 'rating': [5, 3, 4]}
        self.df = pd.DataFrame(data)
        reader = Reader(rating_scale=(1, 5))
        self.surprise_data = Dataset.load_from_df(self.df[['userId', 'movieId', 'rating']], reader)

    def test_train_save_load(self):
        trainset = self.surprise_data.build_full_trainset()
        model = SVDRecommender()
        algo = model.fit(trainset)
        model.save_model("svd_test.pkl")
        loaded = model.load_model("svd_test.pkl")
        self.assertIsNotNone(loaded)

if __name__ == "__main__":
    unittest.main()
