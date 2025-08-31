import unittest
import pandas as pd
from src.models.baselines import PopularityBaseline, RandomBaseline

class TestBaselines(unittest.TestCase):
    def setUp(self):
        data = {'userId': [1, 2, 1, 3], 'movieId': [10, 10, 20, 30], 'rating': [4, 5, 3, 2]}
        self.df = pd.DataFrame(data)

    def test_popularity(self):
        pop = PopularityBaseline()
        pop.fit(self.df)
        recs = pop.recommend(n=2)
        self.assertIn(10, recs)

    def test_random(self):
        rand = RandomBaseline(seed=42)
        rand.fit(self.df)
        recs = rand.recommend(userId=1, n=1)
        self.assertEqual(len(recs), 1)

if __name__ == "__main__":
    unittest.main()
