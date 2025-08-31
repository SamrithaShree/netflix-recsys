import unittest
from src.utils.metrics import precision_at_k, recall_at_k, ndcg_at_k

class TestMetrics(unittest.TestCase):
    def test_precision_recall_ndcg(self):
        recommended = [1, 2, 3, 4, 5]
        ground_truth = [2, 4, 6]
        self.assertAlmostEqual(precision_at_k(recommended, ground_truth, 3), 1/3)
        self.assertAlmostEqual(recall_at_k(recommended, ground_truth, 3), 1/3)
        self.assertGreater(ndcg_at_k(recommended, ground_truth, 5), 0)

if __name__ == "__main__":
    unittest.main()
