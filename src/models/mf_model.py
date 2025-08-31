from surprise import SVD
import pickle

class SVDRecommender:
    def __init__(self, model=None, model_path=None):
        if model_path:
            self.model = self.load_model(model_path)
        elif model:
            self.model = model
        else:
            self.model = None

    def fit(self, trainset, n_factors=20, n_epochs=20, reg_all=0.1, lr_all=0.005):
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, reg_all=reg_all, lr_all=lr_all)
        algo.fit(trainset)
        self.model = algo
        return algo

    def save_model(self, filepath="model_artifacts/svd_model.pkl"):
        if self.model is not None:
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath="../model_artifacts/svd_model.pkl"):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

    def recommend(self, userId, train_df, n=10):
        all_items = set(train_df["movieId"].unique())
        rated_items = set(train_df[train_df["userId"] == userId]["movieId"])
        items_to_predict = all_items - rated_items
        predictions = [(iid, self.model.predict(userId, iid).est) for iid in items_to_predict]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in predictions[:n]]
