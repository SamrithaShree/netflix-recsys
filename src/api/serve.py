from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd
import torch
from src.models.mf_model import SVDRecommender
from src.models.ncf_model import NCF
from src.data.ncf_data import create_id_mappings
import traceback

app = FastAPI(title="Netflix Recommender API")

print("Loading models and mappings...")

try:
    loaded_svd = SVDRecommender.load_model("model_artifacts/svd_model.pkl")
    print("Surprise SVD model loaded.")
except Exception as e:
    loaded_svd = None
    print("MF model load failed:", e)
    traceback.print_exc()

mf_model = SVDRecommender()
mf_model.model = loaded_svd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ncf_model = None
user2idx = None
item2idx = None
train_df = None

try:
    train_df = pd.read_csv("data/sample/train.csv")
    user2idx, item2idx, _, _ = create_id_mappings(train_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    ncf_model = NCF(n_users, n_items, embedding_dim=64, hidden_layers=[128, 64, 32])
    ncf_model.load_state_dict(torch.load("model_artifacts/ncf_best.pth", map_location=device))
    ncf_model.to(device)
    ncf_model.eval()
    print("NCF model loaded.")
except Exception as e:
    print("NCF model load failed:", e)
    traceback.print_exc()

class RecommendationRequest(BaseModel):
    user_id: int
    model_type: str = "mf"  # "mf" or "ncf"
    top_n: int = 10

@app.get("/")
def read_root():
    return {"message": "Welcome to Netflix Recommender API"}

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    try:
        user = req.user_id
        model_type = req.model_type.lower()
        n = req.top_n

        if model_type == "mf":
            if mf_model is None or mf_model.model is None:
                raise HTTPException(status_code=500, detail="MF Model not loaded")
            recs = mf_model.recommend(user, train_df, n)
            recs = [int(x) for x in recs]
            return {"user_id": user, "recommendations": recs}

        elif model_type == "ncf":
            if ncf_model is None or user2idx is None or item2idx is None:
                raise HTTPException(status_code=500, detail="NCF Model or mappings not loaded")
            if user not in user2idx:
                return {"user_id": user, "recommendations": []}
            all_items_set = set(item2idx.values())
            rated_items = set(train_df[train_df['userId'] == user]['movieId'].map(item2idx))
            to_predict = list(all_items_set - rated_items)
            if not to_predict:
                return {"user_id": user, "recommendations": []}
            user_tensor = torch.tensor([user2idx[user]] * len(to_predict), dtype=torch.long).to(device)
            items_tensor = torch.tensor(to_predict, dtype=torch.long).to(device)
            with torch.no_grad():
                scores = ncf_model(user_tensor, items_tensor)
            top_indices = torch.topk(scores, n).indices
            inv_item2idx = {v: k for k, v in item2idx.items()}
            top_items = [int(inv_item2idx[to_predict[i.item()]]) for i in top_indices]
            return {"user_id": user, "recommendations": top_items}

        else:
            raise HTTPException(status_code=400, detail="Invalid model_type; choose 'mf' or 'ncf'.")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
