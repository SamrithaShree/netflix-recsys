# Netflix Recommender System 

A personalized movie recommendation system built with the Netflix Kaggle dataset.  
Features collaborative filtering (SVD/ALS), baseline models, evaluation with ranking metrics, and deployment via FastAPI + Docker.

## Quick Start
```bash
git clone https://github.com/<your-username>/netflix-recsys.git
cd netflix-recsys
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt