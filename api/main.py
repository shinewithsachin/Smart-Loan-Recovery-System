from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from ml.train import train
from ml.score import score
from ml.strategy import apply_strategies
import os

app = FastAPI(title="Smart Loan Recovery API")

DATA_DIR = os.path.join(os.getcwd(), "data")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # save uploaded CSV to data
    content = await file.read()
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(content)
    return {"status": "ok", "inserted": 1}

@app.post("/train")
async def api_train(filename: str = "data/loan-recovery.csv"):
    if not os.path.exists(filename):
        raise HTTPException(status_code=400, detail=f"{filename} not found")
    train(filename)
    return {"status": "trained"}

@app.post("/score")
async def api_score(filename: str = "data/loan-recovery.csv"):
    if not os.path.exists(filename):
        raise HTTPException(status_code=400, detail=f"{filename} not found")
    
    df = pd.read_csv(filename)
    out = score(df)
    out = apply_strategies(out)
    

    res = out[["Borrower_ID", "risk_score", "recommended_action"]].to_dict(orient="records")
    return res
