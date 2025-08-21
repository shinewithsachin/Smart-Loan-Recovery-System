from pydantic import BaseModel
from typing import List

class IngestResponse(BaseModel):
    status: str
    inserted: int

class ScoreRequest(BaseModel):
    borrower_ids: List[str]

class ScoreResponse(BaseModel):
    Borrower_ID: str
    risk_score: float
    recommended_action: str
