from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from core.fingerprint import build_fingerprint_from_pitch
from core.similarity import calculate_similarity

app = FastAPI(title="Mantra Similarity Engine")


class CompareRequest(BaseModel):
    melody_a: List[int]
    melody_b: List[int]


class CompareResponse(BaseModel):
    similarity: float


@app.get("/")
def root():
    return {"status": "Mantra similarity engine is running"}


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    f1 = build_fingerprint_from_pitch(request.melody_a)
    f2 = build_fingerprint_from_pitch(request.melody_b)

    score = calculate_similarity(f1, f2)

    return CompareResponse(similarity=round(score, 4))