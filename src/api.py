"""Integration - one FastAPI app, four endpoints, JSON contracts shared by all tracks.
Stubs first; swap in the real models later."""

from fastapi import FastAPI
from pydantic import BaseModel
from router import risk_score, daily_capacity, route_day
import pandas as pd

app = FastAPI(title="ReturnLens")

SCORER = None  # Track A: set via set_scorer() or the RETURNLENS_ARTIFACTS folder (see score_claim.save_artifacts)


def set_scorer(scorer):
    global SCORER
    SCORER = scorer


@app.on_event("startup")
def _load_scorer():
    import os

    d = os.environ.get("RETURNLENS_ARTIFACTS")
    if d and SCORER is None:
        from score_claim import load_scorer

        set_scorer(load_scorer(d))


class EvidenceOut(BaseModel):  # Track A -> everyone
    claim_id: str
    reuse_risk: float
    evidence_uncertainty: float
    cross_customer_reuse: bool = False
    catalogue_image: bool = False
    matches: list = []


class ClaimOut(BaseModel):  # Track B -> everyone
    claim_id: str
    damage_visible: str
    product_match: str
    consistent: str
    evidence_quality: str
    action: str
    reason: str
    contradiction_risk: float


class ForecastOut(BaseModel):  # Track C -> everyone
    date: str
    p10: float
    p50: float
    p90: float
    inspection_capacity: int


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/evidence", response_model=EvidenceOut)
def evidence(claim_id: str, image_path: str = "", customer_id: str = ""):
    if SCORER is not None and image_path:  # real Track A scorer
        return EvidenceOut(**SCORER.score(image_path, claim_id, customer_id or None))
    return EvidenceOut(claim_id=claim_id, reuse_risk=0.05, evidence_uncertainty=0.1)  # STUB (A)


@app.post("/claim", response_model=ClaimOut)
def claim(claim_id: str, text: str = ""):  # STUB (B)
    return ClaimOut(
        claim_id=claim_id,
        damage_visible="yes",
        product_match="yes",
        consistent="yes",
        evidence_quality="good",
        action="approve",
        reason="stub",
        contradiction_risk=0.05,
    )


@app.get("/forecast", response_model=ForecastOut)
def forecast(date: str = "2026-11-08"):  # STUB (C)
    p50, p90 = 900.0, 1400.0
    return ForecastOut(
        date=date, p10=600, p50=p50, p90=p90, inspection_capacity=daily_capacity(p50, p90)
    )


class RouteIn(BaseModel):
    reuse_risk: float
    contradiction_risk: float
    evidence_uncertainty: float
    capacity: int = 50


@app.post("/route")
def route(x: RouteIn):
    df = pd.DataFrame(
        {"risk": [risk_score(x.reuse_risk, x.contradiction_risk)], "unc": [x.evidence_uncertainty]}
    )
    return {"route": str(route_day(df, x.capacity).iloc[0]), "risk": float(df.risk.iloc[0])}
