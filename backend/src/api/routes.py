from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import get_current_token
from core.pipeline import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    lang: Optional[str] = "fr"


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    forms: List[str]
    sources: List[str]
    meta: Optional[Dict[str, Any]] = None


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query_ircc(req: QueryRequest, token: Optional[str] = Depends(get_current_token)) -> QueryResponse:
    try:
        result = answer_question(req.question, lang=req.lang or "fr")
        return QueryResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during /query: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur interne")
