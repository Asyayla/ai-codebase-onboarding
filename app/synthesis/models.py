from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.retrieval.models import RetrievedChunk


@dataclass
class AnswerResult:

    answer: str
    sources: List[RetrievedChunk]