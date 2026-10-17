from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

_TEMPORARY_STATUS_CODES = {429, 500, 502, 503, 504}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(genai_errors.APIError)
    async def handle_gemini_error(
        request: Request, exc: genai_errors.APIError
    ) -> JSONResponse:
        logger.error("Gemini API error: %s", exc)
        if exc.code in _TEMPORARY_STATUS_CODES:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "The language model service is temporarily unavailable. "
                    "Please try again in a moment."
                },
            )
        return JSONResponse(
            status_code=502,
            content={"detail": "The language model service rejected the request."},
        )