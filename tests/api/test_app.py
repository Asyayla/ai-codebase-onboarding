from __future__ import annotations

from app.api.app import create_app


def test_create_app_returns_an_app():
    app = create_app()

    assert app is not None


def test_create_app_does_not_require_openai_api_key_or_database_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    app = create_app()

    assert app is not None


def test_create_app_with_no_injected_services_does_not_build_production_services_eagerly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    app = create_app()

    assert app.state.query_service is None
    assert app.state.answer_service is None


def test_create_app_stores_injected_services_without_building_production_ones():
    class _Sentinel:
        pass

    injected_query_service = _Sentinel()
    injected_answer_service = _Sentinel()

    app = create_app(query_service=injected_query_service, answer_service=injected_answer_service)

    assert app.state.query_service is injected_query_service
    assert app.state.answer_service is injected_answer_service
