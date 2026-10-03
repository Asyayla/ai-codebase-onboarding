from __future__ import annotations
from openai import OpenAIError
from app.synthesis.answer_service import AnswerService
from tests.fakes.fake_llm_provider import FakeLLMProvider

import importlib
import subprocess
from pathlib import Path

import pytest

import app.composition as composition
from app.pipeline.repository_query_service import RepositoryQueryService
from app.retrieval.vector_store import FakeVectorStore
from tests.fakes.fake_chunk_store import FakeChunkStore
from tests.fakes.fake_embedding_provider import FakeEmbeddingProvider


def _init_local_git_repo(root: Path) -> Path:
    """Create a minimal real local git repository with one Python file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "module.py").write_text("def greet():\n    return 'hello'\n")

    env_git_args = ["-c", "user.email=test@example.com", "-c", "user.name=Test"]
    subprocess.run(["git", *env_git_args, "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", *env_git_args, "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", *env_git_args, "commit", "-m", "initial commit"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def test_build_with_injected_fakes_returns_a_repository_query_service():
    service = composition.build_repository_query_service(
        embedding_provider=FakeEmbeddingProvider(),
        chunk_store=FakeChunkStore(),
        vector_store=FakeVectorStore(),
    )

    assert isinstance(service, RepositoryQueryService)


def test_composition_shares_one_embedding_provider_between_indexing_and_retrieval(
    tmp_path,
):
    repo_path = _init_local_git_repo(tmp_path / "repo")
    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    service = composition.build_repository_query_service(
        embedding_provider=provider,
        chunk_store=FakeChunkStore(),
        vector_store=vector_store,
    )

    service.index_repository(str(repo_path))
    service.ask("some-repository-id", "how does greet work?", top_k=5)

    assert len(provider.received_batches) >= 2
    assert ["how does greet work?"] in provider.received_batches


def test_importing_the_module_requires_no_environment_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    importlib.reload(composition)


def test_default_embedding_provider_reads_config_from_environment_not_hardcoded(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIError, match="api_key"):
        composition.build_repository_query_service(
            chunk_store=FakeChunkStore(),
            vector_store=FakeVectorStore(),
        )


def test_build_answer_service_does_not_call_the_production_query_service_factory_when_injected(
    monkeypatch,
):
    injected_query_service = composition.build_repository_query_service(
        embedding_provider=FakeEmbeddingProvider(),
        chunk_store=FakeChunkStore(),
        vector_store=FakeVectorStore(),
    )
    injected_llm_provider = FakeLLMProvider()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "build_repository_query_service() must not be called "
            "when query_service is explicitly injected"
        )

    monkeypatch.setattr(
        "app.composition.build_repository_query_service",
        _fail_if_called,
    )

    service = composition.build_answer_service(
        query_service=injected_query_service,
        llm_provider=injected_llm_provider,
    )

    assert isinstance(service, AnswerService)