"""
Unit tests for tests/fakes/fake_chunk_store.py.

All tests are pure/offline - no network, no disk I/O. Tests the observable
ChunkStore contract (save_chunks, get_chunk_hashes, delete_chunks,
delete_repository), plus the fake's test-only inspection helper
(get_stored_chunk), not private implementation details.
"""

from app.embedding.models import EmbeddedChunk
from app.persistence.content_hash import compute_content_hash
from tests.fakes.fake_chunk_store import FakeChunkStore


def _make_embedded_chunk(
    chunk_id: str = "chunk-1",
    file_path: str = "app/foo.py",
    language: str = "python",
    symbol_type: str = "function",
    symbol_name: str = "foo",
    start_line: int = 1,
    end_line: int = 5,
    content: str = "def foo(): pass",
    parent_symbol_name=None,
    docstring=None,
    embedding=None,
    embedding_model: str = "fake-embedding-model",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        repository_id="unused-embedded-chunk-field",  # not read by the store; repository_id is passed separately to save_chunks
        file_path=file_path,
        language=language,
        symbol_type=symbol_type,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        parent_symbol_name=parent_symbol_name,
        docstring=docstring,
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        embedding_model=embedding_model,
    )


def test_save_then_get_chunk_hashes_returns_the_saved_chunk():
    store = FakeChunkStore()
    chunk = _make_embedded_chunk(chunk_id="chunk-1", content="def foo(): pass")

    store.save_chunks("repo-1", [chunk])
    hashes = store.get_chunk_hashes("repo-1")

    assert hashes == {"chunk-1": compute_content_hash("def foo(): pass")}


def test_stored_chunk_contains_expected_embedded_chunk_fields():
    store = FakeChunkStore()
    chunk = _make_embedded_chunk(
        chunk_id="chunk-1",
        file_path="app/services/order_service.py",
        language="python",
        symbol_type="method",
        symbol_name="OrderService.calculate_total",
        start_line=42,
        end_line=57,
        content="def calculate_total(self): ...",
        parent_symbol_name="OrderService",
        docstring="Calculates the order total.",
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test-model",
    )

    store.save_chunks("repo-1", [chunk])
    stored = store.get_stored_chunk("repo-1", "chunk-1")

    assert stored is not None
    assert stored.chunk_id == "chunk-1"
    assert stored.repository_id == "repo-1"
    assert stored.file_path == chunk.file_path
    assert stored.language == chunk.language
    assert stored.symbol_type == chunk.symbol_type
    assert stored.symbol_name == chunk.symbol_name
    assert stored.start_line == chunk.start_line
    assert stored.end_line == chunk.end_line
    assert stored.content == chunk.content
    assert stored.parent_symbol_name == chunk.parent_symbol_name
    assert stored.docstring == chunk.docstring
    assert stored.embedding == chunk.embedding
    assert stored.embedding_model == chunk.embedding_model


def test_content_hash_is_computed_correctly_on_save():
    store = FakeChunkStore()
    chunk = _make_embedded_chunk(chunk_id="chunk-1", content="some specific content")

    store.save_chunks("repo-1", [chunk])
    stored = store.get_stored_chunk("repo-1", "chunk-1")

    assert stored.content_hash == compute_content_hash("some specific content")


def test_saving_an_existing_chunk_id_replaces_the_stored_record():
    store = FakeChunkStore()
    original = _make_embedded_chunk(chunk_id="chunk-1", content="original content")
    updated = _make_embedded_chunk(chunk_id="chunk-1", content="updated content")

    store.save_chunks("repo-1", [original])
    store.save_chunks("repo-1", [updated])

    stored = store.get_stored_chunk("repo-1", "chunk-1")
    assert stored.content == "updated content"
    # Exactly one record remains for this chunk_id, not two.
    assert len(store.get_chunk_hashes("repo-1")) == 1


def test_changed_content_produces_a_changed_stored_hash():
    store = FakeChunkStore()
    original = _make_embedded_chunk(chunk_id="chunk-1", content="original content")
    updated = _make_embedded_chunk(chunk_id="chunk-1", content="updated content")

    store.save_chunks("repo-1", [original])
    hash_before = store.get_chunk_hashes("repo-1")["chunk-1"]

    store.save_chunks("repo-1", [updated])
    hash_after = store.get_chunk_hashes("repo-1")["chunk-1"]

    assert hash_before != hash_after


def test_delete_chunks_removes_only_the_requested_ids():
    store = FakeChunkStore()
    chunk_a = _make_embedded_chunk(chunk_id="chunk-a")
    chunk_b = _make_embedded_chunk(chunk_id="chunk-b")
    chunk_c = _make_embedded_chunk(chunk_id="chunk-c")

    store.save_chunks("repo-1", [chunk_a, chunk_b, chunk_c])
    store.delete_chunks("repo-1", ["chunk-b"])

    remaining = store.get_chunk_hashes("repo-1")
    assert set(remaining.keys()) == {"chunk-a", "chunk-c"}


def test_deleting_a_nonexistent_chunk_id_is_a_no_op():
    store = FakeChunkStore()
    chunk = _make_embedded_chunk(chunk_id="chunk-1")
    store.save_chunks("repo-1", [chunk])

    store.delete_chunks("repo-1", ["chunk-does-not-exist"])

    assert store.get_chunk_hashes("repo-1") == {"chunk-1": compute_content_hash(chunk.content)}


def test_deleting_from_an_unknown_repository_is_a_no_op():
    store = FakeChunkStore()

    # Should not raise, even though "repo-that-never-existed" was never saved to.
    store.delete_chunks("repo-that-never-existed", ["chunk-1"])

    assert store.get_chunk_hashes("repo-that-never-existed") == {}


def test_delete_repository_removes_all_chunks_for_that_repository():
    store = FakeChunkStore()
    chunk_a = _make_embedded_chunk(chunk_id="chunk-a")
    chunk_b = _make_embedded_chunk(chunk_id="chunk-b")
    store.save_chunks("repo-1", [chunk_a, chunk_b])

    store.delete_repository("repo-1")

    assert store.get_chunk_hashes("repo-1") == {}


def test_unknown_repository_returns_empty_dict():
    store = FakeChunkStore()

    assert store.get_chunk_hashes("never-saved-to") == {}


def test_repositories_are_fully_isolated():
    store = FakeChunkStore()
    chunk_repo_one = _make_embedded_chunk(chunk_id="shared-id", content="repo one content")
    chunk_repo_two = _make_embedded_chunk(chunk_id="shared-id", content="repo two content")

    store.save_chunks("repo-1", [chunk_repo_one])
    store.save_chunks("repo-2", [chunk_repo_two])

    repo_one_stored = store.get_stored_chunk("repo-1", "shared-id")
    repo_two_stored = store.get_stored_chunk("repo-2", "shared-id")

    assert repo_one_stored.content == "repo one content"
    assert repo_two_stored.content == "repo two content"
    assert repo_one_stored.repository_id == "repo-1"
    assert repo_two_stored.repository_id == "repo-2"


def test_deleting_one_repository_does_not_affect_another():
    store = FakeChunkStore()
    store.save_chunks("repo-1", [_make_embedded_chunk(chunk_id="chunk-1")])
    store.save_chunks("repo-2", [_make_embedded_chunk(chunk_id="chunk-1")])

    store.delete_repository("repo-1")

    assert store.get_chunk_hashes("repo-1") == {}
    assert list(store.get_chunk_hashes("repo-2").keys()) == ["chunk-1"]


def test_duplicate_chunk_id_within_one_save_call_uses_last_write_wins():
    store = FakeChunkStore()
    first = _make_embedded_chunk(chunk_id="chunk-1", content="first version")
    second = _make_embedded_chunk(chunk_id="chunk-1", content="second version")

    store.save_chunks("repo-1", [first, second])

    stored = store.get_stored_chunk("repo-1", "chunk-1")
    assert stored.content == "second version"
    assert len(store.get_chunk_hashes("repo-1")) == 1