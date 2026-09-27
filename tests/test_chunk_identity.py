import re

from app.embedding.chunk_identity import compute_chunk_id

_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_BASE_ARGS = dict(
    repository_id="repo-1",
    file_path="app/foo.py",
    symbol_type="function",
    symbol_name="do_thing",
    start_line=10,
    end_line=20,
)


def _id(**overrides) -> str:
    args = {**_BASE_ARGS, **overrides}
    return compute_chunk_id(**args)


def test_same_inputs_produce_the_same_id():
    first = _id()
    second = _id()

    assert first == second


def test_output_is_a_valid_stable_hex_sha256_digest():
    result = _id()

    assert isinstance(result, str)
    assert len(result) == 64
    assert _HEX_SHA256_PATTERN.match(
        result
    ), f"not a lowercase hex sha256 digest: {result!r}"


def test_changing_repository_id_changes_the_id():
    assert _id() != _id(repository_id="repo-2")


def test_changing_file_path_changes_the_id():
    assert _id() != _id(file_path="app/bar.py")


def test_changing_symbol_type_changes_the_id():
    assert _id() != _id(symbol_type="method")


def test_changing_symbol_name_changes_the_id():
    assert _id() != _id(symbol_name="do_other_thing")


def test_changing_start_line_changes_the_id():
    assert _id() != _id(start_line=11)


def test_changing_end_line_changes_the_id():
    assert _id() != _id(end_line=21)


def test_content_is_not_an_input_and_therefore_cannot_affect_the_id():
    same_identity_id_one = _id()
    same_identity_id_two = _id()

    assert same_identity_id_one == same_identity_id_two

    import inspect

    signature = inspect.signature(compute_chunk_id)
    assert "content" not in signature.parameters


def test_different_chunks_produce_different_ids():
    chunk_a = compute_chunk_id(
        repository_id="repo-1",
        file_path="app/services/order_service.py",
        symbol_type="class",
        symbol_name="OrderService",
        start_line=1,
        end_line=40,
    )
    chunk_b = compute_chunk_id(
        repository_id="repo-1",
        file_path="app/services/order_service.py",
        symbol_type="method",
        symbol_name="OrderService.calculate_total",
        start_line=10,
        end_line=15,
    )
    chunk_c = compute_chunk_id(
        repository_id="repo-2",
        file_path="app/services/order_service.py",
        symbol_type="class",
        symbol_name="OrderService",
        start_line=1,
        end_line=40,
    )

    assert len({chunk_a, chunk_b, chunk_c}) == 3


def test_field_boundary_shift_does_not_collide():
    # Length-prefixing keeps different field boundaries distinguishable.
    id_one = compute_chunk_id(
        repository_id="repo",
        file_path="a|b",
        symbol_type="function",
        symbol_name="c",
        start_line=1,
        end_line=2,
    )
    id_two = compute_chunk_id(
        repository_id="repo",
        file_path="a",
        symbol_type="function",
        symbol_name="b|c",
        start_line=1,
        end_line=2,
    )

    assert id_one != id_two


def test_multiple_field_changes_still_change_the_id_independently():
    baseline = _id()
    changed = _id(repository_id="repo-2", symbol_name="different_name")

    assert baseline != changed