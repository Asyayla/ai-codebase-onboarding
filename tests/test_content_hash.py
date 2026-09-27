import hashlib
import re

from app.persistence.content_hash import compute_content_hash

_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_deterministic_for_the_same_content():
    first = compute_content_hash("def foo(): pass")
    second = compute_content_hash("def foo(): pass")

    assert first == second


def test_same_content_produces_same_hash_across_independent_calls():
    text = "class Widget:\n    def build(self):\n        return 1\n"

    assert compute_content_hash(text) == compute_content_hash(text)


def test_different_content_produces_different_hash():
    first = compute_content_hash("def foo(): pass")
    second = compute_content_hash("def bar(): pass")

    assert first != second


def test_empty_string_works():
    result = compute_content_hash("")

    assert isinstance(result, str)
    assert _HEX_SHA256_PATTERN.match(result)


def test_result_is_a_valid_hex_sha256_digest():
    result = compute_content_hash("some content")

    assert len(result) == 64
    assert _HEX_SHA256_PATTERN.match(result)


def test_matches_exact_sha256_of_utf8_encoded_content():
    content = "some content with unicode: café"

    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert compute_content_hash(content) == expected


def test_content_is_not_normalized_before_hashing():
    # Whitespace and casing differences must remain hash-significant.
    assert compute_content_hash("foo") != compute_content_hash("Foo")
    assert compute_content_hash("foo") != compute_content_hash(" foo")
    assert compute_content_hash("foo") != compute_content_hash("foo\n")