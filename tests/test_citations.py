"""Tests for citations.py — every citation must map to a chunk that was sent."""

from financial_rag.citations import extract_ids, validate_citations
from financial_rag.retrieval import RetrievedChunk


def _chunks(n):
    return [
        RetrievedChunk(
            page_number=i + 10, text=f"chunk {i}", chunk_index=i, source="r.pdf",
            similarity=0.8, vector_similarity=0.8,
        )
        for i in range(n)
    ]


def test_extract_ids_handles_single_adjacent_and_grouped():
    assert extract_ids("A [1]. B [2][3]. C [1, 4].") == [1, 2, 3, 1, 4]


def test_extract_ids_ignores_non_numeric_brackets():
    assert extract_ids("See [Note A] and [source, page 3].") == []


def test_valid_citations_map_to_the_right_chunk_and_page():
    check = validate_citations("Margin rose [2].", _chunks(3))

    assert check.invalid_ids == []
    assert len(check.citations) == 1
    assert check.citations[0]["chunk_index"] == 1
    assert check.citations[0]["page_number"] == 11


def test_out_of_range_citation_is_reported_and_stripped():
    check = validate_citations("Real [1]. Invented [7].", _chunks(2))

    assert check.invalid_ids == [7]
    assert [c["id"] for c in check.citations] == [1]
    assert "[7]" not in check.text
    assert "[1]" in check.text


def test_stripping_a_marker_leaves_no_space_before_punctuation():
    check = validate_citations("Revenue was $412.6M [1]. Guidance is $435M [9].", _chunks(2))
    assert check.text == "Revenue was $412.6M [1]. Guidance is $435M."


def test_mixed_group_keeps_only_valid_ids():
    check = validate_citations("Both [1, 9].", _chunks(2))
    assert "[1]" in check.text
    assert "9" not in check.text


def test_zero_is_never_a_valid_excerpt_id():
    check = validate_citations("Claim [0].", _chunks(2))
    assert check.invalid_ids == [0]
    assert check.citations == []


def test_repeated_citation_is_listed_once():
    check = validate_citations("A [1]. B [1].", _chunks(2))
    assert len(check.citations) == 1


def test_no_chunks_means_every_citation_is_invalid():
    check = validate_citations("Claim [1].", [])
    assert check.invalid_ids == [1]
    assert check.citations == []


def test_full_width_brackets_are_read_as_citations():
    # gpt-oss on Groq cites like this.
    check = validate_citations("Matures in March 2029【1】. Total debt【2】.", _chunks(2))
    assert [c["id"] for c in check.citations] == [1, 2]
    assert check.text == "Matures in March 2029[1]. Total debt[2]."


def test_invalid_full_width_citation_is_still_stripped():
    check = validate_citations("Claim【1】. Made up【7】.", _chunks(1))
    assert check.invalid_ids == [7]
    assert check.text == "Claim[1]. Made up."
