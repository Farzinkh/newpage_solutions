from app.ingestion.chunker import chunk_turns
from app.models import Turn


def _turns():
    return [
        Turn(index=0, speaker="A", timestamp="00:00:01", text="Short turn."),
        Turn(index=1, speaker="B", timestamp="00:00:05", text="Another turn here."),
    ]


def test_one_chunk_per_short_turn():
    chunks = chunk_turns("m1", _turns(), max_chunk_chars=1200)
    assert len(chunks) == 2
    assert chunks[0].speaker == "A"
    assert chunks[0].timestamp == "00:00:01"


def test_chunk_ids_are_stable_and_idempotent():
    a = chunk_turns("m1", _turns(), max_chunk_chars=1200)
    b = chunk_turns("m1", _turns(), max_chunk_chars=1200)
    assert [c.id for c in a] == [c.id for c in b]


def test_long_turn_is_split_on_sentences():
    long = Turn(index=0, speaker="A", timestamp="00:00:01",
                text="One. " * 200)  # ~1000+ chars
    chunks = chunk_turns("m1", [long], max_chunk_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)  # small overshoot tolerance


def test_single_over_long_sentence_is_hard_wrapped():
    # One unpunctuated monologue must not become a single oversized chunk that
    # silently overruns the embedding model's token window.
    monologue = Turn(index=0, speaker="A", timestamp="00:00:01",
                     text="word " * 300)  # ~1500 chars, no sentence breaks
    chunks = chunk_turns("m1", [monologue], max_chunk_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
