from app.ingestion.parser import parse_transcript


def test_parses_speaker_and_timestamp():
    raw = "[00:00:04] Priya: Morning everyone.\n[00:00:19] Daniel: Quick context here."
    turns = parse_transcript(raw)
    assert len(turns) == 2
    assert turns[0].speaker == "Priya"
    assert turns[0].timestamp == "00:00:04"
    assert turns[0].text == "Morning everyone."
    assert turns[1].index == 1


def test_multiline_turn_is_merged():
    raw = "[00:00:04] Priya: First sentence.\ncontinued on next line.\n[00:00:19] Daniel: Second."
    turns = parse_transcript(raw)
    assert len(turns) == 2
    assert turns[0].text == "First sentence. continued on next line."


def test_leading_preamble_dropped():
    raw = "Meeting recording start\n[00:00:04] Priya: Hello."
    turns = parse_transcript(raw)
    assert len(turns) == 1
    assert turns[0].speaker == "Priya"


def test_untimed_format_falls_back():
    # A transcript with speaker labels but no timestamps must not silently
    # produce zero turns.
    raw = "Alice: Let's ship it.\nBob: Agreed, next week."
    turns = parse_transcript(raw)
    assert [t.speaker for t in turns] == ["Alice", "Bob"]
    assert turns[0].timestamp  # synthesised, non-empty


def test_untimed_fallback_does_not_hijack_timestamped_continuations():
    # A colon inside a continuation of a timestamped turn must stay part of that
    # turn, not be read as a new speaker.
    raw = "[00:00:04] Priya: Here's the plan.\nNote: ship on Friday."
    turns = parse_transcript(raw)
    assert len(turns) == 1
    assert "Note: ship on Friday." in turns[0].text
