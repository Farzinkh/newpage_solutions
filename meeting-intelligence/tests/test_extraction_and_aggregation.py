from app.generation.aggregate import answer_from_items, detect_intent
from app.ingestion.extractor import extract_items
from app.ingestion.parser import parse_transcript

TRANSCRIPT = (
    "[00:01:22] Priya: Decision: offline mode ships in Q3, single-device only.\n"
    "[00:01:40] Daniel: I'll write the design doc by Friday and tag Maya.\n"
    "[00:02:38] Maya: I can join the rotation starting next sprint.\n"
    "[00:03:04] Priya: Great, thanks everyone.\n"
)


def _items():
    return extract_items("m1", parse_transcript(TRANSCRIPT))


def test_extractor_tags_decisions_and_action_items():
    items = _items()
    kinds = {i.turn_index: i.kind for i in items}
    assert kinds[0] == "decision"       # "Decision: ..."
    assert kinds[1] == "action_item"    # "I'll ... by Friday"
    assert kinds[2] == "action_item"    # "I can join ..."
    assert 3 not in kinds               # a plain thank-you is neither


def test_extracted_items_keep_speaker_and_timestamp():
    doc = next(i for i in _items() if i.turn_index == 1)
    assert doc.speaker == "Daniel"
    assert doc.timestamp == "00:01:40"


def test_detect_intent_routes_aggregation_queries():
    assert detect_intent("What are all the action items?") == "action_item"
    assert detect_intent("List the decisions we made") == "decision"
    assert detect_intent("Summarise the meeting") == "summary"
    # A specific factual question should fall through to normal retrieval.
    assert detect_intent("Why did we change the on-call rotation?") is None


def test_answer_from_items_is_complete_and_cited():
    ans = answer_from_items("action_item", _items())
    assert ans.grounded
    # Both action items are present — not just the top-k most similar.
    assert "design doc" in ans.text and "rotation" in ans.text
    assert len(ans.citations) == 2


def test_summary_combines_decisions_and_actions():
    ans = answer_from_items("summary", _items())
    assert "Decisions:" in ans.text and "Action items:" in ans.text
