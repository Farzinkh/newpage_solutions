from app.ingestion.redactor import clean_text, preprocess, redact_pii


def test_email_and_phone_redacted():
    text = "Reach me at maya.roberts@example.com or call 415-555-0142."
    out, counts = redact_pii(text)
    assert "[EMAIL]" in out and "[PHONE]" in out
    assert "example.com" not in out
    assert counts["EMAIL"] == 1 and counts["PHONE"] == 1


def test_speaker_names_are_preserved():
    # Names are identity, not leaked PII: they must survive redaction.
    text = "Daniel owns the offline design doc and Maya reviews it."
    out, counts = redact_pii(text)
    assert "Daniel" in out and "Maya" in out
    assert counts == {}


def test_stopwords_are_not_removed():
    # 'not' is meaning-bearing for retrieval; cleaning must never drop it.
    text = "I did not approve the budget."
    assert "not" in clean_text(text)


def test_artifacts_stripped_but_words_kept():
    text = "So um the [inaudible] plan is uh solid."
    out = clean_text(text)
    assert "[inaudible]" not in out
    assert "plan" in out and "solid" in out


def test_preprocess_can_disable_redaction():
    text = "call 415-555-0142"
    out, counts = preprocess(text, redact=False)
    assert "415-555-0142" in out and counts == {}
