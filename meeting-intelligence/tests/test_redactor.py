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


def test_redaction_does_not_glue_adjacent_words():
    # A real Visa test number (passes Luhn) followed by a word.
    text = "The card 4111 1111 1111 1111 was declined."
    out, counts = redact_pii(text)
    assert "[CREDIT_CARD] was declined" in out  # trailing space preserved
    assert counts["CREDIT_CARD"] == 1


def test_non_card_digit_runs_are_not_redacted():
    # An order id that fails the Luhn check is not a card; leave it intact.
    text = "Order number 12345678901234 shipped."
    out, counts = redact_pii(text)
    assert "12345678901234" in out
    assert "CREDIT_CARD" not in counts


def test_filler_removal_leaves_no_orphan_punctuation():
    assert clean_text("Um, I think we should ship it.") == "I think we should ship it."
