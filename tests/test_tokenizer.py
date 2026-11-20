from tokenizer import CharTokenizer


def test_roundtrip():
    text = "To be, or not to be.\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3
