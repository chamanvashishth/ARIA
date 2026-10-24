import pytest

from aria.tokenizer import ByteTokenizer


def test_ascii_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "ARIA learns code."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unicode_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "नमस्ते ARIA — quantum computing"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_code_and_symbols_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "def f(x): return x**2 >= 0 # test"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_stable() -> None:
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert tokens[0] == tokenizer.BOS_ID
    assert tokens[-1] == tokenizer.EOS_ID
    assert tokenizer.vocab_size() == 260


def test_invalid_token_is_rejected() -> None:
    with pytest.raises(ValueError):
        ByteTokenizer().decode([999])


def test_empty_text_round_trip() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_tokenization_is_deterministic() -> None:
    tokenizer = ByteTokenizer()
    text = "Hinglish: quantum kya hai?"
    assert tokenizer.encode(text) == tokenizer.encode(text)
