"""Unit tests for the SLM tokenizer contract."""

from aria.slm import Tokenizer


class DummyTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, tokens: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in tokens)


def test_tokenizer_contract_accepts_matching_implementation() -> None:
    tokenizer = DummyTokenizer()

    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.encode("hi") == (104, 105)
    assert tokenizer.decode((104, 105)) == "hi"


def test_tokenizer_contract_rejects_incomplete_implementation() -> None:
    class IncompleteTokenizer:
        def encode(self, text: str) -> tuple[int, ...]:
            return tuple(ord(char) for char in text)

    assert not isinstance(IncompleteTokenizer(), Tokenizer)
