"""Deterministic byte-level tokenizer for ARIA."""

from __future__ import annotations


class ByteTokenizer:
    """Encode UTF-8 text into stable byte token IDs.

    IDs 0-255 map directly to byte values. Special IDs are kept outside that
    range so the vocabulary is deterministic and independent of training data.
    """

    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    UNK_ID = 259
    VOCAB_SIZE = 260

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.BOS_ID)
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            if 0 <= token_id <= 255:
                byte_values.append(token_id)
            elif token_id in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            else:
                raise ValueError(f"invalid token id: {token_id}")
        return bytes(byte_values).decode("utf-8", errors="strict")

    def token_to_bytes(self, token_id: int) -> bytes:
        if 0 <= token_id <= 255:
            return bytes([token_id])
        raise ValueError("special tokens do not map to byte values")

    def vocab_size(self) -> int:
        return self.VOCAB_SIZE
