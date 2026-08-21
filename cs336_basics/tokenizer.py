import json
from pathlib import Path
from typing import Iterable, Iterator

import regex as re


# def _gpt2_bytes_to_unicode() -> dict[int, str]:
#     """Return the GPT-2 byte-to-unicode mapping used by the assignment fixtures."""
#     bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
#     cs = bs[:]
#     n = 0
#     for b in range(256):
#         if b not in bs:
#             bs.append(b)
#             cs.append(256 + n)
#             n += 1
#     return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:

    PRETOKENIZATION_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("special tokens must be unique")
        if any(token == "" for token in self.special_tokens):
            raise ValueError("special tokens must be non-empty")

        self.token_to_id: dict[bytes, int] = {
            token: token_id for token_id, token in self.vocab.items()
        }

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

        self.special_token_bytes = {
            token.encode("utf-8") for token in self.special_tokens
        }

    # @classmethod
    # def from_files(
    #     cls,
    #     vocab_filepath: str | Path,
    #     merges_filepath: str | Path,
    #     special_tokens: list[str] | None = None,
    # ):
    #     """Load a tokenizer from a GPT-2-style vocab JSON and BPE merges file."""
    #     with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
    #         vocab_data = json.load(vocab_file)

    #     byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}
    #     vocab: dict[int, bytes] = {}

    #     if isinstance(next(iter(vocab_data)), str):
    #         for token_str, token_id in vocab_data.items():
    #             if isinstance(token_str, bytes):
    #                 token_bytes = token_str
    #             else:
    #                 try:
    #                     token_bytes = bytes(byte_decoder[ch] for ch in token_str)
    #                 except KeyError:
    #                     token_bytes = token_str.encode("utf-8")
    #             vocab[int(token_id)] = token_bytes
    #     else:
    #         for token_id, token_bytes in vocab_data.items():
    #             vocab[int(token_id)] = token_bytes if isinstance(token_bytes, bytes) else token_bytes.encode("utf-8")

    #     merges: list[tuple[bytes, bytes]] = []
    #     with open(merges_filepath, "r", encoding="utf-8") as merges_file:
    #         for line in merges_file:
    #             line = line.strip()
    #             if not line or line.startswith("#"):
    #                 continue
    #             left, right = line.split()
    #             left_bytes = left.encode("utf-8")
    #             right_bytes = right.encode("utf-8")
    #             try:
    #                 left_bytes = bytes(byte_decoder[ch] for ch in left)
    #                 right_bytes = bytes(byte_decoder[ch] for ch in right)
    #             except KeyError:
    #                 pass
    #             merges.append((left_bytes, right_bytes))

    #     return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []

        pretokens = self._pretokenize(text)
        for pretoken in pretokens:
            if pretoken in self.special_token_bytes:
                ids.append(self.token_to_id[pretoken])
                continue

            token = [bytes([byte]) for byte in pretoken]
            token = self._merge(token)

            for token_bytes in token:
                ids.append(self.token_to_id[token_bytes])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")

    def _pretokenize(self, text: str) -> list[bytes]:
        pretokens  = []
        pretoken_pattern = re.compile(self.PRETOKENIZATION_PATTERN)

        def text_to_pretokens(text: str):
            for match in pretoken_pattern.finditer(text):
                pretokens.append(match.group().encode("utf-8"))

        if not self.special_tokens:
            text_to_pretokens(text)
            return pretokens

        special_pattern = re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))

        start = 0
        for match in special_pattern.finditer(text):
            text_to_pretokens(text[start:match.start()])
            pretokens.append(match.group().encode("utf-8"))
            start = match.end()

        text_to_pretokens(text[start:])

        return pretokens

    def _merge(self, token: list[bytes]) -> list[bytes]:
        for pair in self.merges:
            merged_token = []

            i = 0
            while i < len(token):
                if i + 1 < len(token) and token[i] == pair[0] and token[i+1] == pair[1]:
                    merged_token.append(token[i] + token[i+1])
                    i += 2
                else:
                    merged_token.append(token[i])
                    i += 1

            token = merged_token

        return token
