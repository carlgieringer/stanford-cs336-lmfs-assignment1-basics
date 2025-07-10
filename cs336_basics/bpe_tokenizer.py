"""
Test:

```
uv run pytest tests/test_tokenizer.py
```
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import zip_longest
import logging
import pickle
from typing import Generator, Iterable, Optional

import regex
from tqdm import tqdm

from cs336_basics import bpe

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class TqdmParams:
    position: int
    tqdm_position_desc: str


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        if special_tokens:
            BpeTokenizer._add_special_tokens_to_vocab(vocab, special_tokens)
        self.special_tokens = special_tokens
        self.vocab = vocab
        self.token_by_bytes = {
            token_bytes: token for token, token_bytes in vocab.items()
        }
        self.merges = merges

    @staticmethod
    def _add_special_tokens_to_vocab(
        vocab: dict[int, bytes], special_tokens: list[str]
    ):
        new_vocab = dict()
        for special_token in special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in set(vocab.values()):
                new_vocab[len(vocab)] = special_token_bytes
        vocab.update(new_vocab)

    @staticmethod
    def from_file(data_filepath, special_tokens=None):
        with open(data_filepath, "rb") as f:
            data = pickle.load(f)
        return BpeTokenizer(data["vocab"], data["merges"], special_tokens)

    def encode(self, text: str, tqdm_params: Optional[TqdmParams] = None) -> list[int]:
        return list(self._generate_tokens(text, tqdm_params))

    def _generate_tokens(
        self, text: str, tqdm_params: Optional[TqdmParams] = None
    ) -> Generator[int]:
        if self.special_tokens:
            special_tokens_pattern = bpe.make_special_tokens_pattern(
                self.special_tokens or []
            )
            special_token_matches = list(regex.finditer(special_tokens_pattern, text))
            if special_token_matches:
                split_offsets = (
                    [(0, special_token_matches[0].start())]
                    + [
                        (m1.end(), m2.start())
                        for m1, m2 in zip(
                            special_token_matches, special_token_matches[1:]
                        )
                    ]
                    + [(special_token_matches[-1].end(), len(text))]
                )
                splits = [text[start:end] for start, end in split_offsets]
            else:
                special_token_matches = []
                splits = [text]
        else:
            special_token_matches = []
            splits = [text]
        splits_pretoken_bytes_lists: list[list[list[bytes]]] = []

        splits_iter = (
            tqdm(splits, desc="Splits", unit="split")
            if tqdm_params is None
            else tqdm(
                splits,
                desc=f"Splits ({tqdm_params.tqdm_position_desc})",
                unit="split",
                position=tqdm_params.position,
                leave=False,
            )
        )
        for split in splits_iter:
            split_pretoken_bytes_lists: list[list[bytes]] = []
            splits_pretoken_bytes_lists.append(split_pretoken_bytes_lists)
            pretokens = regex.findall(bpe.PRETOKEN_PATTERN, split)
            for pretoken in pretokens:
                pretoken_bytes_list = list(
                    bpe.BYTE_CACHE[b] for b in pretoken.encode("utf-8")
                )
                for merge in self.merges:
                    i = 0
                    while i < len(pretoken_bytes_list) - 1:
                        if (
                            pretoken_bytes_list[i] == merge[0]
                            and pretoken_bytes_list[i + 1] == merge[1]
                        ):
                            pretoken_bytes_list[i : i + 2] = [b"".join(merge)]
                        i += 1
                split_pretoken_bytes_lists.append(pretoken_bytes_list)
        for split_token_bytes_list, special_token_match in zip_longest(
            splits_pretoken_bytes_lists, special_token_matches
        ):
            for token_bytes_list in split_token_bytes_list:
                for token_bytes in token_bytes_list:
                    yield self.token_by_bytes[token_bytes]
            if special_token_match:
                special_token_bytes = special_token_match.group(0).encode("utf-8")
                yield self.token_by_bytes[special_token_bytes]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._generate_tokens(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")
