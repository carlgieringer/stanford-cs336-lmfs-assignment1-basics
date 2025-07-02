import os
import collections
from typing import BinaryIO, List, Tuple

import regex

BYTE_TOKEN_COUNT = 256


def bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pretoken_byte_counts = collections.Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            kwargs.get("num_processes", 1),
            [t.encode("utf-8") for t in special_tokens],
        )

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            chunk_pretoken_byte_counts = count_pretoken_bytes(chunk, special_tokens)
            pretoken_byte_counts.update(chunk_pretoken_byte_counts)

    vocab_list = [bytes([i]) for i in range(BYTE_TOKEN_COUNT)]
    for special_token in special_tokens:
        vocab_list.append(special_token.encode("utf-8"))

    merge_token_allowance = vocab_size - len(vocab_list)
    merges = determine_merges(pretoken_byte_counts, merge_token_allowance)
    for token in [b"".join((a, b)) for a, b in merges]:
        vocab_list.append(token)

    vocab = {i: token for i, token in enumerate(vocab_list)}
    return vocab, merges


def determine_merges(
    pretoken_byte_counts: collections.Counter[Tuple[bytes]], merge_token_allowance: int
) -> List[Tuple[bytes, bytes]]:
    merges = []
    did_merge = True
    while len(merges) < merge_token_allowance and did_merge:
        # Find the most common pair
        # {(b'a', b't'): 2}
        prototoken_pair_counts = collections.Counter()
        for pretoken_bytes, count in pretoken_byte_counts.items():
            pairs = [pretoken_bytes[i : i + 2] for i in range(len(pretoken_bytes) - 1)]
            for pair in pairs:
                prototoken_pair_counts[pair] += count
        # Find the most common pair. In case of ties, take the lexicographically greatest
        sorted_pair_counts = prototoken_pair_counts.most_common()
        most_common_prototoken_pair, most_common_prototoken_pair_count = (
            sorted_pair_counts[0]
        )
        for i in range(1, len(sorted_pair_counts)):
            next_pretoken_count = sorted_pair_counts[i][1]
            if next_pretoken_count < most_common_prototoken_pair_count:
                break
            next_pretoken_pair = sorted_pair_counts[i][0]
            if next_pretoken_pair < most_common_prototoken_pair:
                # Check other pairs for being lexicographically greater
                continue
            most_common_prototoken_pair = next_pretoken_pair
        del prototoken_pair_counts[most_common_prototoken_pair]
        merges.append(most_common_prototoken_pair)
        most_common_bytes = b"".join(most_common_prototoken_pair)

        # merge within pretoken_byte_counts keys.
        # We need to update prototoken_pair_counts too.
        # When we find most_common_prototoken_pair in the pretoken_byte_counts, we need
        # to replace the pairs made from the prototoken before and after it.
        did_merge = False
        new_pretoken_byte_counts = collections.Counter()
        merged_pretoken_bytes = []
        for pretoken_bytes, count in pretoken_byte_counts.items():
            new_pretoken_bytes = []
            i = 0
            while i < len(pretoken_bytes) - 1:
                if pretoken_bytes[i : i + 2] == most_common_prototoken_pair:
                    new_pretoken_bytes.append(most_common_bytes)
                    i += 2
                else:
                    new_pretoken_bytes.append(pretoken_bytes[i])
                    i += 1
            # Add the last byte if it wasn't picked up as a pair
            if i < len(pretoken_bytes):
                new_pretoken_bytes.append(pretoken_bytes[i])
            # Store the count of the merged pretoken
            if len(new_pretoken_bytes) < len(pretoken_bytes):
                merged_pretoken_bytes.append(pretoken_bytes)
                new_pretoken_byte_counts[tuple(new_pretoken_bytes)] = (
                    pretoken_byte_counts[pretoken_bytes]
                )
                did_merge = True
        pretoken_byte_counts.update(new_pretoken_byte_counts)
        for pretoken_bytes in merged_pretoken_bytes:
            del pretoken_byte_counts[pretoken_bytes]
    return merges


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    special_tokens_pattern = b"|".join(regex.escape(t) for t in special_tokens)
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            match = regex.search(special_tokens_pattern, mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


BASE_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def count_pretoken_bytes(string: str, special_tokens: List[str]):
    """
    Returns:
      pretoken_counts: a Counter keyed by the pretoken bytes.
    """

    # ["fat", " cat"]
    special_tokens_pattern = "|".join(regex.escape(t) for t in special_tokens)
    pretoken_pattern = special_tokens_pattern + "|" + BASE_PRETOKEN_PATTERN
    pretoken_matches = regex.finditer(pretoken_pattern, string)
    # {(b'f', b'a', b't'): 1, (b' ', b'c', b'a', b't'): 1}
    pretoken_byte_counts = collections.Counter(
        tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
        for m in pretoken_matches
        if not regex.match(special_tokens_pattern, m.group(0))
    )
    return pretoken_byte_counts
