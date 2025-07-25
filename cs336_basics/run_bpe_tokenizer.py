"""
Run:

```sh
uv run python cs336_basics/run_bpe_tokenizer.py\
 --tokenizer-pickle-file=data/bpe-TinyStoriesV2-GPT4-train.pk\
 --corpus-path=data/TinyStoriesV2-GPT4-valid.txt\
 --process-count=12\
 --max-memory=5MiB\
 --output-path=data/tokens-TinyStoriesV2-GPT4-valid
```
"""

import argparse
from dataclasses import dataclass
import enum
import logging
import math
from multiprocessing import Pool
import os
import pickle
import random
from typing import Optional

import numpy as np
from tqdm import tqdm
from humanfriendly import parse_size, format_size

from cs336_basics import bpe
from cs336_basics.bpe_tokenizer import BpeTokenizer, TqdmParams

END_OF_TEXT = "<|endoftext|>"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessChunkArgs:
    input_path: str
    """The file to read offset content from"""

    start: int
    """The offset in the file to start reading from"""

    end: int
    """The offset in the file to end reading from"""

    special_tokens: list[str]

    tokenizer_pickle_file_path: str
    """The file serializing the BPE tokenizer params (vocab and merges)"""

    base_output_path: str

    chunk_index: int
    """The index of the chunk in all chunks"""

    chunk_count: int
    """The total number of chunks. Useful for displaying the chunk_index in fixed width."""

    tqdm_position: int
    """The position to use with TQDM to display multiple progress bars simultaneously."""


def tokenize_corpus(
    tokenizer_pickle_file_path: str,
    corpus_path: str,
    chunk_count: int,
    process_count: int,
    base_output_path: str,
    random_seed: int,
    keep_chunks: bool,
    start_offset: Optional[int],
):
    random.seed(random_seed)

    special_tokens = [END_OF_TEXT]

    logger.info(f"Reading corpus {corpus_path}")

    with open(corpus_path, "rb") as f:
        boundaries = bpe.find_chunk_boundaries(
            f,
            chunk_count,
            [t.encode("utf-8") for t in special_tokens],
        )

    chunk_args = [
        ProcessChunkArgs(
            corpus_path,
            start,
            end,
            special_tokens,
            tokenizer_pickle_file_path,
            base_output_path,
            chunk_index,
            chunk_count,
            tqdm_position=(chunk_index + 1) % process_count,
        )
        for chunk_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    # Filter after determining chunk indices to preserve them.
    if start_offset:
        all_chunk_count = len(chunk_args)
        chunk_args = [ca for ca in chunk_args if ca.end > start_offset]
        chunk_args[0].start = start_offset
        logging.info(
            f"Skipping {all_chunk_count - len(chunk_args)} out of {all_chunk_count} chunks to start at {start_offset}"
        )

    pool = Pool(process_count)
    chunk_tokenization_filenames = []
    try:
        for filename in tqdm(
            pool.imap_unordered(process_chunk, chunk_args),
            desc="Chunks",
            unit="chunk",
            position=0,
            leave=True,
        ):
            chunk_tokenization_filenames.append(filename)
    finally:
        pool.close()
        pool.join()
        pool.terminate()

    # Combine all chunk files into a single file
    chunk_tokenization_paths = [f"{name}.npy" for name in chunk_tokenization_filenames]
    combine_chunk_files(chunk_tokenization_paths, base_output_path, keep_chunks)

    logger.info(f"Done tokenizing.")


def process_chunk(args: ProcessChunkArgs):
    # Use zero-padded chunk index for lexicographical sorting
    file_chunk_width = math.ceil(math.log10(args.chunk_count))
    output_filename = f"{args.base_output_path}.{args.chunk_index:0{file_chunk_width}d}.{args.start}-{args.end-1}"
    if os.path.exists(f"{output_filename}.npy"):
        logger.info(f"{output_filename} already exists, exiting early.")
        return output_filename

    with open(args.input_path, "rb") as f:
        f.seek(args.start)
        chunk = f.read(args.end - args.start).decode("utf-8", errors="ignore")
    logger.info(f"Tokenizing {args.input_path} in range {(args.start, args.end)}")
    with open(args.tokenizer_pickle_file_path, "rb") as f:
        tokenizer_pickle_data = pickle.load(f)
    tokenizer = BpeTokenizer(
        tokenizer_pickle_data["vocab"],
        tokenizer_pickle_data["merges"],
        special_tokens=args.special_tokens,
    )

    tqdm_position_width = math.ceil(math.log10(args.chunk_count))
    tqdm_position_desc = (
        f"{args.chunk_index + 1:0{tqdm_position_width}d}/{args.chunk_count}"
    )
    tokenization = tokenizer.encode(
        chunk,
        tqdm_params=TqdmParams(
            # +1 so that the Chunks progress at 0 is not overwritten
            position=args.tqdm_position,
            tqdm_position_desc=tqdm_position_desc,
        ),
    )

    tokenization_array = np.array(tokenization, dtype=np.uint16)
    np.save(output_filename, tokenization_array)
    logger.info(
        f"Done tokenizing chunk {args.chunk_index} for range {(args.start, args.end)}"
    )
    return output_filename


def combine_chunk_files(
    chunk_tokenization_paths: list[str],
    output_path: str,
    keep_chunks: bool,
):
    """
    Combine multiple NumPy chunk files into a single file by loading arrays into memory.

    Args:
        chunk_tokenization_paths: Paths of tokenization files to combine
        output_path: Path for the final combined file (e.g., "data/tokens-file-combined.npy")
        keep_chunks: Whether to keep individual chunk files after combining

    Returns:
        Path to the final combined file
    """
    if not output_path.endswith(".npy"):
        output_path = f"{output_path}.npy"
    # Sort files lexicographically (works because we use fixed-width zero-padded integers)
    chunk_tokenization_paths.sort()

    logger.info(f"Combining chunks: {chunk_tokenization_paths}")

    # Load all chunk arrays into memory
    chunk_arrays = []
    dtype = None
    total_length = 0

    for file_path in chunk_tokenization_paths:
        chunk_array = np.load(file_path)
        if dtype is None:
            dtype = chunk_array.dtype
        elif chunk_array.dtype != dtype:
            raise ValueError(f"Inconsistent dtypes: {dtype} vs {chunk_array.dtype}")

        chunk_arrays.append(chunk_array)
        total_length += chunk_array.shape[0]

        logger.info(
            f"Loaded chunk from {file_path} ({chunk_array.shape[0]} tokens)"
        )

    logger.info(f"Total combined array length: {total_length}, dtype: {dtype}")

    # Combine all arrays in memory using concatenate
    combined_array = np.concatenate(chunk_arrays, axis=0)

    # Write the combined array to disk
    np.save(output_path, combined_array)

    logger.info(
        f"Successfully combined {len(chunk_tokenization_paths)} chunks into {output_path}"
    )

    # Optionally remove chunk files
    if not keep_chunks:
        for file_path in chunk_tokenization_paths:
            os.remove(file_path)
            logger.info(f"Removed chunk file: {file_path}")


class Action(enum.Enum):
    TOKENIZE = enum.auto()
    COMBINE = enum.auto()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--action",
    choices=list(Action),
    type=lambda v: Action[v.upper()],
    help="The action to take.",
)
parser.add_argument(
    "--tokenizer-pickle-file-path", help="The file from which to load the tokenizer"
)
parser.add_argument("--corpus-path", help="The file containing docs to tokenize")
parser.add_argument(
    "--process-count", type=int, help="The number of chunks to process in parallel."
)
parser.add_argument(
    "--max-memory",
    type=parse_size,
    help="The target maximum memory usage (e.g., '5GiB', '512MiB').",
)
parser.add_argument(
    "--output-path",
    help="The path to which to serialize the tokenization (NumPy uint16 array). The extension .npy is automatically appended.",
)
parser.add_argument(
    "--random-seed", type=int, default=42, help="The seed for Python's random"
)
parser.add_argument(
    "--keep-chunks",
    action="store_true",
    help="Keep individual chunk files after combining (default: False)",
)
parser.add_argument(
    "--start-offset", type=int, help="If present, starts tokenization from this offset."
)
parser.add_argument(
    "--chunk-files",
    type=lambda v: v.split(","),
    help="A comma-separated list of files to combine.",
)

# The program's approximate memory usage factor over the corpus file size itself
# Determined by actually running the thing on a MacBook Pro M4 and grabbing the actual memory usage
# from Activity Monitor during a chunk's execution.
MEM_OVERHEAD_FACTOR = 36


def determine_chunk_count(
    corpus_path: str, max_memory_bytes: int, process_count: int
) -> int:
    file_size = os.path.getsize(corpus_path)
    estimated_required_memory = MEM_OVERHEAD_FACTOR * file_size
    required_memory_factor = estimated_required_memory / max_memory_bytes
    # max: if the allowed max memory happens to be much more than is necessary, still spread out the
    # tokenization across the total number of processes
    chunk_count = max(
        process_count,
        math.ceil(process_count * required_memory_factor),
    )
    logger.info(
        f"Breaking up {format_size(file_size, binary=True)} byte corpus into {chunk_count} chunks"
        f" to keep memory usage below {format_size(max_memory_bytes, binary=True)} bytes across {process_count} processes"
    )
    return chunk_count


def main():
    args = parser.parse_args()
    logger.info(f"Running {parser.prog} with args: {args}")

    if args.action == Action.TOKENIZE:
        chunk_count = determine_chunk_count(
            args.corpus_path, args.max_memory, args.process_count
        )

        tokenize_corpus(
            args.tokenizer_pickle_file_path,
            args.corpus_path,
            chunk_count,
            args.process_count,
            args.output_path,
            args.random_seed,
            args.keep_chunks,
            args.start_offset,
        )
    elif args.action == Action.COMBINE:
        if not args.chunk_files:
            raise ValueError(f"COMBINE action requires chunk_files")
        combine_chunk_files(args.chunk_files, args.output_path, args.keep_chunks)
    else:
        raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
