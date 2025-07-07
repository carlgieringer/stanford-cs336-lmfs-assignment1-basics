"""
Run:

```sh
uv run python cs336_basics/run_bpe_tokenizer.py\
 --tokenizer-pickle-file=data/bpe-TinyStoriesV2-GPT4-train.pk\
 --sample-file=data/TinyStoriesV2-GPT4-valid.txt
```
"""

import argparse
import enum
import logging
import pickle
import random
import time

from tqdm import tqdm

from cs336_basics.bpe_tokenizer import BpeTokenizer

END_OF_TEXT = "<|endoftext|>"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Calculation(enum.Enum):
    COMPRESSION_RATIO = enum.auto()
    THROUGHPUT = enum.auto()


def calculate(
    tokenizer_pickle_file: str,
    sample_file: str,
    sample_count: int,
    random_seed: int,
    calculation: Calculation,
):
    random.seed(random_seed)

    logger.info(f"Calculating {calculation}")

    with open(tokenizer_pickle_file, "rb") as f:
        tokenizer_pickle_data = pickle.load(f)
    tokenizer = BpeTokenizer(
        tokenizer_pickle_data["vocab"],
        tokenizer_pickle_data["merges"],
        special_tokens=[END_OF_TEXT],
    )

    with open(sample_file, "r") as f:
        docs = f.read().split(END_OF_TEXT)
        random.shuffle(docs)
        sample_docs = docs[:sample_count]

    if calculation == Calculation.COMPRESSION_RATIO:
        for doc in tqdm(sample_docs, desc="Calculating compression ratios"):
            byte_count = len(doc.encode("utf-8"))
            token_count = len(tokenizer.encode(doc))
            compression_ratio = token_count / byte_count
            logger.info(f"Compression ratio: {compression_ratio}")
    elif calculation == Calculation.THROUGHPUT:
        total_byte_count = 0
        logger.info("Counting bytes")
        for doc in sample_docs:
            total_byte_count += len(doc.encode("utf-8"))

        start_time = time.time()
        for doc in tqdm(sample_docs, desc="Encoding"):
            tokenizer.encode(doc)
        end_time = time.time()
        elapsed_secs = end_time - start_time
        throughput = total_byte_count / elapsed_secs
        logger.info(f"Throughput: {throughput} B/s")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer-pickle-file", help="The file from which to load the tokenizer"
)
parser.add_argument("--sample-file", help="The file from which to sample docs")
parser.add_argument(
    "--sample-count", type=int, default=10, help="The number of documents to sample."
)
parser.add_argument(
    "--random-seed", type=int, default=42, help="The seed for Python's random"
)
parser.add_argument(
    "--calculation",
    type=lambda v: Calculation[v.upper()],
    choices=list(Calculation),
    help="The calculation to perform",
)


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(f"Running {parser.prog} with args: {args}")
    calculate(
        args.tokenizer_pickle_file,
        args.sample_file,
        args.sample_count,
        args.random_seed,
        args.calculation,
    )
