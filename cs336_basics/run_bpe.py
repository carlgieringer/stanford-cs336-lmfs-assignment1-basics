"""
Run like:

```sh
uv run python cs336_basics/run_bpe.py\
 --input-path=data/TinyStoriesV2-GPT4-valid.txt\
 --vocab-size=10000\
 --special-tokens='<|endoftext|>'\
 --output-path=data/bpe-TinyStoriesV2-GPT4-valid.pk
```
"""

import argparse
import logging
import pickle

from cs336_basics import bpe

parser = argparse.ArgumentParser("run_train_bpe")
parser.add_argument("--input-path", required=True)
parser.add_argument("--vocab-size", type=int, required=True)
parser.add_argument("--special-tokens")
parser.add_argument(
    "--num-processes",
    type=int,
    default=1,
)
parser.add_argument("--output-path", required=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_train_bpe():
    args = parser.parse_args()
    logger.info(f"Running {parser.prog} with {args}")
    special_tokens = args.special_tokens.split(",")
    vocab, merges = bpe.train_bpe(
        args.input_path,
        args.vocab_size,
        special_tokens,
        num_processes=args.num_processes,
    )
    logging.info(
        f"Writing {len(vocab)} vocab and {len(merges)} merges to {args.output_path}"
    )
    with open(args.output_path, "wb") as f:
        pickle.dump(dict(vocab=vocab, merges=merges), f)
    logging.info("Done training.")


if __name__ == "__main__":
    run_train_bpe()
