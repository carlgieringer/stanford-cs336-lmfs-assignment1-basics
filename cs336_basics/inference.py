"""
Interactive CLI for text generation using a trained transformer model.

Usage:
    python cs336_basics/inference.py \
        --model-checkpoint path/to/checkpoint.pt \
        --tokenizer path/to/tokenizer.pkl \
        --max-tokens 100 \
        --temperature 0.8 \
        --top-p 0.9
"""

import argparse
import sys
import traceback
from typing import Optional

import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.checkpointing import load_train_state
from cs336_basics.softmax import softmax
from cs336_basics.transformer import TransformerLm


class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_colored(text: str, color: str, end: str = "\n"):
    """Print text with color."""
    print(f"{color}{text}{Colors.RESET}", end=end)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample next token using temperature and top-p sampling."""
    # Apply temperature scaling
    if temperature > 0:
        logits = logits / temperature
    else:
        # If temperature is 0, use greedy sampling
        return int(logits.argmax().item())

    # Convert to probabilities
    probs = softmax(logits, dim=-1)

    # Apply top-p filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff index
        mask = cumulative_probs <= top_p
        # Always include at least the top token
        if not mask.any():
            mask[0] = True
        else:
            # Include the first token that exceeds top_p
            cutoff_idx = int(mask.sum().item())
            if cutoff_idx < len(mask):
                mask[cutoff_idx] = True

        # Create filtered probability distribution
        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_indices[mask]] = sorted_probs[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()
        probs = filtered_probs

    # Sample from the distribution
    next_token = torch.multinomial(probs, 1)
    return int(next_token.item())


def generate_text(
    model: torch.nn.Module,
    tokenizer: BpeTokenizer,
    prompt: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    endoftext_token_id: int,
    context_length: int,
) -> tuple[str, int, str]:
    """
    Generate text from a prompt.

    Returns:
        tuple of (generated_text, num_tokens_generated, stop_reason)
    """
    model.eval()

    # Tokenize the prompt
    input_tokens = tokenizer.encode(prompt)

    # Check if prompt exceeds context length
    if len(input_tokens) >= context_length:
        return "", 0, "prompt too long"

    # Convert to tensor and move to model's device
    device = next(model.parameters()).device
    tokens = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(
        0
    )  # Add batch dimension

    generated_text = ""
    num_generated = 0

    with torch.no_grad():
        while True:
            # Check stopping conditions
            if len(tokens[0]) >= context_length:
                stop_reason = "context length"
                break

            if max_tokens is not None and num_generated >= max_tokens:
                stop_reason = "max tokens"
                break

            # Forward pass
            logits = model(tokens)

            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Sample next token
            next_token_id = sample_next_token(next_token_logits, temperature, top_p)

            # Check for end of text token
            if next_token_id == endoftext_token_id:
                stop_reason = "end token"
                break

            # Decode and print the token
            token_text = tokenizer.decode([next_token_id])
            print(token_text, end="", flush=True)
            generated_text += token_text

            # Add token to sequence
            next_token_tensor = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=device
            )
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

            num_generated += 1

    return generated_text, num_generated, stop_reason


def main():
    parser = argparse.ArgumentParser(description="Interactive text generation CLI")
    parser.add_argument(
        "--model-checkpoint", required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Path to BPE tokenizer pickle file"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.temperature < 0:
        print_colored("Error: Temperature must be non-negative", Colors.RED)
        sys.exit(1)

    if not (0 < args.top_p <= 1.0):
        print_colored("Error: Top-p must be between 0 and 1", Colors.RED)
        sys.exit(1)

    if args.max_tokens is not None and args.max_tokens <= 0:
        print_colored("Error: Max tokens must be positive", Colors.RED)
        sys.exit(1)

    try:
        # Load tokenizer
        print_colored("Loading tokenizer...", Colors.YELLOW)
        tokenizer = BpeTokenizer.from_file(
            args.tokenizer, special_tokens=["<|endoftext|>"]
        )
        vocab_size = len(tokenizer.vocab)

        # Get the end of text token ID
        endoftext_token_id = tokenizer.token_by_bytes["<|endoftext|>".encode("utf-8")]

        # Load model
        print_colored("Loading model...", Colors.YELLOW)
        train_state = load_train_state(vocab_size, args.model_checkpoint)
        model = train_state["model"]

        # Ensure we have a TransformerLm model
        if not isinstance(model, TransformerLm):
            raise ValueError(f"Expected TransformerLm model, got {type(model)}")

        # Get model parameters
        context_length = 512  # Default fallback
        if "training_params" in train_state and isinstance(
            train_state["training_params"], dict
        ):
            training_params = train_state["training_params"]
            if (
                "model_params" in training_params
                and training_params["model_params"] is not None
            ):
                model_params = training_params["model_params"]
                if hasattr(model_params, "context_length"):
                    context_length = int(model_params.context_length)

        # If we still don't have context_length from training params, keep the default
        # The context_length should be saved in the training_params, so this fallback
        # is mainly for older checkpoints that might not have this information

        # Print model info
        print_colored(f"Model loaded successfully!", Colors.GREEN)
        print_colored(f"Vocabulary size: {vocab_size}", Colors.CYAN)
        print_colored(f"Context length: {context_length}", Colors.CYAN)
        print_colored(f"Temperature: {args.temperature}", Colors.CYAN)
        print_colored(f"Top-p: {args.top_p}", Colors.CYAN)
        if args.max_tokens:
            print_colored(f"Max tokens: {args.max_tokens}", Colors.CYAN)
        print_colored("", Colors.RESET)  # Empty line

        # Interactive loop
        print_colored(
            "Enter text to generate completions. Press Ctrl+C to exit.", Colors.YELLOW
        )
        print_colored("", Colors.RESET)  # Empty line

        while True:
            try:
                # Get user input
                print_colored("You: ", Colors.GREEN, end="")
                user_input = input()

                if not user_input.strip():
                    continue

                # Generate response
                print_colored("Model: ", Colors.BLUE, end="")
                generated_text, num_tokens, stop_reason = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    endoftext_token_id=endoftext_token_id,
                    context_length=context_length,
                )

                # Print generation stats
                print()  # New line after generation
                print_colored(
                    f"Generated {num_tokens} tokens (stopped: {stop_reason})",
                    Colors.CYAN,
                )
                print_colored("", Colors.RESET)  # Empty line

            except KeyboardInterrupt:
                print_colored("\nGoodbye!", Colors.YELLOW)
                break
            except Exception as e:
                print_colored(f"\nError during generation: {e}", Colors.RED)
                print_colored("Stack trace:", Colors.RED)
                traceback.print_exc()
                continue

    except FileNotFoundError as e:
        print_colored(f"Error: File not found - {e}", Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        print_colored("Stack trace:", Colors.RED)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
