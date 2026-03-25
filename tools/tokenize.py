#!/usr/bin/env python3
"""Tokenize text using HuggingFace tokenizers library (100% correct).

Called by MnemoCUDA engine as subprocess:
  python3 tokenize.py <model_dir> "text to encode"
  → outputs binary: [4B n_tokens] [4B token_id] * n_tokens

Also supports pipe mode:
  echo "text" | python3 tokenize.py <model_dir> --pipe
"""

import struct
import sys
import os

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <text>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    text = sys.argv[2] if sys.argv[2] != "--pipe" else sys.stdin.read()

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    ids = tok.encode(text).ids

    # Write binary to stdout
    sys.stdout.buffer.write(struct.pack("<I", len(ids)))
    for tid in ids:
        sys.stdout.buffer.write(struct.pack("<i", tid))

if __name__ == "__main__":
    main()
