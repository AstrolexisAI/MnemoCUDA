#!/usr/bin/env python3
"""Pre-process HuggingFace tokenizer.json into MnemoCUDA binary format.

Output: tokenizer.bin in the model directory.

Format:
  [4B] magic = 0x4D544F4B ("MTOK")
  [4B] vocab_size
  [4B] n_merges
  [4B] n_special
  [4B] eos_id
  [4B] im_start_id
  [4B] im_end_id

  -- Vocab section: vocab_size entries --
  For each token:
    [2B] length (0 if slot is empty)
    [NB] UTF-8 string (no null terminator)

  -- Merges section: n_merges entries --
  For each merge:
    [2B] length
    [NB] UTF-8 string "tokenA tokenB"

  -- Special tokens section: n_special entries --
  For each special:
    [4B] token_id
    [2B] length
    [NB] UTF-8 string
"""

import json
import struct
import sys
import os

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    src = os.path.join(model_dir, "tokenizer.json")
    dst = os.path.join(model_dir, "tokenizer.bin")

    with open(src, "r") as f:
        tok = json.load(f)

    vocab = tok["model"]["vocab"]  # str → id
    merges = tok["model"].get("merges", [])
    added = tok.get("added_tokens", [])

    # Build id → str mapping
    max_id = max(vocab.values()) if vocab else 0
    for at in added:
        if at["id"] > max_id:
            max_id = at["id"]

    vocab_size = max_id + 1
    id_to_str = [""] * vocab_size
    for token_str, token_id in vocab.items():
        id_to_str[token_id] = token_str

    # Add added_tokens to vocab
    for at in added:
        id_to_str[at["id"]] = at["content"]

    # Special tokens
    specials = [(at["id"], at["content"]) for at in added if at.get("special", False)]

    eos_id = 151643
    im_start_id = 151644
    im_end_id = 151645
    for sid, scontent in specials:
        if scontent == "<|endoftext|>": eos_id = sid
        elif scontent == "<|im_start|>": im_start_id = sid
        elif scontent == "<|im_end|>": im_end_id = sid

    with open(dst, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x4D544F4B))  # magic
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(specials)))
        f.write(struct.pack("<I", eos_id))
        f.write(struct.pack("<I", im_start_id))
        f.write(struct.pack("<I", im_end_id))

        # Vocab
        for i in range(vocab_size):
            s = id_to_str[i].encode("utf-8")
            f.write(struct.pack("<H", len(s)))
            f.write(s)

        # Merges
        for m in merges:
            # merges can be ["tokenA", "tokenB"] or "tokenA tokenB"
            if isinstance(m, list):
                merge_str = " ".join(m)
            else:
                merge_str = m
            s = merge_str.encode("utf-8")
            f.write(struct.pack("<H", len(s)))
            f.write(s)

        # Special tokens
        for sid, scontent in specials:
            s = scontent.encode("utf-8")
            f.write(struct.pack("<I", sid))
            f.write(struct.pack("<H", len(s)))
            f.write(s)

    print(f"Written {dst}: {vocab_size} vocab, {len(merges)} merges, {len(specials)} special")
    print(f"  EOS={eos_id}, IM_START={im_start_id}, IM_END={im_end_id}")
    print(f"  Size: {os.path.getsize(dst) / 1024:.0f} KB")

if __name__ == "__main__":
    main()
