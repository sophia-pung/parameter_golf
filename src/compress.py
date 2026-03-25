"""
compress.py — serialize a trained DualJEPAModel to a ≤16MB submission artifact.

Strategy:
  1. Load the final checkpoint.
  2. Strip the JEPA head (train-only; contributes zero bytes to the submission).
  3. Quantize weights to float16 (halves size vs float32).
  4. Save the state-dict with torch.save and then gzip it for extra compression.
  5. Print the final artifact size so we know whether it fits the 16MB limit.
"""

import os                           # for makedirs and file size checks
import gzip                         # stdlib gzip: compress the serialized bytes
import shutil                       # for copying / moving files
import argparse                     # for parsing --checkpoint and --output flags

import torch                        # for loading and re-saving the model state dict
import yaml                         # for reading the config file

from model import DualJEPAModel     # needed to instantiate the model before loading weights


# ── Strip the JEPA head ────────────────────────────────────────────────────────

def strip_jepa_head(state_dict: dict) -> dict:
    """
    Remove all keys belonging to the JEPA projection head from the state dict.
    These keys begin with 'jepa_head.' and are only used during training.
    Stripping them saves ~130K parameters (≈260 KB in fp16) from the artifact.
    """
    return {                                        # rebuild dict without jepa_head keys
        k: v for k, v in state_dict.items()
        if not k.startswith("jepa_head.")           # drop any key that belongs to the JEPA head
    }


# ── Float16 quantization ───────────────────────────────────────────────────────

def quantize_to_fp16(state_dict: dict) -> dict:
    """
    Cast every floating-point parameter tensor to float16.
    Integer tensors (e.g. running_mean buffers) are left unchanged.
    float16 halves storage vs float32 with negligible quality loss for inference.
    """
    out = {}
    for k, v in state_dict.items():
        if v.is_floating_point():                  # only cast float tensors (not int masks, etc.)
            out[k] = v.half()                      # .half() converts float32 → float16
        else:
            out[k] = v                             # keep integer buffers as-is
    return out


# ── Serialize + gzip ───────────────────────────────────────────────────────────

def save_compressed(state_dict: dict, out_path: str):
    """
    Save the state dict to a gzip-compressed .pt.gz file.

    Steps:
      1. torch.save → raw bytes in memory (via a BytesIO buffer).
      2. gzip.compress those bytes.
      3. Write the compressed bytes to out_path.

    gzip typically achieves 30–50% additional compression on top of float16,
    because many weight tensors have repeated or near-zero values.
    """
    import io                                      # BytesIO: in-memory byte stream for torch.save

    buf = io.BytesIO()                             # create an in-memory buffer
    torch.save(state_dict, buf)                    # serialize state dict into the buffer (not to disk)
    raw_bytes = buf.getvalue()                     # extract the raw serialized bytes

    compressed = gzip.compress(raw_bytes, compresslevel=9)  # maximum gzip compression (level 9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  # ensure output dir exists
    with open(out_path, "wb") as f:               # open output file in binary-write mode
        f.write(compressed)                        # write compressed bytes to disk

    size_mb = os.path.getsize(out_path) / 1024 ** 2        # convert bytes → megabytes
    limit_mb = 16.0                                         # challenge hard limit
    status = "✓ FITS" if size_mb <= limit_mb else "✗ TOO LARGE"
    print(f"[compress] {out_path} → {size_mb:.2f} MB  {status}")
    return size_mb


# ── Load for inference ─────────────────────────────────────────────────────────

def load_compressed(out_path: str, cfg: dict, device: str = "cpu") -> DualJEPAModel:
    """
    Reverse of save_compressed: decompress and deserialize back into a DualJEPAModel.
    The JEPA head will be absent from the state dict, so we instantiate the model
    and load with strict=False to allow missing keys.
    """
    import io                                      # BytesIO for in-memory decompression

    with open(out_path, "rb") as f:               # open compressed artifact
        compressed = f.read()                      # read all compressed bytes

    raw_bytes = gzip.decompress(compressed)        # decompress back to raw serialized bytes
    buf = io.BytesIO(raw_bytes)                    # wrap in a stream for torch.load
    state_dict = torch.load(buf, map_location=device)  # deserialize to CPU (or device of choice)

    model = DualJEPAModel(                         # rebuild model skeleton using config values
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_sym=cfg["model"]["d_sym"],
    ).to(device)

    model.load_state_dict(state_dict, strict=False)  # strict=False: silently accept missing jepa_head keys
    model.eval()                                   # switch to inference mode (disables any dropout)
    return model


# ── Main pipeline ──────────────────────────────────────────────────────────────

def compress(checkpoint_path: str, output_path: str, cfg: dict):
    device = "cpu"                                 # always compress from CPU to avoid device mismatches

    print(f"[compress] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)  # load raw checkpoint dict
    state_dict = ckpt["model"]                     # extract only the model weights (not optimizer state)

    state_dict = strip_jepa_head(state_dict)       # remove JEPA head keys — not needed at inference
    state_dict = quantize_to_fp16(state_dict)      # cast all float tensors to fp16 to halve size

    size_mb = save_compressed(state_dict, output_path)  # gzip and write to disk; prints size + status
    return size_mb


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress trained model to ≤16MB artifact")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt",  # path to trained checkpoint
                        help="Path to the .pt checkpoint file produced by train.py")
    parser.add_argument("--output", default="artifact/model.pt.gz",      # where to write the artifact
                        help="Destination path for the compressed artifact")
    parser.add_argument("--config", default="configs/base.yaml",         # config for model architecture
                        help="Path to YAML config (needed to reconstruct model for verification)")
    args = parser.parse_args()                     # parse flags from the command line

    with open(args.config) as f:                   # load config so compress() knows model shape
        cfg = yaml.safe_load(f)

    compress(args.checkpoint, args.output, cfg)    # run the full compress pipeline
