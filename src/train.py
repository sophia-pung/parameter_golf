import os                           # for creating checkpoint directories
import sys                          # for platform detection (macOS DataLoader fix)
import math                         # for learning rate cosine schedule calculation
import argparse                     # for parsing command-line flags
import yaml                         # for reading configs/base.yaml
from pathlib import Path            # cross-platform filesystem paths

import torch                        # core PyTorch library
import torch.nn.functional as F     # stateless ops like cross_entropy and normalize
from torch.utils.data import DataLoader  # batches and shuffles dataset samples

from model import DualJEPAModel     # our dual-stream + JEPA model architecture
from data import TextDataset        # dataset class that chunks raw text into token windows


# ── Loss function ──────────────────────────────────────────────────────────────

def compute_loss(model, ids, lambda_jepa=0.1):
    # ids shape: (B, T) — a batch of token-ID sequences
    logits, h, jepa_pred = model(ids)              # run both LM and JEPA heads in one forward pass

    # L_LM: cross-entropy over every position except the last
    # logits[:, :-1] are predictions for positions 0..T-2
    # ids[:, 1:]     are the ground-truth tokens  at positions 1..T-1
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),   # flatten to (B*(T-1), vocab_size) for cross_entropy
        ids[:, 1:].reshape(-1),                        # flatten to (B*(T-1),) ground-truth token IDs
    )

    # L_JEPA: predict the *representation* of the next position, not the raw token
    # This pushes hidden states to encode predictive, structured information
    target = h[:, 1:].detach()                     # (B, T-1, D): future hidden states, no gradient
    pred = jepa_pred[:, :-1]                       # (B, T-1, D): JEPA head's predictions for those states
    jepa_loss = F.mse_loss(
        F.normalize(pred, dim=-1),                 # L2-normalize so loss is cosine-based (prevents scale collapse)
        F.normalize(target, dim=-1),               # L2-normalize target for the same reason
    )

    total = lm_loss + lambda_jepa * jepa_loss      # weighted sum: λ controls JEPA regularization strength
    return total, lm_loss, jepa_loss               # return all three so we can log each separately


# ── Learning rate schedule ─────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # Linear warmup: ramp lr from 0 to max_lr over the first warmup_steps
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # Cosine decay: smoothly decay from max_lr to min_lr for the rest of training
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)  # 0 → 1 over training
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))                  # 1 → 0 following cosine curve
    return min_lr + cosine * (max_lr - min_lr)                            # scale to [min_lr, max_lr]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create checkpoints/ dir if it doesn't exist
    torch.save({
        "step": step,                              # current training step for resuming
        "model": model.state_dict(),               # all learned parameters
        "optimizer": optimizer.state_dict(),       # optimizer moments (needed to resume correctly)
    }, path)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)   # load onto the correct device directly
    model.load_state_dict(ckpt["model"])            # restore model weights
    optimizer.load_state_dict(ckpt["optimizer"])    # restore optimizer state
    return ckpt["step"]                             # return step so training can resume from here


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg):
    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available()        # prefer GPU if available
        else "mps" if torch.backends.mps.is_available()  # Apple Silicon GPU second choice
        else "cpu"                                 # fall back to CPU
    )
    print(f"[device] using {device}")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = TextDataset(
        path=cfg["data"]["path"],                  # path to raw .txt training corpus
        seq_len=cfg["model"]["seq_len"],           # number of tokens per training window
        vocab_size=cfg["model"]["vocab_size"],     # tokenizer vocabulary size
    )
    # num_workers=0 on macOS: multiprocessing DataLoader workers deadlock on macOS
    # due to how the OS handles forked processes. 0 means data loads in the main process.
    num_workers = 0 if sys.platform == "darwin" else 2
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],     # number of sequences per gradient step
        shuffle=True,                              # shuffle order of windows each epoch
        num_workers=num_workers,                   # 0 on macOS to avoid multiprocessing deadlock
        pin_memory=(device == "cuda"),             # pin memory for faster GPU transfer
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DualJEPAModel(
        vocab_size=cfg["model"]["vocab_size"],     # number of tokens in the tokenizer vocabulary
        d_model=cfg["model"]["d_model"],           # hidden dimension of the transformer
        n_layers=cfg["model"]["n_layers"],         # number of stacked transformer blocks
        n_heads=cfg["model"]["n_heads"],           # attention heads per block
        d_sym=cfg["model"]["d_sym"],               # Stream B auxiliary embedding dimension
    ).to(device)                                   # move all parameters to the target device

    total_params = sum(p.numel() for p in model.parameters())  # count all trainable parameters
    print(f"[model] {total_params:,} parameters")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["max_lr"],                 # initial LR; scheduler overrides this each step
        weight_decay=cfg["train"]["weight_decay"], # L2 regularization to reduce overfitting
        betas=(0.9, 0.95),                         # standard AdamW betas for LLM training
    )

    # ── Optional: resume from checkpoint ─────────────────────────────────────
    start_step = 0                                 # default: start from scratch
    ckpt_path = cfg["train"].get("resume_from")    # path is None if not set in config
    if ckpt_path and os.path.exists(ckpt_path):
        start_step = load_checkpoint(model, optimizer, ckpt_path, device)
        print(f"[resume] starting from step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()                                  # enable dropout, train-time behaviour
    step = start_step                              # global step counter (persists across epochs)
    max_steps = cfg["train"]["max_steps"]          # total number of gradient updates to perform

    for epoch in range(cfg["train"]["epochs"]):    # outer loop over full passes through the data
        for ids in loader:                         # inner loop over mini-batches
            if step >= max_steps:                  # hard stop at max_steps regardless of epoch
                break

            ids = ids.to(device)                   # move token IDs to GPU/MPS/CPU

            # Adjust learning rate according to warmup + cosine schedule
            lr = get_lr(
                step,
                warmup_steps=cfg["train"]["warmup_steps"],
                max_steps=max_steps,
                max_lr=cfg["train"]["max_lr"],
                min_lr=cfg["train"]["min_lr"],
            )
            for pg in optimizer.param_groups:      # apply new lr to all parameter groups
                pg["lr"] = lr

            optimizer.zero_grad()                  # clear gradients from the previous step

            loss, lm_loss, jepa_loss = compute_loss(
                model, ids,
                lambda_jepa=cfg["train"]["lambda_jepa"],  # JEPA loss weight from config
            )

            loss.backward()                        # compute gradients via backpropagation

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train"]["grad_clip"],         # clip gradient norm to prevent exploding gradients
            )

            optimizer.step()                       # update parameters using computed gradients
            step += 1                              # increment global step counter

            # Logging: print metrics every log_every steps
            if step % cfg["train"]["log_every"] == 0:
                bits_per_token = lm_loss.item() / math.log(2)   # convert nats → bits for interpretability
                print(
                    f"step {step:6d} | lr {lr:.2e} | "
                    f"loss {loss.item():.4f} | lm {lm_loss.item():.4f} | "
                    f"jepa {jepa_loss.item():.4f} | bpt {bits_per_token:.3f}"
                )

            # Save a checkpoint every save_every steps
            if step % cfg["train"]["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, step,
                    path=f"checkpoints/step_{step:06d}.pt",
                )

        if step >= max_steps:                      # exit epoch loop when max_steps reached
            break

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, step, path="checkpoints/final.pt")
    print("[done] training complete")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualJEPA model")
    parser.add_argument("--config", default="configs/base.yaml",  # default config file path
                        help="Path to YAML config file")
    args = parser.parse_args()                     # parse command-line arguments

    with open(args.config) as f:                   # open and parse the YAML config
        cfg = yaml.safe_load(f)

    train(cfg)                                     # kick off training with loaded config
