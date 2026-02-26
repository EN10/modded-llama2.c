"""
Training script for a small Llama-style Transformer on TinyStories.
Single-process only. Supports training from scratch or resuming from checkpoint.
"""

import math
import os
import time
from contextlib import nullcontext
from functools import partial

import torch

from model import Transformer, ModelArgs
from tinystories import Task
from export import model_export

# -----------------------------------------------------------------------------
# Default configuration (overridable via configurator.py / CLI args)
out_dir = "out"
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
# data
vocab_source = "custom"
vocab_size = 128
batch_size = 32
max_seq_len = 256
# model
dim = 128
n_layers = 5
n_heads = 8
n_kv_heads = 4
multiple_of = 32
dropout = 0.0
# optimizer
gradient_accumulation_steps = 4
base_learning_rate = 5e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
adam_eps = 1e-10
# lr schedule
decay_lr = True
max_iters = 100
min_lr = base_learning_rate / 10.0
cooldown_frac = 0.1
# system
device = "cuda"
dtype = "float16"
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
try:
    exec(open("configurator.py").read())
except FileNotFoundError:
    print("WARNING: configurator.py not found. Using default settings.")
config = {k: globals()[k] for k in config_keys}
compile = config['compile']

# Derived values
stable_iters = int(max_iters * (1.0 - cooldown_frac))
padded_vocab_size = math.ceil(vocab_size / 128) * 128
tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len

print(f"Starting run in {out_dir}")
print(f"Vocab size (original/padded): {vocab_size}/{padded_vocab_size}")
print(f"Tokens per iteration: {tokens_per_iter:,}")
print(f"Max iterations: {max_iters:,}")
print(f"Compiling: {compile}")
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
iter_batches = partial(
    Task.iter_batches, batch_size=batch_size, max_seq_len=max_seq_len,
    vocab_size=vocab_size, vocab_source=vocab_source, device=device, num_workers=0,
)

# Model initialization
iter_num = 0
best_val_loss = 1e9
model_args = dict(
    dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
    vocab_size=padded_vocab_size, multiple_of=multiple_of,
    max_seq_len=max_seq_len, dropout=dropout, norm_eps=1e-5,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = Transformer(ModelArgs(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    model = Transformer(ModelArgs(**checkpoint.get("model_args", model_args)))
    state_dict = checkpoint["model"]
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    del checkpoint
    print("WARNING: Resuming only model weights. Optimizer starts from scratch.")

model.to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, base_learning_rate, (beta1, beta2), adam_eps, device_type)

# Compile
if compile:
    print("Compiling the model... (this may take a ~minute)")
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")

raw_model = model
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

# Evaluation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = next(batch_iter)
            except StopIteration:
                losses = losses[:k]
                break
            with ctx:
                model(X, Y)
                losses[k] = raw_model.last_loss.item()
        out[split] = losses.mean().item() if losses.numel() > 0 else float('inf')
    model.train()
    return out

# LR schedule: stable then cosine cooldown
def get_lr(step):
    if not decay_lr:
        return base_learning_rate
    if step < stable_iters:
        mult = 1.0
    elif step >= max_iters:
        mult = min_lr / base_learning_rate
    else:
        decay_ratio = (step - stable_iters) / (max_iters - stable_iters)
        mult = (min_lr / base_learning_rate) + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (1.0 - min_lr / base_learning_rate)
    return mult

# Training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()

print("Starting training loop...")
while True:
    # Set learning rate
    lr_mult = get_lr(iter_num)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_mult
    current_lr = optimizer.param_groups[0]["lr"]

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                print(f"New best val loss: {best_val_loss:.4f}")
                checkpoint = {
                    "model": raw_model.state_dict(), "model_args": model_args,
                    "iter_num": iter_num, "best_val_loss": best_val_loss, "config": config,
                }
                print(f"Saving checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                try:
                    model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
                except Exception as e:
                    print(f"Model export failed: {e}")

    if eval_only or iter_num >= max_iters:
        break

    # Forward-backward with gradient accumulation
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            model(X, Y)
            loss = raw_model.last_loss / gradient_accumulation_steps

        try:
            X, Y = next(train_batch_iter)
        except StopIteration:
            train_batch_iter = iter_batches(split="train")
            X, Y = next(train_batch_iter)

        scaler.scale(loss).backward()

    if grad_clip > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"{iter_num} | loss {lossf:.4f} | lr {current_lr:.2e} | {dt*1000:.2f}ms")

    iter_num += 1

print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
