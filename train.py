# --- START OF HIGHLY SIMPLIFIED train.py (No WandB, No DDP, No Compile Logic) ---

"""
Highly simplified training script focused on a single-process, non-compiled run.
Assumes a compatible model definition exists in model.py.
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial, lru_cache

import torch
import torch.nn.functional as F # Needed for estimate_loss fallback

# Import model definition from model.py
from model import Transformer, ModelArgs

# NOTE: Removed DDP imports
# from torch.distributed import destroy_process_group, init_process_group
# from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task # Uses tinystories dataloader
from export import model_export

# -----------------------------------------------------------------------------
# Helper function
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

# -----------------------------------------------------------------------------
# Default configuration
out_dir = "out"          # Output directory for checkpoints and logs
eval_interval = 100      # How often to evaluate the model on the validation set
log_interval = 10        # How often to print training logs
eval_iters = 50          # Number of batches to use for estimating loss
eval_only = False        # If True, run evaluation and exit without training
always_save_checkpoint = True  # If True, save a checkpoint on every validation improvement
init_from = "scratch"    # Start from scratch or 'resume' from a checkpoint
# data
vocab_source = "custom"  # Source of the vocabulary
vocab_size = 128         # Vocabulary size (must match tokenizer)
batch_size = 32          # Number of sequences in a single batch
max_seq_len = 256        # Maximum length of a sequence
# model
dim = 128                # The dimension of the model's embeddings
n_layers = 5             # Number of transformer layers
n_heads = 8              # Number of attention heads
n_kv_heads = 4           # Number of key/value heads (for Grouped Query Attention)
multiple_of = 32         # Ensures hidden dimensions are multiples of this value for efficiency
dropout = 0.0            # Dropout rate for regularization (0.0 means no dropout)
# adamw optimizer
gradient_accumulation_steps = 4  # How many steps to accumulate gradients over before updating weights
base_learning_rate = 5e-4    # The starting learning rate
weight_decay = 1e-1          # Weight decay for regularization
beta1 = 0.9                  # AdamW beta1 parameter
beta2 = 0.95                 # AdamW beta2 parameter
grad_clip = 1.0              # Gradient clipping to prevent exploding gradients
adam_eps = 1e-10             # Epsilon value for AdamW to prevent division by zero
# learning rate decay settings
decay_lr = True              # Whether to decay the learning rate
max_iters = 100              # Total number of training iterations
min_lr = base_learning_rate / 10.0  # The minimum learning rate to decay to
cooldown_frac = 0.1          # Fraction of iterations to spend in the cooldown phase
# system
device = "cuda"              # The device to train on ('cuda' or 'cpu')
dtype = "float16"            # The data type for training ('float16', 'bfloat16', 'float32')
compile = True               # Whether to compile the model with torch.compile for speed
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# configurator.py is ESSENTIAL for applying your command-line arguments
try:
    exec(open("configurator.py").read())
except FileNotFoundError:
    print("WARNING: configurator.py not found. Using default settings.")
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

compile = config['compile']


# Derived hyperparameters
stable_iters = int(max_iters * (1.0 - cooldown_frac))
padded_vocab_size = next_multiple_of_n(vocab_size, n=128)

# --- DDP Setup Removed ---
master_process = True # Always master process in non-DDP run
seed_offset = 0
ddp_world_size = 1
# gradient_accumulation_steps remains unchanged as world_size is 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"Starting run in {out_dir}")
    print(f"Vocab size (original/padded): {vocab_size}/{padded_vocab_size}")
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"Max iterations: {max_iters:,}")
    print(f"Compiling: {compile}") # Print final compile status
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
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

# Init model state
iter_num = 0
best_val_loss = 1e9

# Model initialization
model_args = dict(
    dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
    vocab_size=padded_vocab_size, multiple_of=multiple_of,
    max_seq_len=max_seq_len, dropout=dropout, norm_eps=1e-5, # Assuming norm_eps=1e-5 in model.py
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    # Simplified resume logic (only loads model weights, not optimizer)
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # TODO: Add more robust checkpoint arg loading if needed
    gptconf = ModelArgs(**checkpoint.get("model_args", model_args)) # Use checkpoint args if available
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod." # In case compiled model was saved
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    checkpoint = None # Free memory
    print("WARNING: Resuming only model weights. Optimizer starts from scratch.")


model.to(device)

# Optimizer setup
optimizer = model.configure_optimizers(weight_decay, base_learning_rate, (beta1, beta2), adam_eps, device_type)
# NOTE: Optimizer state is NOT loaded when resuming in this simplified version

# Ensure initial_lr is stored
for group in optimizer.param_groups:
    if 'initial_lr' not in group:
         group['initial_lr'] = group['lr']

if compile:
    print("Compiling the model... (this may take a ~minute)")
    try:
        model = torch.compile(model) # Requires PyTorch 2.0+
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")
        # Optionally, set compile to False if it fails, to avoid issues later
        # compile = False 

# --- DDP Wrapping Removed ---
raw_model = model # No DDP wrapping, raw_model is just model

# GradScaler
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

# Evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters, device='cpu')
        for k in range(eval_iters):
            try: X, Y = next(batch_iter)
            except StopIteration: losses = losses[:k]; break
            with ctx:
                 # Assumes model calculates loss internally and stores in raw_model.last_loss
                 _ = model(X, Y)
                 if hasattr(raw_model, 'last_loss') and raw_model.last_loss is not None:
                     loss = raw_model.last_loss
                     losses[k] = loss.item()
                 else: # Fallback if last_loss isn't set
                     print(f"Warning: raw_model.last_loss not set during {split} eval step {k}. Recalculating.")
                     logits = model(X) # Expect logits if no target
                     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                     losses[k] = loss.item()

        out[split] = losses.mean().item() if losses.numel() > 0 else float('inf')
    model.train()
    return out

# LR decay scheduler
@lru_cache(1)
def get_lr_multiplier(step: int) -> float:
    if step < stable_iters: return 1.0
    if step > max_iters: return min_lr / base_learning_rate
    decay_ratio = (step - stable_iters) / (max_iters - stable_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    final_lr_ratio = (min_lr / base_learning_rate) + coeff * (1.0 - (min_lr / base_learning_rate))
    return final_lr_ratio

# --- WandB Logging Removed ---

# Training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0

print("Starting training loop...")
while True:

    # Determine and set learning rate
    lr_mult = get_lr_multiplier(iter_num) if decay_lr else 1.0
    current_lr = -1.0
    for param_group in optimizer.param_groups:
        initial_lr = param_group.get('initial_lr', param_group['lr'])
        param_group["lr"] = initial_lr * lr_mult
        if current_lr < 0: current_lr = param_group["lr"]

    # Evaluate loss and save checkpoints (master_process is always True here)
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # --- WandB Log Call Removed ---

        if losses["val"] < best_val_loss or always_save_checkpoint:
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                print(f"New best val loss: {best_val_loss:.4f}")
                # Simplified checkpoint, omits optimizer state
                checkpoint = {
                    "model": raw_model.state_dict(), "model_args": model_args,
                    "iter_num": iter_num, "best_val_loss": best_val_loss, "config": config,
                }
                ckpt_path = os.path.join(out_dir, "ckpt.pt")
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)
                try:
                    model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
                except Exception as e:
                    print(f"Model export failed: {e}")

    if eval_only: break
    if iter_num >= max_iters: break

    # Forward-backward update
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        # --- DDP Sync Logic Removed ---
        with ctx:
            _ = model(X, Y) # Model forward pass stores loss
            if hasattr(raw_model, 'last_loss') and raw_model.last_loss is not None:
                loss = raw_model.last_loss / gradient_accumulation_steps
            else:
                print("Error: Model did not store loss.")
                loss = torch.tensor(0.0, device=device, requires_grad=True) # Dummy loss to prevent crash

        try: X_next, Y_next = next(train_batch_iter)
        except StopIteration:
             print("Warning: Training data iterator exhausted. Resetting.")
             train_batch_iter = iter_batches(split="train"); X_next, Y_next = next(train_batch_iter)

        scaler.scale(loss).backward()
        X, Y = X_next, Y_next

    # Clip gradients
    if grad_clip > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        try:
            loss_val = loss.item()
            lossf = loss_val * gradient_accumulation_steps
            print(f"{iter_num} | loss {lossf:.4f} | lr {current_lr:.2e} | {dt*1000:.2f}ms")
        except NameError: # Handle case where 'loss' wasn't defined
             print(f"{iter_num} | loss ERROR | lr {current_lr:.2e} | {dt*1000:.2f}ms")

    iter_num += 1
    local_iter_num += 1

# --- DDP Cleanup Removed ---

print("Training finished.")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

# --- END OF HIGHLY SIMPLIFIED train.py ---
