# --- START OF model.py (relu² FFN, no-param RMSNorm, RoPE, GQA) ---

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    """
    Model configuration arguments. Defaults resemble Llama 7B but are typically overridden.
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None # For Grouped Query Attention (GQA)
    vocab_size: int = 32000         # Usually set dynamically based on tokenizer + padding
    hidden_dim: Optional[int] = None # Calculated by FeedForward if None
    multiple_of: int = 256          # Makes FFN hidden layer size a multiple of this
    norm_eps: float = 1e-5          # RMSNorm epsilon, consistent with Llama 2
    max_seq_len: int = 2048          # Max sequence length for RoPE precomputation
    dropout: float = 0.0            # Dropout rate


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the rotary frequency positional embeddings (RoPE)
    cis(theta*m) = cos(theta*m) + i*sin(theta*m) represented by separate cos/sin tensors.
    """
    # Calculate frequencies based on dimension and theta
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create sequence position indices [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # Compute outer product: (seq_len, dim/2) tensor where freqs[i, j] = i * freqs[j]
    freqs = torch.outer(t, freqs).float()
    # Compute cosine and sine values: (seq_len, dim/2) each
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes frequency tensors to allow broadcasting with input query/key tensors.
    Input x shape: (batch, seq_len, heads, head_dim)
    Output freqs shape: (1, seq_len, 1, head_dim/2 or head_dim if complex)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # Ensure freqs shape matches (seq_len, head_dim_feature_part)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape=} vs {(x.shape[1], x.shape[-1])=}"
    # Get shape for broadcasting: [1, seq_len, 1, head_dim_feature_part]
    shape = [1 if i != 1 and i != ndim - 1 else d for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embedding (RoPE) to query and key tensors.
    """
    # Reshape xq/xk to separate real/imaginary parts (handled as pairs of features)
    # xq: (bs, seqlen, n_heads, head_dim) -> (bs, seqlen, n_heads, head_dim/2, 2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape freqs_cos/sin for broadcasting: (1, seqlen, 1, head_dim/2)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotation using real components
    # Formula: (x_r + i*x_i) * (c + i*s) = (x_r*c - x_i*s) + i*(x_r*s + x_i*c)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine back into (bs, seqlen, n_heads, head_dim) shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    # Cast back to original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value tensors for Grouped Query Attention (GQA).
    If n_rep=1 (Multi-Head Attention), returns input unchanged.
    Otherwise, duplicates KV heads to match the number of query heads.
    Input x shape: (bs, seqlen, n_kv_heads, head_dim)
    Output shape: (bs, seqlen, n_heads, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # Expand and reshape: Insert new dim, expand, reshape to final size
    return (
        x[:, :, :, None, :] # (bs, slen, n_kv_heads, 1, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # (bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # (bs, slen, n_heads, head_dim)
    )

class Attention(nn.Module):
    """
    Multi-Head or Grouped Query Attention module.
    Uses RoPE for positional embeddings and supports Flash Attention 2 if available.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads # Number of query heads
        # Determine number of key/value heads for GQA/MQA
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        # Repetition factor for repeating KV heads in GQA
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Linear layers for Q, K, V projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Output projection layer
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Dropout layers
        self.attn_dropout = nn.Dropout(args.dropout) # Dropout on attention scores (for non-Flash)
        self.resid_dropout = nn.Dropout(args.dropout) # Dropout on output projection
        self.dropout = args.dropout # General dropout rate

        # Check for Flash Attention 2 support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Precompute causal mask for manual attention implementation
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape # Input shape: (batch_size, sequence_length, dimension)

        # Project input to Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape Q, K, V for multi-head structure
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE positional embeddings *before* GQA repetition
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Repeat KV heads if using GQA/MQA (n_rep > 1)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for attention calculation: (bs, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Perform attention calculation
        if self.flash:
            # Use Flash Attention 2 (efficient, handles masking and dropout internally)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None, # Causal mask is implicit
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention calculation (less efficient)
            # (bs, n_heads, seqlen, head_dim) @ (bs, n_heads, head_dim, seqlen) -> (bs, n_heads, seqlen, seqlen)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # Apply causal mask
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            # Softmax and dropout on scores
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # (bs, n_heads, seqlen, seqlen) @ (bs, n_heads, seqlen, head_dim) -> (bs, n_heads, seqlen, head_dim)
            output = torch.matmul(scores, xv)

        # Restore shape: (bs, seqlen, n_heads, head_dim) -> (bs, seqlen, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Final output projection
        output = self.wo(output)
        output = self.resid_dropout(output) # Apply residual dropout
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward network using relu² activation.
    relu²(x) = relu(x)²  — simpler and faster than SwiGLU.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int], multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x)).square()))


class TransformerBlock(nn.Module):
    """
    A single block of the Transformer model, combining Attention and FeedForward layers.
    Uses pre-normalization (Norm -> Layer -> Add) structure.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        # Normalization layers before Attention and FeedForward
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        """ Forward pass with residual connections. """
        # Attention part: Residual + Attention(Norm(x))
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        # FeedForward part: Residual + FeedForward(Norm(h))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    The main Transformer model class.
    """
    last_loss: Optional[torch.Tensor] # Stores loss from the last forward pass if targets are provided

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size # Note: This should be the *padded* vocab size
        self.n_layers = params.n_layers

        # Token embeddings
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # Input dropout
        self.dropout = nn.Dropout(params.dropout)
        # Transformer layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # Final normalization layer
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # Output linear layer (logits head)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Weight Tying: Share weights between token embeddings and final output layer
        # See: https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # Precompute RoPE frequencies for max sequence length
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        # Register as non-persistent buffers (not saved in state_dict)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize all weights using standard method first
        self.apply(self._init_weights)

        # === Zero-Initialize Residual Projections (Modification) ===
        # This overrides the standard init for wo.weight and w2.weight.
        # Zero-init helps stabilize training, especially early on, by ensuring
        # residual blocks initially act closer to identity functions.
        with torch.no_grad():
            for name, p in self.named_parameters():
                # Target Attention output projection (wo) and FFN down-projection (w2)
                if name.endswith('wo.weight') or name.endswith('w2.weight'):
                     p.zero_()
                     # print(f"Zero-initialized {name}") # Optional: for verification
        # =============================================================

        # Initialize attribute for storing the loss
        self.last_loss = None

    def _init_weights(self, module):
        """ Standard weight initialization for linear and embedding layers. """
        if isinstance(module, nn.Linear):
            # Normal init with std=0.02 (common practice for Transformers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Biases (if they existed) would be zero-initialized
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal init for embeddings
            # Note: Due to weight tying, this also initializes self.output.weight
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        Calculates logits and optionally computes the loss if targets are provided.
        """
        _bsz, seqlen = tokens.shape
        # Token embeddings + dropout
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # Get RoPE frequencies for the current sequence length
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # Final normalization
        h = self.norm(h)

        # Calculate output logits
        if targets is not None:
            # Training or validation with loss calculation
            logits = self.output(h) # Calculate logits for the entire sequence
            # Calculate cross-entropy loss
            # Loss is calculated on the *padded* vocab size dimension
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: Optimize by calculating logits only for the last token
            # Keep the sequence dimension using list slicing [-1]
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits # Return logits (raw, without softcap in this version)

    def configure_optimizers(self, weight_decay, learning_rate, betas, adam_eps, device_type):
        """
        Configures the AdamW optimizer with parameter groups for different LR/WD settings.
        This setup is inspired by practices observed in other successful training recipes.

        Args:
            weight_decay: Base weight decay value for hidden matrices.
            learning_rate: Base learning rate, multipliers applied per group.
            betas: Adam betas (e.g., (0.9, 0.95)).
            adam_eps: Adam epsilon (e.g., 1e-10 or 1e-8).
            device_type: 'cuda' or 'cpu'.
        """
        # Start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # --- Parameter Grouping Strategy ---
        # 1. Tied Embeddings/Output Head: Typically performs better without weight decay.
        # 2. Norms & Scalars (1D params like biases, norm gains): No weight decay; often benefit from lower LR.
        # 3. Hidden Matrices (>=2D weights in Linear layers, excluding tied embed/head): Apply weight decay here.

        # Identify the single tied weight tensor (embedding & output head)
        # It contains 'tok_embeddings.weight' in its name.
        embed_head_params = [p for n, p in param_dict.items() if "tok_embeddings.weight" in n]

        # Identify any 1D parameters (scalars/biases); RMSNorm has no learnable params so this group may be empty
        scalar_norm_params = [p for n, p in param_dict.items() if p.ndim < 2 or "norm.weight" in n]

        # Identify all other >= 2D weights (these are the hidden matrices in Attention/FFN)
        hidden_matrix_params = [
            p for n, p in param_dict.items()
            if p.ndim >= 2
            and "tok_embeddings.weight" not in n # Exclude the tied weight
            and "norm.weight" not in n          # Exclude norm weights (already in scalar_norm_params)
        ]

        # Sanity check: Ensure all parameters are assigned to exactly one group
        num_params_total = sum(p.numel() for p in param_dict.values())
        num_params_grouped = sum(p.numel() for p in embed_head_params) + \
                             sum(p.numel() for p in scalar_norm_params) + \
                             sum(p.numel() for p in hidden_matrix_params)
        assert num_params_total == num_params_grouped, f"Parameter mismatch: {num_params_total} != {num_params_grouped}"
        assert len(param_dict) == len(embed_head_params) + len(scalar_norm_params) + len(hidden_matrix_params), "Tensor count mismatch"

        # Define LR multipliers for each group (can be tuned)
        lr_mult_embed_head = 1.0 # Example: Use base LR
        lr_mult_hidden = 1.0     # Example: Use base LR
        lr_mult_scalar_norm = 0.1 # Example: Use 10x lower LR for norms/scalars

        # Create optimizer groups with specific settings
        optim_groups = [
            {"params": embed_head_params,    "lr": learning_rate * lr_mult_embed_head, "weight_decay": 0.0}, # No WD for tied embed/head
            {"params": scalar_norm_params,   "lr": learning_rate * lr_mult_scalar_norm,"weight_decay": 0.0}, # No WD, low LR for norms/scalars
            {"params": hidden_matrix_params, "lr": learning_rate * lr_mult_hidden,     "weight_decay": weight_decay}, # Apply WD here
        ]

        # Log the parameter group setup
        print("Optimizer groups:")
        total_params = 0
        for i, group in enumerate(optim_groups):
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            # Use group['lr'] directly as initial_lr might not be set yet
            lr_mult = group['lr'] / learning_rate if learning_rate > 0 else 0
            print(f"  Group {i}: {group_params:,} params, LR_mult={lr_mult:.2f}, WD={group['weight_decay']}")
        print(f"Total optimizable parameters: {total_params:,}")

        # Create AdamW optimizer, preferring fused implementation if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, eps=adam_eps, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        # Store the initial learning rate in each group dictionary; needed by the LR scheduler in train.py
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate Model Flops Utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        Provides an approximate measure of hardware efficiency.
        """
        # Estimate number of FLOPs per iteration using PaLM paper appendix B formula
        N = sum(p.numel() for p in self.parameters()) # Total parameters
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        # Approx formula: 6*N*T (matmuls) + 12*L*H*Q*T^2 (attention) -- simplified here assuming T=max_seq_len
        # Simplified PaLM formula: assumes T is factored out for flops/token
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T # Multiply by sequence length
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter # Multiply by samples per iteration

        # Calculate achieved FLOPS per second
        flops_achieved = flops_per_iter * (1.0/dt)
        # Reference peak FLOPS for A100 bfloat16 (adjust if using different hardware)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates text sequences based on a conditioning sequence `idx`.
        Simple implementation without Key/Value caching (inefficient for long sequences).
        """
        self.eval() # Set model to evaluation mode
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max sequence length
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # Forward pass to get logits for the next token
            logits = self(idx_cond) # Calls forward with targets=None
            logits = logits[:, -1, :] # Get logits for the last token only

            # Apply temperature scaling
            if temperature == 0.0:
                # Greedy sampling: pick the single most likely token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                # Optional Top-K sampling: Restrict sampling to top k most likely tokens
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Set logits below the k-th threshold to -infinity
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # Convert logits to probabilities using softmax
                probs = F.softmax(logits, dim=-1)
                # Sample the next token index from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token index to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train() # Restore model to training mode if needed elsewhere
        return idx

# --- END OF model.py ---
