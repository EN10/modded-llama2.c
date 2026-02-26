#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simplified and concise implementation for Llama-2-style Transformer inference.
This version refactors the original for clarity and brevity while maintaining functionality.
"""

import argparse
import time
import os
import struct
import sys
import re
from dataclasses import dataclass
from typing import List

import numpy as np

# ----------------------------------------------------------------------------
# Transformer model definition

@dataclass
class Config:
    """Configuration for the Transformer model."""
    dim: int; hidden_dim: int; n_layers: int; n_heads: int
    n_kv_heads: int; vocab_size: int; seq_len: int
    shared_weights: bool = False

    @classmethod
    def from_file(cls, f):
        """Reads configuration from a binary file."""
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack('iiiiiii', f.read(7 * 4))
        return cls(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                   abs(vocab_size), seq_len, shared_weights=(vocab_size > 0))

class Transformer:
    """The Transformer model, including weights and the forward pass logic."""
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.head_size = config.dim // config.n_heads
        
        with open(checkpoint_path, "rb") as f:
            f.seek(7 * 4) # Skip config header
            weights_data = np.fromfile(f, dtype=np.float32)

        self.weights = self._load_weights(weights_data)
        self.state = self._init_state()

    def _load_weights(self, data: np.ndarray) -> dict:
        """Slice and reshape weights from the flat data array."""
        cfg = self.config
        kv_dim = (cfg.dim * cfg.n_kv_heads) // cfg.n_heads
        
        # Define weight shapes in a declarative map
        weight_map = [
            ('token_embedding_table', (cfg.vocab_size, cfg.dim)),
            ('wq', (cfg.n_layers, cfg.dim, cfg.dim)),
            ('wk', (cfg.n_layers, kv_dim, cfg.dim)),
            ('wv', (cfg.n_layers, kv_dim, cfg.dim)),
            ('wo', (cfg.n_layers, cfg.dim, cfg.dim)),
            ('w1', (cfg.n_layers, cfg.hidden_dim, cfg.dim)),
            ('w2', (cfg.n_layers, cfg.dim, cfg.hidden_dim)),
        ]
        
        weights = {}
        p = 0
        for name, shape in weight_map:
            size = np.prod(shape)
            weights[name] = data[p : p + size].reshape(shape)
            p += size
        
        # Handle the final classifier weight, which might be shared
        p += cfg.seq_len * self.head_size # Skip RoPE frequencies
        if not cfg.shared_weights:
            weights['wcls'] = data[p:].reshape(cfg.vocab_size, cfg.dim)
        else:
            weights['wcls'] = weights['token_embedding_table']
            
        return weights

    def _init_state(self) -> dict:
        """Initialize the run state buffers for the forward pass."""
        cfg = self.config
        kv_dim = (cfg.dim * cfg.n_kv_heads) // cfg.n_heads
        
        # These buffers store the intermediate results of the computation (the "state")
        # for a single forward pass. They are initialized once and reused for each token,
        # which is more memory-efficient than reallocating them on every call.
        state_map = {
            'x': (cfg.dim,),             # current token's embedding
            'xb': (cfg.dim,),            # buffer for intermediate results
            'xb2': (cfg.dim,),           # another buffer for intermediate results
            'hb': (cfg.hidden_dim,),     # buffer for the hidden state of the FFN

            'q': (cfg.dim,),             # query vector for the current token
            'key_cache': (cfg.n_layers, cfg.seq_len, kv_dim),    # cache for key vectors for all tokens
            'value_cache': (cfg.n_layers, cfg.seq_len, kv_dim), # cache for value vectors for all tokens
            'logits': (cfg.vocab_size,), # output logits for the next token prediction
        }
        return {name: np.zeros(shape, dtype=np.float32) for name, shape in state_map.items()}

    def forward(self, token: int, pos: int) -> np.ndarray:
        """
        The forward pass of the transformer for a single token.
        This function calculates the logits for the next token.
        """
        cfg, w, s = self.config, self.weights, self.state
        x = s['x']
        kv_dim = (cfg.dim * cfg.n_kv_heads) // cfg.n_heads
        kv_mul = cfg.n_heads // cfg.n_kv_heads

        # 1. Token Embedding: Retrieve the embedding vector for the current token.
        x[:] = w['token_embedding_table'][token]

        # 2. Transformer Layers: Process the token through each layer of the transformer.
        for l in range(cfg.n_layers):
            # 2a. RMS Normalization (before Attention)
            # Normalizes the input to have a stable distribution, improving training stability.
            s['xb'] = x / np.sqrt(np.mean(x**2) + 1e-5)

            # 2b. Q, K, V Calculation
            # Project the normalized input into Query, Key, and Value vectors.
            s['q'] = s['xb'] @ w['wq'][l].T # Query: what I am looking for
            k = s['xb'] @ w['wk'][l].T      # Key: what I contain
            v = s['xb'] @ w['wv'][l].T      # Value: what I offer
            
            # 2c. Rotary Positional Embedding (RoPE)
            # Injects positional information by rotating the Q and K vectors.
            for i in range(0, cfg.dim, 2):
                head_dim = i % self.head_size
                freq = 1.0 / (10000.0 ** (head_dim / self.head_size))
                val = pos * freq
                fcr, fci = np.cos(val), np.sin(val)
                if i < kv_dim: # Apply RoPE to key vectors
                    k[i:i+2] = k[i] * fcr - k[i+1] * fci, k[i] * fci + k[i+1] * fcr
                s['q'][i:i+2] = s['q'][i] * fcr - s['q'][i+1] * fci, s['q'][i] * fci + s['q'][i+1] * fcr

            # 2d. KV Cache: Store the current token's Key and Value vectors.
            # This avoids re-computation for previous tokens in the sequence.
            s['key_cache'][l, pos], s['value_cache'][l, pos] = k, v
            
            # 2e. Multi-Head Attention
            # Allows the model to focus on different parts of the input sequence.
            s['xb'][:] = 0
            for h in range(cfg.n_heads):
                # Get the query for the current head
                q_h = s['q'][h * self.head_size:(h + 1) * self.head_size]
                # Get the keys and values from the cache for all previous tokens
                k_cache_h = s['key_cache'][l, :pos+1, (h//kv_mul)*self.head_size:(h//kv_mul+1)*self.head_size]
                v_cache_h = s['value_cache'][l, :pos+1, (h//kv_mul)*self.head_size:(h//kv_mul+1)*self.head_size]

                # Calculate attention scores (how much each token should attend to every other token)
                scores = (q_h @ k_cache_h.T) / np.sqrt(self.head_size)
                # Apply softmax to get attention weights
                scores = np.exp(scores - np.max(scores))
                att = scores / np.sum(scores)
                
                # Weight the value vectors by the attention weights and sum them up
                s['xb'][h*self.head_size:(h+1)*self.head_size] = att @ v_cache_h

            # 2f. Output Projection and Residual Connection
            # Project the attention output back to the main dimension and add it to the original input.
            x += s['xb'] @ w['wo'][l].T

            # 2g. Feed-Forward Network (FFN)
            # A two-layer neural network that provides additional modeling capacity.
            s['xb'] = x / np.sqrt(np.mean(x**2) + 1e-5) # RMSNorm
            h1 = s['xb'] @ w['w1'][l].T # First linear layer
            h1 = np.maximum(h1, 0) ** 2 # relu² activation
            x += h1 @ w['w2'][l].T # Down projection and residual connection

        # 3. Final Normalization and Classifier
        # Apply final normalization and project to the vocabulary space to get logits.
        x = x / np.sqrt(np.mean(x**2) + 1e-5)
        s['logits'] = x @ w['wcls'].T
        return s['logits']

class Tokenizer:
    """BPE Tokenizer that translates strings to/from token IDs."""
    def __init__(self, tokenizer_path: str, vocab_size: int):
        self.vocab, self.vocab_scores = [], []
        with open(tokenizer_path, 'rb') as f:
            f.read(4) # Skip max_token_length
            for i in range(vocab_size):
                self.vocab_scores.append(struct.unpack('f', f.read(4))[0])
                str_len = struct.unpack('i', f.read(4))[0]
                self.vocab.append(f.read(str_len).decode('utf-8', errors='ignore'))
        
        self.vocab_map = {s: i for i, s in enumerate(self.vocab)}
        # Pre-compile a regex for fast initial tokenization.
        # This finds all occurrences of vocabulary tokens in the input text.
        sorted_vocab = sorted(self.vocab, key=len, reverse=True)
        self.regex_pattern = re.compile('|'.join(re.escape(s) for s in sorted_vocab))

    def decode(self, prev_token: int, token: int) -> str:
        piece = self.vocab[token]
        if prev_token == 1 and piece.startswith(' '): piece = piece[1:]
        if piece.startswith('<0x') and piece.endswith('>'):
            try: return chr(int(piece[3:5], 16))
            except (ValueError, IndexError): pass
        return piece

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        # Fast initial tokenization using pre-compiled regex
        tokens = [self.vocab_map[s] for s in self.regex_pattern.findall(" " + text)]
        
        # Byte-Pair Encoding (BPE) merge loop:
        # Sequentially merges the most frequent adjacent token pairs.
        while True:
            best_score, best_id, best_idx = -1e10, -1, -1
            # Find the best pair to merge
            for i in range(len(tokens) - 1):
                merged_str = self.vocab[tokens[i]] + self.vocab[tokens[i+1]]
                if merged_str in self.vocab_map:
                    merged_id = self.vocab_map[merged_str]
                    if self.vocab_scores[merged_id] > best_score:
                        best_score, best_id, best_idx = self.vocab_scores[merged_id], merged_id, i
            
            if best_idx == -1: # No more merges possible
                break
            
            # Merge the best pair
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]

        if bos: tokens.insert(0, 1) # Add Beginning-Of-Sentence token
        if eos: tokens.append(2)  # Add End-Of-Sentence token
        return tokens

class Sampler:
    """Sampling utilities for selecting a token from logits."""
    def __init__(self, vocab_size: int, temp: float, top_p: float, seed: int):
        self.vocab_size, self.temp, self.top_p = vocab_size, temp, top_p
        self.rng = np.random.default_rng(seed)

    def sample(self, logits: np.ndarray) -> int:
        # Greedy sampling: always select the token with the highest probability
        if self.temp == 0.0: return np.argmax(logits)
        
        # Temperature scaling: modulates the sharpness of the probability distribution
        # Higher temperature -> more random; lower temperature -> more deterministic
        logits /= self.temp
        # Apply softmax to convert logits to probabilities
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        
        # Top-p (nucleus) sampling: considers only the smallest set of tokens whose
        # cumulative probability exceeds the threshold 'top_p'.
        if self.top_p >= 1.0: 
            # Sample from the full distribution if top_p is not used
            return self.rng.choice(self.vocab_size, p=probs)
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        # Find the cutoff index where cumulative probability exceeds top_p
        cutoff_idx = np.searchsorted(cumulative_probs, self.top_p)
        
        # Truncate the distribution to the nucleus
        truncated_probs = sorted_probs[:cutoff_idx + 1]
        truncated_probs /= np.sum(truncated_probs) # Renormalize
        # Sample from the truncated distribution
        return self.rng.choice(sorted_indices[:cutoff_idx + 1], p=truncated_probs)

def run(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, prompt: str, steps: int):
    """Main generation loop."""
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    token = prompt_tokens[0]
    pos, start_time = 0, 0.0

    while pos < steps:
        logits = transformer.forward(token, pos)
        next_token = prompt_tokens[pos + 1] if pos < len(prompt_tokens) - 1 else sampler.sample(logits)
        if next_token == 1: break
        
        print(tokenizer.decode(token, next_token), end='', flush=True)
        token, pos = next_token, pos + 1
        if start_time == 0.0 and pos > len(prompt_tokens) -1: start_time = time.time()

    if pos > 1 and start_time > 0:
        print(f"\n\nachieved tok/s: {(pos - len(prompt_tokens)) / (time.time() - start_time):.2f}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Concise Llama-2-style Transformer inference.")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint file (.bin)")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-p", "--topp", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("-s", "--seed", type=int, default=int(time.time()), help="Random seed")
    parser.add_argument("-n", "--steps", type=int, default=256, help="Number of steps to generate")
    parser.add_argument("-i", "--prompt", type=str, default="", help="Input prompt")
    parser.add_argument("-z", "--tokenizer", type=str, default="data/tok128.bin", help="Tokenizer file")
    
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint) or not os.path.exists(args.tokenizer):
        sys.exit("Error: Checkpoint or tokenizer file not found.")

    with open(args.checkpoint, 'rb') as f:
        config = Config.from_file(f)
    
    steps = args.steps if 0 < args.steps <= config.seq_len else config.seq_len
    
    transformer = Transformer(config, args.checkpoint)
    tokenizer = Tokenizer(args.tokenizer, config.vocab_size)
    sampler = Sampler(config.vocab_size, args.temperature, args.topp, args.seed)
    run(transformer, tokenizer, sampler, args.prompt, steps)

if __name__ == "__main__":
    main()