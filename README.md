# modded-llama2.c

An updated version of BabyLlama that runs in a Jupyter notebook, building on llama2.c and incorporating training improvements from modded-nanogpt. Focuses on training with the TinyStories dataset.

## Overview

This project is:
- An updated version of [BabyLlama](https://github.com/EN10/BabyLlama) - Simplified LLaMA for Jupyter notebooks
- Based on [llama2.c](https://github.com/karpathy/llama2.c) - Pure C implementation of LLaMA 2
- Incorporates training improvements from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- Focuses on efficient training with the TinyStories dataset

## Key Improvements from modded-nanogpt

### Architecture Improvements (via model.py modifications)
- QK-Normalization for improved training stability
- Zero-initialized projections for better convergence
- Untied & padded vocabulary head
- Logit Softcapping to control output distributions

### Training Improvements
- Parameter-grouped AdamW optimizer with different learning rates for:
  - Embedding layers
  - Hidden matrix weights
  - Norm/scalar parameters
  - Head layer
- Improved learning rate schedule: stable period followed by cosine decay
- Efficient gradient accumulation with mixed precision
- TinyStories dataset focus, with flexible vocabulary options (e.g., Llama 2 default, or smaller custom vocabularies like 128 tokens as seen in examples).

## Features

- Simplified implementation designed to run in Jupyter notebooks or standalone Python
- Pure C implementation for maximum performance and minimal dependencies for inference
- Efficient training with gradient accumulation and mixed precision
- Cosine learning rate scheduling with warmup
- Gradient clipping and optimization techniques
- Checkpointing and model export capabilities
- The primary `train.py` script is a simplified single-process version. Advanced features like DDP or comprehensive WandB integration might require custom modifications.

## Requirements

- C compiler (gcc/clang)
- CUDA toolkit (for GPU support)
- Python 3.8+ (for training script and data preparation)
- PyTorch
- SentencePiece (for custom vocabulary training via `tinystories.py`)

## Building

```bash
make
```

## Data Preparation (using tinystories.py)

The `tinystories.py` script handles dataset downloading and tokenization.

**1. Download TinyStories data:**
```bash
python tinystories.py download
```

**2. Pretokenize the data:**

*   **Option A: Using the default Llama 2 tokenizer (vocab size ~32000):**
    This is the default for `train.py` if `--vocab_source` is not specified or set to `llama2`.
    ```bash
    python tinystories.py pretokenize
    ```
    The training script will expect `vocab_size` to match the Llama 2 tokenizer's vocabulary size.

*   **Option B: Training a custom SentencePiece tokenizer and pretokenizing:**
    For example, to train a 128-token vocabulary:
    ```bash
    python tinystories.py train_vocab --vocab_size=128
    python tinystories.py pretokenize --vocab_size=128
    ```
    When training with this custom tokenizer, you'll need to specify `--vocab_source=custom` and `--vocab_size=128` to `train.py`. The C-level tokenizer file will be `data/tok128.bin`. Note that the [`Baby_Llama_128.ipynb`](Baby_Llama_128.ipynb) example downloads pre-built tokenizer files for a 128-token vocabulary, including [`tok128.vocab`](https://huggingface.co/datasets/enio/TinyStories/blob/main/tok128/tok128.vocab) and `tok128.model`, from Hugging Face to generate `data/tok128.bin`.

## Training

The `train.py` script is highly simplified for single-process runs. Optimizer state is not resumed by default when using `init_from="resume"`.

To train a model (example using a custom 128-token vocabulary):

```bash
python train.py \
  --out_dir=out/my_tinystory_model \
  --vocab_source=custom \
  --vocab_size=128 \
  --dim=288 \
  --n_layers=6 \
  --n_heads=6 \
  --n_kv_heads=6 \
  --batch_size=32 \
  --gradient_accumulation_steps=4 \
  --base_learning_rate=5e-4 \
  --max_iters=2000 \
  --eval_interval=100 \
  --always_save_checkpoint=True
```

Key training parameters:
- `--out_dir`: Output directory for checkpoints and the final `model.bin`.
- `--vocab_source`: `llama2` (default) or `custom`. If `custom`, ensure you've run `tinystories.py train_vocab` and `pretokenize` for the specified `--vocab_size`.
- `--vocab_size`: Vocabulary size. Must match the pretokenized data. (e.g., 32000 for `llama2`, or your custom size).
- `--dim`, `--n_layers`, `--n_heads`, `--n_kv_heads`: Model dimensions.
- `--batch_size`: Batch size for training.
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation.
- `--base_learning_rate`: Initial learning rate.
- `--max_iters`: Maximum number of training iterations.
- `--cooldown_frac`: Fraction of training to spend cooling down LR.
- `--compile`: Set to `True` to attempt PyTorch model compilation (requires PyTorch 2.0+). Default is `False` in the simplified script if not otherwise configured.

Refer to `Baby_Llama_128.ipynb` for a practical example of training a small model.

## Inference

To run inference with a trained model (e.g., the one from the training example above):

```bash
./run out/my_tinystory_model/model.bin -z data/tok128.bin -t 0.8 -n 256 -i "Once upon a time"
```
Parameters:
- First argument: path to the `model.bin` file.
- `-z <tokenizer_path>`: Path to the C-level tokenizer file (e.g., `data/tok<vocab_size>.bin` generated by `tokenizer.py` based on your SentencePiece model, or `data/llama2_tokenizer.bin` if using Llama 2 tokenizer).
- `-t <temperature>`: Temperature for sampling (e.g., 0.8).
- `-n <steps>`: Number of tokens to generate.
- `-i <prompt>`: Input prompt.

## Model Architecture

The implementation follows the LLaMA 2 architecture with:
- RMSNorm for layer normalization
- RoPE positional embeddings
- SwiGLU activation function
- Grouped-Query Attention (GQA)
- Sliding window attention

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [BabyLlama](https://github.com/EN10/BabyLlama) by EN10 - This project is an updated version of BabyLlama
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan