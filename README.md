# modded-llama2.c

`modded-llama2.c` is a streamlined and enhanced version of [BabyLlama](https://github.com/EN10/BabyLlama), designed for simplicity and power. It runs seamlessly in Jupyter notebooks, making it easy to train and experiment with LLaMA 2 models. This project builds upon the solid foundation of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and integrates ideas from [nanochat](https://github.com/karpathy/nanochat) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

The primary focus is on providing an accessible yet powerful platform for training language models on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). Whether you're a researcher, a student, or a hobbyist, this repository offers a hands-on approach to understanding and building modern LLMs.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Quick Start: The `Baby_Llama_128.ipynb` Notebook](#quick-start-the-baby_llama_128ipynb-notebook)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Dataset: TinyStories](#dataset-tinystories)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Simplified implementation designed to run in Jupyter notebooks or standalone Python.
- Pure C implementation for maximum performance and minimal dependencies for inference.
- Efficient training with gradient accumulation and mixed precision.
- Cosine learning rate scheduling with warmup.
- Gradient clipping and optimization techniques.
- Checkpointing and model export capabilities.
- The primary `train.py` script is a simplified single-process version. Advanced features like DDP or comprehensive WandB integration might require custom modifications.

### Key Improvements from nanochat / modded-nanogpt

<details>
<summary>Click to expand for details</summary>

- **Architecture Improvements (via `model.py` modifications)**
  - **relu² activation** (from nanochat) — simpler and faster than SwiGLU, uses 2 linear layers instead of 3
  - **No-parameter RMSNorm** (from nanochat) — purely functional normalization with no learnable weights
  - Zero-initialized projections for better convergence
  - Padded vocabulary head
- **Training Improvements**
  - Parameter-grouped AdamW optimizer with different learning rates for:
    - Embedding layers
    - Hidden matrix weights
  - Improved learning rate schedule: stable period followed by cosine decay
  - Efficient gradient accumulation with mixed precision
  - TinyStories dataset focus, with flexible vocabulary options (e.g., Llama 2 default, or smaller custom vocabularies like 128 tokens).
</details>

## Getting Started

### Quick Start: The `Baby_Llama_128.ipynb` Notebook

For a complete, hands-on example of the entire workflow, the [`Baby_Llama_128.ipynb`](Baby_Llama_128.ipynb) notebook is the recommended starting point.

The notebook will guide you through:
1.  **Environment Setup**: It clones the repository and installs dependencies.
2.  **Data Download**: It runs a script to download the pre-tokenized TinyStories dataset, a custom 128-token SentencePiece model, and compiles the C code for inference.
3.  **Model Training**: It trains a small LLaMA model with a 128-token vocabulary for 1 iteration.
4.  **Inference**: It demonstrates how to run inference with the newly trained model to generate text.

This notebook is a self-contained example that showcases the project's capabilities from data preparation to generation.

## Usage

### Data Preparation

All data preparation, including downloading and pre-tokenization, is handled by the `tinystories.py` script. For detailed instructions on how to prepare the TinyStories dataset for training, please refer to the [`TinyStories.md`](TinyStories.md#data-preparation-and-pre-tokenization) file.

### Training

The `train.py` script is used for training. Optimizer state is not resumed by default when using `init_from="resume"`.

**Example command (custom 128-token vocabulary):**
```bash
python train.py 
  --out_dir=out/my_tinystory_model 
  --vocab_source=custom 
  --vocab_size=128 
  --dim=288 
  --n_layers=6 
  --n_heads=6 
  --n_kv_heads=6 
  --batch_size=32 
  --gradient_accumulation_steps=4 
  --base_learning_rate=5e-4 
  --max_iters=2000 
  --eval_interval=100 
  --always_save_checkpoint=True
```

<details>
<summary><b>Key Training Parameters</b></summary>

| Parameter                       | Description                                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--out_dir`                     | Output directory for checkpoints and the final `model.bin`.                                                                            |
| `--vocab_source`                | `llama2` (default) or `custom`. If `custom`, ensure you've run `tinystories.py train_vocab` and `pretokenize`.                           |
| `--vocab_size`                  | Vocabulary size. Must match the pretokenized data (e.g., 32000 for `llama2`).                                                          |
| `--dim`, `--n_layers`, etc.     | Model dimensions.                                                                                                                      |
| `--batch_size`                  | Batch size for training.                                                                                                               |
| `--gradient_accumulation_steps` | Number of steps for gradient accumulation.                                                                                             |
| `--base_learning_rate`          | Initial learning rate.                                                                                                                 |
| `--max_iters`                   | Maximum number of training iterations.                                                                                                 |
| `--cooldown_frac`               | Fraction of training to spend cooling down LR.                                                                                         |
| `--compile`                     | Set to `True` to attempt PyTorch model compilation (requires PyTorch 2.0+).                                                            |

</details>

<details>
<summary><b>Chinchilla-Optimal Training</b></summary>

According to the DeepMind "Chinchilla" paper, compute-optimal training is achieved by using approximately **20 tokens of training data for every parameter** in the model. For the default model in this project:

-   **Model Parameters**: ~0.96 Million
-   **Optimal Tokens**: `0.96M params * 20 tokens/param = 19.2M tokens`
-   **Tokens per Iteration**: `32 (batch) * 512 (seq_len) = 16,384 tokens`

The Chinchilla-optimal number of iterations is therefore:
`19,200,000 / 16,384 ≈ 1,167 iterations`

This suggests that for the default model size, training for around **1,167 iterations** would provide the best performance for the given compute budget.

</details>

### Inference

To run inference with a trained model, use the compiled `run` executable.

```bash
# First, compile the C code if you haven't already
gcc -O3 -o run run.c -lm

# Run inference
./run out/my_tinystory_model/model.bin -z data/tok128.bin -t 0.8 -n 256 -i "Once upon a time"
```
**Parameters:**
-   `model.bin`: Path to the trained model weights.
-   `-z <tokenizer_path>`: Path to the C-level tokenizer file (e.g., `data/tok128.bin`).
-   `-t <temperature>`: Sampling temperature (e.g., 0.8).
-   `-n <steps>`: Number of tokens to generate.
-   `-i <prompt>`: Input prompt.

## Model Architecture

The implementation follows the LLaMA 2 architecture with:
- No-parameter RMSNorm for layer normalization
- RoPE positional embeddings
- relu² activation function (from nanochat)
- Grouped-Query Attention (GQA)

<details>
<summary><b>Default Configuration (~0.96M Parameters)</b></summary>

| Parameter      | Value |
| -------------- | ----- |
| `dim`          | 128   |
| `n_layers`     | 5     |
| `n_heads`      | 8     |
| `vocab_size`   | 128   |

**Parameter Breakdown:**
-   **Token Embeddings**: ~16k
-   **Transformer Blocks (5 layers)**: ~923k
-   **Final Output Layer**: ~16k
-   **Total**: **~0.96 Million Parameters**

</details>

## Dataset: TinyStories

This project uses the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), a collection of short stories generated by language models, ideal for training smaller models.

For a detailed explanation of the data, how it's structured for training, and what constitutes a batch, please see the [`TinyStories.md`](TinyStories.md) file.

<details>
<summary><b>Training Iterations per Epoch</b></summary>

An "epoch" in this context refers to one full pass over the training data.

-   **Total Tokens in `data00.bin`**: 57,979,674
-   **Default Batch Size**: 32
-   **Default Sequence Length**: 512
-   **Tokens per Iteration**: `32 * 512 = 16,384`

Therefore, the total number of training iterations for one epoch is:
`57,979,674 / 16,384 ≈ 3,538 iterations`

</details>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BabyLlama](https://github.com/EN10/BabyLlama) by EN10 - This project is an updated version of BabyLlama
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan
