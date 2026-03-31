# TinyStories Dataset Info

This document provides a comprehensive guide to the TinyStories dataset as used in this project. It covers the data source, file structure, content, and the pre-processing steps required for training a language model, from downloading the raw data to pre-tokenizing it for training.

## Table of Contents

- [Data Source](#data-source)
- [File Structure](#file-structure)
- [Entry Structure](#entry-structure)
- [Story Size](#story-size)
- [Data Preparation and Pre-tokenization](#data-preparation-and-pre-tokenization)
- [Token Count](#token-count)

This document explains the data from the [TinyStories dataset on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories), providing a brief overview of the structure and content of the `data00.json` file.

## Data Source

It was downloaded using `wget`:
```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz
```

And extracted with `tar`:
```bash
tar -I pigz -xvf TinyStories_all_data.tar.gz ./data{00..01}.json
```

## File Structure

The file `data00.json` contains a JSON array. We can count the number of entries using `jq`:
```bash
jq 'length' data00.json
```
This shows there are **100,000** entries. Each entry is a JSON object representing a single data record.

## Entry Structure

Each object in the array has four keys. We can see them by inspecting the first entry:
```bash
jq '.[0] | keys' data00.json
```
The keys are:

*   `story`: (String) A short story, often written with simple language.
*   `instruction`: (Object) Contains the generation parameters for the story.
    *   `prompt`: (String) The detailed prompt used to generate the story, including constraints on vocabulary, length, and features.
    *   `words`: (Array of Strings) A list of specific words that must be included in the story.
    *   `features`: (Array of Strings) A list of specific literary features the story must contain (e.g., "Dialogue").
*   `summary`: (String) A brief one-sentence summary of the story.
*   `source`: (String) The source of the data entry (e.g., "GPT-4").

### Example Entry

We can view the first full entry with:
```bash
jq '.[0]' data00.json
```

```json
{
  "story": "\n\nLily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n\"Push me, Ben!\" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n\"Can I swing too, Lily?\" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n\"Ow, ow, ow!\" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben's hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.",
  "instruction": {
    "prompt:": "Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb \"hang\", the noun \"foot\" and the adjective \"cute\". The story has the following features: the story should contain at least one dialogue. Remember to only use simple words!\n\nPossible story:",
    "words": [
      "hang",
      "foot",
      "cute"
    ],
    "features": [
      "Dialogue"
    ]
  },
  "summary": "Lily and Ben play in the park and Lily gets too caught up in swinging, causing Ben to leave. Lily falls off the swing and hurts herself, but Ben leaves his hat for her as a kind gesture.",
  "source": "GPT-4"
}
```

## Story Size

The average size of a story can be calculated using `jq` and `awk`:

```bash
jq '.[] | .story | length' data00.json | awk '{ total += $1; count++ } END { print "Average story size:", total/count }'
```

This gives an average story size of approximately **775 characters**.

### Average Story Length (Words)

Based on a sample of 10,000 stories from the dataset:

| Metric        | Value     |
|---------------|-----------|
| Average       | **169 words** |
| Median        | 151 words |
| Min           | 43 words  |
| Max           | 878 words |
| Std deviation | 71 words  |

### Total Story Size

The total number of characters in all stories can be calculated with a simple Python script.

First, create a file named `count_chars.py`:
```python
import json

def count_story_chars(file_path):
    total_chars = 0
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            if 'story' in item:
                total_chars += len(item['story'])
    return total_chars

if __name__ == "__main__":
    file_path = 'data00.json'
    total_characters = count_story_chars(file_path)
    print(f"The total number of characters in all stories is: {total_characters}")
```

Then, run the script:
```bash
python3 count_chars.py
```

This gives a total of **77,486,885 characters**.

### What data is used for training?

Only the text from the `story` key is used. The other fields, like `instruction` and `summary`, are ignored during training.

### How is the data structured for training?

During pre-tokenization, all stories are concatenated into a single, continuous stream of tokens. The model is trained on this long sequence, without regard for where individual stories begin or end.

### What is a training batch?

A batch does **not** consist of a set number of stories. Instead, it is a collection of smaller chunks sliced from the continuous token stream. For example, with `batch_size=32` and `max_seq_len=512`, a single batch consists of 32 parallel sequences, each 512 tokens long. A sequence within a batch can therefore contain the end of one story and the beginning of another. This approach ensures efficient training on large text corpora.

## Data Preparation and Pre-tokenization

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

*   **Option B: Training a custom SentencePiece tokenizer and pretokenizing:**
    For example, to train a 128-token vocabulary:
    ```bash
    # 1. Train a custom tokenizer with a 128-token vocabulary
    python tinystories.py train_vocab --vocab_size=128
    
    # 2. Pretokenize the data using the new tokenizer
    python tinystories.py pretokenize --vocab_size=128
    ```
    When training with this custom tokenizer, you'll need to specify `--vocab_source=custom` and `--vocab_size=128` to `train.py`.

The pre-tokenization process converts the raw `.json` files into binary (`.bin`) files that are used for training. It performs the following steps:

1.  Reads each `dataXX.json` file.
2.  Extracts the `story` text from each entry.
3.  Tokenizes the text into a sequence of integers using the specified tokenizer.
4.  Adds a special "beginning of sentence" (BOS) token to the start of each tokenized story.
5.  Concatenates all tokenized stories from a single `.json` file into one long sequence.
6.  Saves this sequence as a binary file (e.g., `data00.bin`).

These `.bin` files are the final pre-processed data used for training the language model.

## Token Count

The number of tokens in a `.bin` file can be calculated by dividing the file size in bytes by 2 (since each token is a `uint16`).

A simple Python script `count_tokens.py` can do this:
```python
import os

# filepath: count_tokens.py
file_path = 'data00.bin'
# Each token is a uint16, which is 2 bytes
token_size_in_bytes = 2

try:
    file_size_in_bytes = os.path.getsize(file_path)
    num_tokens = file_size_in_bytes // token_size_in_bytes
    print(f"The file '{file_path}' contains {num_tokens:,} tokens.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please generate it first.")
except Exception as e:
    print(f"An error occurred: {e}")
```

Running this script on `data00.bin`:
```bash
python3 count_tokens.py
```
Shows that `data00.bin` contains **57,979,674** tokens.
