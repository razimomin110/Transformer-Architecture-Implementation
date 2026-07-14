# Transformer Architecture

This repository contains a PyTorch implementation of the Transformer model, built from scratch following the architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The model is designed for sequence-to-sequence tasks and is currently configured for **English-to-Bengali** translation.

## 📂 Files

- `transformer_architecture.py`: Contains the `Transformer` model architecture (Encoder, Decoder, Attention mechanisms).
- `main.py`: The entry point script that handles data loading, training, and testing.
- `transformer_explained.ipynb`: A Jupyter Notebook containing the full implementation with descriptive explanations of each component.
- `data.csv`: A custom dataset containing English-Bengali sentence pairs.

## 🚀 Getting Started

### Prerequisites

- Python 3.13.x
- **PyTorch**: This is the main dependency. CUDA is recommended for faster training. You can install it via pip:

For CPU only:

```bash
pip install torch
```

For GPU (CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Installation & Setup

1.  Clone this repository or download the files.
2.  Ensure `transformer_architecture.py`, `main.py`, and `data.csv` are in the same directory.

## 🏃 Usage

Run the `main.py` script to train the model and test it:

```bash
python main.py
```

### What happens when you run it?
1.  **Data Loading**: Loads the English-Bengali pairs from `data.csv`.
2.  **Vocabulary Building**: Creates a vocabulary mapping for both languages.
3.  **Training**: Trains the Transformer model for **500 epochs**.
    - *Note: Training logs will print loss every 50 epochs.*
4.  **Testing**: After training, the model attempts to translate an unseen sentence:
    - *Input*: "rohan bought a new computer"
    - The script prints the predicted Bengali output and the expected output.

## 🧠 Model Architecture

The implementation includes the following key components from scratch:

- **SelfAttention**: Multi-head attention mechanism.
- **TransformerBlock**: The encoder block containing self-attention and feed-forward networks with residual connections.
- **Encoder**: Stacks multiple Transformer blocks. Uses learned positional embeddings.
- **DecoderBlock**: The decoder block with masked self-attention (to prevent peeping into the future) and cross-attention (attending to encoder outputs).
- **Decoder**: Stacks multiple Decoder blocks.
- **Masking**: Implements both source padding masks and target causal masks (look-ahead masks).

## 📊 Dataset Format

The `data.csv` file is formatted with English sentences in the first column and Bengali sentences in the second column.

**Example:**
```csv
arjun is driving a red car,অর্জুন একটি লাল গাড়ি চালাচ্ছে
rohan is driving a blue car,রোহন একটি নীল গাড়ি চালাচ্ছে
```

*(Note: The current script is set up to skip the first row of the CSV, assuming it might be a header.)*

## 🔧 Hyperparameters

The hyperparameters are significantly lowered compared to the original ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. This is because we are training on a **tiny dataset** (approx. 50 sentence pairs), and the original model would drastically overfit.

| Hyperparameter | Original Paper | My Implementation |
| :--- | :--- | :--- |
| **Embedding Size** | 512 | 64 |
| **Layers (Encoder/Decoder)** | 6 | 2 |
| **Attention Heads** | 8 | 4 |
| **Forward Expansion** | 4 | 2 |
| **Dropout** | 0.1 | 0.1 |

**Why lower the parameters?**
- **Overfitting**: A large model (millions of parameters) would memorize the small dataset instantly but fail to generalize.
- **Speed**: Smaller dimensions (`embed_size=64`) allow for extremely fast training on CPUs or standard GPUs.
- **Simplicity**: Demonstrating the architecture works even with a minimal configuration.

