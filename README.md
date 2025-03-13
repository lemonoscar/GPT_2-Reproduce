# GPT-2 (124M) Reproduction in PyTorch

This repository is dedicated to reproducing the GPT-2 (124M) model as demonstrated in Andrej Karpathy's YouTube video ["Let’s reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=MutbZKX2jAE). Unlike the original GPT-2 implementation by OpenAI, which used TensorFlow, this project leverages PyTorch for a cleaner and more modern approach to building and training the model from scratch.

The goal of this project is to document my step-by-step experience in replicating the GPT-2 (124M) model, including challenges, insights, and solutions. All experiments, code, and notes are shared via Jupyter notebooks for transparency and educational purposes.

## Project Motivation

Andrej Karpathy's video provides a comprehensive guide to building GPT-2 (124M) from an empty file to a fully functional model. Inspired by his approach, I aim to:
- Understand the inner workings of the GPT-2 architecture.
- Adapt the implementation from TensorFlow to PyTorch.
- Record the process for others to learn from my journey.

## Repository Contents

- **`notebooks/`**: Contains Jupyter notebooks documenting the step-by-step process of building and training the GPT-2 (124M) model.
- **`src/`**: Source code for the model architecture, training loop, and utilities (e.g., tokenization, data loading).
- **`data/`**: Scripts or instructions for downloading and preparing the dataset (e.g., FineWeb or OpenWebText).
- **`config/`**: Configuration files for model hyperparameters and training settings.
- **`results/`**: Logs, checkpoints, and generated outputs from training runs.
- `README.md`: This file, providing an overview and instructions.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU training)
- Additional dependencies: `numpy`, `transformers`, `datasets`, `tiktoken`, `tqdm`

## Progress

Here’s the current status of the project, tracked as a checklist:

- [ ] **Model Architecture**: Implement the GPT-2 (124M) architecture in PyTorch.
- [ ] **Data Preparation**: Tokenize and preprocess the dataset (e.g., FineWeb or OpenWebText).
- [ ] **Training Setup**: Set up the training loop with PyTorch, including loss function and optimizer.
- [ ] **Training & Evaluation**: Train the model and evaluate performance using benchmarks like HellaSwag.
- [ ] **Text Generation**: Generate sample text and analyze the quality of outputs.

Each milestone will be documented in detail within the Jupyter notebooks.

## Repository Structure

'''text
The repository is organized as follows:
GPT-2-Reproduce/
│
├── notebooks/                  # Jupyter notebooks documenting the process
│   ├── 01_data_preprocessing.ipynb    # Dataset preparation steps
│   ├── 02_model_implementation.ipynb  # Model architecture implementation
│   ├── 03_training.ipynb              # Training loop and experiments
│   └── 04_evaluation_and_generation.ipynb  # Evaluation and text generation
│
├── src/                        # Core source code
│   ├── model.py                # GPT-2 model definition
│   ├── train.py                # Training script with main loop
│   ├── utils.py                # Utility functions (e.g., data loading, logging)
│   └── generate.py             # Script for text generation
│
├── data/                       # Data handling
│   ├── prepare.py              # Script to download and preprocess data
│   └── README.md               # Instructions for dataset setup
│
├── config/                     # Configuration files
│   └── train_gpt2.py           # Hyperparameters for training
│
├── results/                    # Output directory
│   ├── logs/                   # Training logs (e.g., loss curves)
│   ├── checkpoints/            # Saved model weights
│   └── samples/                # Generated text samples
│
├── requirements.txt            # List of Python dependencies
├── README.md                   # Project overview (this file)
└── LICENSE                     # MIT License file
'''

This structure separates code, documentation, and outputs for clarity and ease of use.
