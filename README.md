# QF (Quick Fine-tuning) Learning Framework

A rapid fine-tuning framework for large language models based on ChatGLM3, implementing efficient parameter updates through matrix operations.

## Overview

QF Learning is an innovative approach to quickly adapt large language models to new information without traditional fine-tuning. It uses matrix operations to compute updated weight matrices (`W_prime`) that can be applied during inference to incorporate new knowledge.

## Features

- **Rapid Learning**: Update model knowledge in seconds instead of hours
- **Selective Updates**: Apply significance weights to different parts of responses
- **Memory Efficient**: Only stores necessary intermediate tensors
- **Compatible**: Works with Qwen2.5-1.5B-Instruct and similar models
- **Continual Learning**: Support for multiple rounds of knowledge updates

## Architecture

The framework operates on specific transformer layers (default: layer 23) and computes updated weight matrices using the formula:

```
W' = W - [W(Y - X) - B] (Y^T Y)^{-1} Y^T
```

Where:
- `W`: Original weight matrix
- `X`: Input activations (u)
- `Y`: Target activations (up) 
- `B`: Output difference (v - vp)

## Installation

### Prerequisites

- Python 3.10
- PyTorch 2.3.1+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd QF
```

2. Install PyTorch with CUDA support:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

3. Install accelerate:
```bash
pip install accelerate==1.8.1
```

4. Install the modified transformers library:
```bash
cd transformers-4.42-release
pip install .
cd ..
```

5. Install other dependencies:
```bash
pip install numpy
```

6. Download the base model:
```bash
# The script expects Qwen2.5-1.5B-Instruct at:
# /home/ls/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct
```

## Usage

### Basic Usage

Run the main learning script:

```bash
python qf_learn_simple.py
```

### Configuration

Key parameters in `qf_learn_simple.py`:

- `model_path`: Path to the base model
- `qflayer`: Target transformer layer (default: 23)
- `qffolder`: Directory for storing parameters
- `qfseed`: Random seed for reproducibility

### Learning Modes

The framework supports several modes:

1. **QF-infer-w**: Inference with original weights
2. **QF-infer-wp**: Inference with updated weights
3. **QF-instruct**: Instruction phase for learning
4. **QF-update**: Update phase for computing new weights

### Example Learning Process

```python
# First round learning
system = "Qi started Oxinnovate."
user = "Who started Oxinnovate?"
assistant = "Oxinnovate was started by Qi."
qfsignificance = [0, 1, 1, 1, 1]  # Weight different words

# Apply learning
qf_response(model, tokenizer, system, user, assistant, 
           qfmode='QF-instruct', qfsignificance=qfsignificance)
qf_response(model, tokenizer, "", user, assistant, 
           qfmode='QF-update', qfsignificance=qfsignificance)

# Compute updated weights
W_prime, W = calc_this_w_prime()
```

## File Structure

```
QF/
├── qf_learn_simple.py      # Main learning script
├── parameters/             # Stored model parameters
│   ├── layer23_u.pt       # Input activations
│   ├── layer23_up.pt      # Target activations  
│   ├── layer23_v.pt       # Output activations
│   ├── layer23_vp.pt      # Target outputs
│   ├── layer23_weight.pt  # Original weights
│   └── layer23_w_prime.pt # Updated weights
├── transformers-4.42-release/  # Modified transformers library
└── result.md             # Learning results log
```

## Key Functions

### `compute_W_prime(W, u, up, v, vp)`
Computes the updated weight matrix using matrix operations.

### `calc_this_w_prime()`
Loads tensors and computes W_prime for the specified layer.

### `qf_assistant_process(words, qfsignificances, tokenizer)`
Processes assistant responses with significance weights.

### `qf_response(model, tokenizer, system, user, assistant, ...)`
Main function for generating responses with different QF modes.

## Results

The framework demonstrates successful knowledge learning:

- **Before learning**: Model cannot answer "Who started Oxinnovate?"
- **After learning**: Model correctly responds "Oxinnovate was started by Qi"
- **Continual learning**: Model can learn additional facts about the same entity

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{qf_learning_2025,
  title={QF: Quick Feedforward AI Model Training without Gradient Back Propagation},
  author={Feng Qi},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub. 
