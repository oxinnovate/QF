# QF (Quick Feedforward) Learning Framework

We propose Quick Feedforward (QF) Learning, a novel knowledge consolidation framework for transformer-based models that enables efficient transfer of instruction-derived knowledge into model weights through feedforward activations—without any gradient backpropagation. Unlike traditional fine-tuning, QF updates are computed in closed form, require minimal parameter modification, and preserve prior knowledge. Importantly, QF allows models to train and infer within the same runtime environment, making the process more resource-efficient and closely aligned with how the human brain operates. Code and models are open sourced on GitHub. I hope QF Learning inspires a more efficient and brain-like paradigm for AI systems.


<img width="960" height="668" alt="image" src="https://github.com/user-attachments/assets/502df78e-e54d-4ded-8341-095b40a98c2b" />

## Overview

QF (Quick Feedforward) Learning is an innovative approach to quickly adapt large language models to new information without traditional fine-tuning. It uses matrix operations to compute updated weight matrices (`W_prime`) that can be applied during inference to incorporate new knowledge.


## Installation

### Prerequisites

- Python 3.10
- PyTorch 2.3.1+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/oxinnovate/QF
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
# Qwen/Qwen2.5-1.5B-Instruct   Modelscope or HuggingFace
```

## Usage

### Basic Usage

Run the main learning script:

```bash
python qf_learn_simple.py
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{qf_learning_2025,
  title={QF: Quick Feedforward AI Model Training without Gradient Back Propagation},
  author={Feng Qi},
  year={2025},
  cite={https://arxiv.org/abs/2507.04300}
}
@misc{qf2_learning_2025,
  title={QF2: Quick Firing Model Weight Updates},
  author={Feng Qi},
  year={2025},
  url={https://doi.org/10.36227/techrxiv.175417357.76989782/v1}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
