# QF (Quick Feedforward) Learning Framework

A rapid fine-tuning framework for large language models, implementing efficient parameter updates through matrix operations without gradient back propagation nor loss function.

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
  url={https://doi.org/10.20944/preprints202507.2318.v1}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
